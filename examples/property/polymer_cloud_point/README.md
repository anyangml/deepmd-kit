# Polymer cloud-point — weighted embedding-pool property head

This example finetunes a pretrained **DPA-3.3** multitask descriptor to predict a
polymer's cloud point, using the custom `polymer_pool` property fitting
(`deepmd/pt/model/task/polymer_pool.py`).

## Why a custom fitting

The native `property` fitting applies the readout MLP **per atom** and then
reduces (sum/mean) — i.e. it is *additive* over atoms. A polymer's cloud point is
**not** additive over its monomers: comonomer interactions and coupling with the
process conditions matter. So we pool the per-fragment **embeddings** first and run
one head on the pooled representation together with the process parameters:

```
E_r = Σ_i w_i · g_i     over atoms of role r        (w_i, role in aparam)
ŷ   = head( concat[ norm(E) , fparam ] )            (fparam = process params)
```

Implementation trick: the fitting computes `ŷ` for the frame and **broadcasts it to
every atom**, so with `intensive: true` the model's mean atom-reduce returns `ŷ`
unchanged. Nothing else in `PropertyModel` / the reduce stack changes, and the
stock `property` loss is used as-is.

## Data

Each polymer is **one frame = one system**; its fragments (end groups = role 0,
repeating units = role 1) are laid out > rcut apart so the descriptor embeds each
independently. Build it with the converter in the polymer project:

```bash
python polymer_to_dpdata.py --out dpdata_polymer \
    --type-map-file type_map.raw --split ood
# -> dpdata_polymer/{train,val,test_low,test_high}/poly_XXXXX/
```

Per-frame arrays: `coord`, `real_atom_types`, `aparam=[w, role0, role1]`
(`w = unit_weight / n_atoms_in_unit`, raw pooling weight; `numb_aparam = n_roles+1`),
`fparam` = 38 process params **pre-standardized on the train split** (DeePMD does
not compute fparam stats in finetune mode), and the label `cloud_point.npy`.

> **type_map** must match the checkpoint's 118-element periodic-order map. Extract
> it once from your ckpt (`model.get_type_map()`) and pass it as `--type-map-file`.

## Train

Point `training_data` / `validation_data` at the generated `train` / `val` dirs
(DeePMD discovers the ~591 / 104 sub-systems recursively).

**Frozen (linear probe)** — descriptor fixed, only the pooled head trains:

```bash
dp --pt train input_polymer_pool.json \
   --finetune /path/to/model.ckpt-6860000.pt --use-pretrain-script \
   --init-frz-model none   # set descriptor.trainable=false in the config
```

**End-to-end** — set `descriptor.trainable: true`, same command.

`--use-pretrain-script` copies the exact `dpa3` descriptor config from the ckpt
(the `descriptor` block here is only illustrative). Pick which branch's
descriptor/charge-spin embedding to inherit with `--model-branch <BRANCH>`
(e.g. an organic/molecular branch); a new `polymer_pool` fitting is always created
fresh (`new_fitting=True`).

## Test (OOD tails)

```bash
dp --pt test -m frozen_model.pth -s /path/to/dpdata_polymer/test_low   # low tail
dp --pt test -m frozen_model.pth -s /path/to/dpdata_polymer/test_high  # high tail
```

The split reuses the agreed OOD label-tail split (train 591 / val 104 / low 72 /
high 72). Compare against the reference `group_property_e2e.py` (embedding-pool head
+ same split) to check reproduction before moving to multitask + energy replay.

## Notes

- `pool_norm: layer` (default) because one-frame-per-system means one frame per step
  — BatchNorm over the batch would be degenerate. Use `batch` only with multi-frame
  systems.
- `backend-agnostic (.dp) serialize/deserialize` is not implemented for
  `polymer_pool`; use torch checkpoints (training) and `dp --pt freeze` (TorchScript).
