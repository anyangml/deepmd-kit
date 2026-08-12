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

## Additive ablation (`polymer_additive`)

The "why a custom fitting" argument above is a *claim*: that cloud point is not
additive over monomers. `deepmd/pt/model/task/polymer_additive.py` is the ablation
that tests it — same data, same weights, same reduce, but it projects **first** and
weights **after**, exactly like the energy fitting only with mole-fraction weights
instead of a plain sum:

```
ŷ = Σ_i w_i · f(g_i, fparam)          vs.   polymer_pool:  ŷ = head(concat[norm(Σ_i w_i·g_i per role), fparam])
```

Additive over fragments by construction, so it *cannot* represent comonomer
interaction. If it matches `polymer_pool`, the pooled head is not earning its
complexity and the accuracy is coming from the descriptor.

The **role one-hot is unused** here: in the additive form every role's contribution
lands in the same Σ, so there are no per-role channels to separate. `numb_aparam`
must still be `n_roles + 1` so the exact same `aparam.npy` feeds both fittings and
the configs stay swappable on one dataset.

Config-wise it swaps `head_neuron`/`head_activation`/`head_dropout`/`pool_norm` for
the base per-atom net (`neuron`/`activation_function`/`resnet_dt`) — there is no
pooled representation to normalise and no separate head, because the base
`filter_layers` *are* the readout.

Two non-obvious implementation choices (both documented in the module docstring):

- **`intensive: true`, not `false`.** The obvious route is `intensive: false` since
  the extensive reduce *is* a sum, but the property loss then divides pred *and*
  label by `natoms` (`deepmd/pt/loss/property.py`), rescaling the metric to per-atom
  units and reweighting each frame by `1/natoms` — and with one polymer per system
  `natoms` varies per frame. Instead the fitting emits `nloc·w_i·ŷ_i` and lets the
  same intensive mean-reduce return `Σ_i w_i·ŷ_i`. Loss, stats and config then match
  `polymer_pool` exactly, so the A/B is apples-to-apples.
- **`use_aparam_as_mask` internally.** `numb_aparam > 0` is needed so the data
  pipeline loads `aparam.npy`, but the base fitting would then standardize aparam and
  concatenate it per atom — feeding the composition weight `w` into the net as a
  *feature*, which breaks additivity in `w`. This flag keeps aparam loaded but out of
  the input.

Unlike `polymer_pool` (which overrides `forward` outright, so `bias_atom_e` is never
applied), this fitting goes through the base path and the per-type bias **is**
applied, then scaled by `w` — giving `b · Σ_i w_i`. That is a clean global offset
**iff the pooling weights sum to 1 over the frame**, as mole fractions do; it is what
the intensive stat solve assumes. Data whose weights do not sum to 1 breaks the
correspondence.

Swap `"type": "polymer_pool"` → `"polymer_additive"` in any of the configs above, or
use the ready-made multitask pair:

```bash
dp --pt train input_multitask_replay_additive.json \
   --finetune /path/to/model.ckpt-6860000.pt --use-pretrain-script
```

`input_multitask_replay_additive.json` is identical to `input_multitask_replay.json`
except for `model_dict.polymer.fitting_net` — same shared descriptor, same energy
replay branch and `finetune_head`, same `dim_case_embd: 23`, same `model_prob`, same
lr/steps — so `diff`ing the two shows exactly the ablated surface.

## From-scratch baseline (control arm)

`input_polymer_pool_scratch.json` trains the **same** `polymer_pool` head on the
**same** DPA-3.3 descriptor *architecture*, but from a random init — no `--finetune`.
It is the control that makes "pretraining / multitask replay helps" a measurable
claim rather than an assertion: without it, a good finetune number could just mean
the task is easy.

```bash
dp --pt train input_polymer_pool_scratch.json
```

No `--finetune`, no `--use-pretrain-script`, no `--model-branch` — so unlike the
finetune configs, **the `descriptor` block here is load-bearing** (nothing overwrites
it from a ckpt). It is copied verbatim from `input_polymer_pool.json`, so the arms
differ only in weight init and the lr/steps schedule:

| | finetune | from scratch | why |
|---|---|---|---|
| `start_lr` / `decay_steps` | 1e-3 / 2000 | 1e-3 / 5000 | random descriptor needs the lr held up longer |
| `stop_lr` | 1e-5 | 1e-6 | longer anneal |
| `numb_steps` | 200k | 400k | from scratch converges slower |

The schedule is deliberately *not* held fixed across arms — a from-scratch run forced
onto the finetune schedule is a strawman. Everything that would confound the
comparison (descriptor architecture, head, data, split, loss) *is* held fixed.

Things that look like they should change for from-scratch but must not:

- **`type_map`** stays the 118-element periodic-order map. The data's
  `real_atom_types` are indices into it, so shrinking it to the ~5 elements actually
  present would mean rebuilding the data and comparing against a different dataset.
  The unused type-embedding rows are harmless.
- **`skip_stat: true`** stays. It is deprecated and only forces `fix_stat_std=0.3`
  (it does *not* mean "skip computing stats"), so leaving it keeps both arms
  normalising identically.
- **pre-standardized `fparam`** stays as-is. `polymer_pool.forward` consumes `fparam`
  raw — it never applies `fparam_avg`/`fparam_inv_std` — so the README's
  finetune-mode caveat about fparam stats does not change anything here.

`dim_case_embd: 0` (the multitask config needs `23` only to agree with its energy
branch; single-task has no branch to match).

Expect this arm to **overfit**: 591 polymers against a 16-layer / 128-dim repflow.
That gap is the result you are after, but read it honestly — it measures
*pretraining vs none at this architecture size*, not *whether the architecture is
right*. A from-scratch run at this data scale would normally use a far smaller
descriptor, so a strong finetune result does not by itself prove the pretrained
features are what helped, only that a randomly-initialised model this large cannot
be fit on 591 samples. If you want to separate those, add a third arm with a small
descriptor (e.g. `nlayers: 4`, `n_dim: 32`) trained from scratch.

## Multitask + energy replay

`input_multitask_replay.json` co-trains two branches on the **shared** DPA-3.3
descriptor: `polymer` (our `polymer_pool` head, the cloud-point data) and `energy`
(a native `ener` head on an energy+force **replay** set). Replaying energy keeps the
descriptor from forgetting its pretrained physics when it is finetuned on only ~591
polymers.

Fill in before running:
- `training.data_dict.energy.training_data.systems` → your energy replay set
  (a few-thousand-frame OMat24 / MPTrj-style subset with `energy.npy` + `force.npy`).
- `model_dict.energy.finetune_head` → the pretrained branch that replay data came
  from (e.g. `OMol25`); its energy fitting is resumed. The `polymer` branch has no
  `finetune_head`, so its head is re-initialised fresh while inheriting the descriptor.
- `training.model_prob` → step fraction per branch (more `energy` = stronger anchor).

**Case embedding:** all branches must declare the *same* `fitting_net.dim_case_embd`.
The pretrained energy fitting uses `23`, so the `polymer` branch declares `23` too —
`polymer_pool` ignores the case embedding in its forward, and because finetune counts
as *resuming*, `set_case_embd` is skipped so the energy branch keeps OMol25's own case
embedding. (Mismatch → `ValueError: All models must have the same dimension of case
embedding`.)

Run (multitask finetune from the multitask checkpoint):
```bash
dp --pt train input_multitask_replay.json \
   --finetune /path/to/model.ckpt-6860000.pt --use-pretrain-script
```
Validate exactly as single-task: freeze, then `dp --pt test` the `polymer` branch on
`test_low` / `test_high`, and compare to the single-task e2e numbers — if energy
replay helped, the OOD tails improve (or IND holds while the descriptor stays
physical).

Read the arms as a ladder, each isolating one thing:

| arm | config | isolates |
|---|---|---|
| from scratch | `input_polymer_pool_scratch.json` | — (floor) |
| frozen probe | `input_polymer_pool.json`, `descriptor.trainable: false` | value of pretrained features alone |
| single-task e2e | `input_polymer_pool.json`, `descriptor.trainable: true` | + adapting the descriptor to polymers |
| multitask + replay | `input_multitask_replay.json` | + energy replay as an anti-forgetting anchor |

The replay claim lives in the last two rows: replay only earns its keep if e2e's OOD
tails degrade (descriptor forgetting its physics on 591 polymers) *and* multitask
recovers them. If e2e already holds up on the tails, replay is solving a problem you
do not have.

`polymer_additive` is a **second axis**, not another rung — it crosses the ladder
rather than extending it. The cheap version is the one comparison that matters:
`input_multitask_replay_additive.json` against `input_multitask_replay.json`, i.e.
the pooled head vs the additive head at the strongest descriptor you have. Running
the additive variant at *every* rung costs 4x for a question ("is the head
non-additivity load-bearing?") that one well-chosen pair answers.

## Parity plots

`plot_ckpt_parity.py` (in the polymer project) evaluates a frozen model on every
system and draws predicted-vs-true, IND (train/val) vs OOD (low/high) colour-coded:
```bash
dp --pt freeze -o e2e.pth
python plot_ckpt_parity.py --data dpdata_polymer --model e2e.pth:e2e
# or compare arms: --model frozen.pth:frozen --model e2e.pth:e2e
```

## Notes

- `pool_norm: layer` (default) because one-frame-per-system means one frame per step
  — BatchNorm over the batch would be degenerate. Use `batch` only with multi-frame
  systems.
- `backend-agnostic (.dp) serialize/deserialize` is not implemented for
  `polymer_pool`; use torch checkpoints (training) and `dp --pt freeze` (TorchScript).
