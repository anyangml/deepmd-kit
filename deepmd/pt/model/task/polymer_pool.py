# SPDX-License-Identifier: LGPL-3.0-or-later
"""Weighted per-fragment embedding-pool property fitting.

Motivation
----------
The native ``property`` fitting applies the readout MLP *per atom* and then
reduces (sum/mean) — i.e. it is additive over atoms. For a polymer we instead
want to pool the per-fragment *embeddings* first and run a single non-linear head
on the pooled representation together with process parameters:

    E_r = Σ_i w_i · g_i     over atoms of role r      (w_i in aparam[..., 0])
    E   = concat_r E_r
    ŷ   = head( concat[ norm(E), fparam ] )

so composition interactions between comonomers and coupling with the process
parameters are representable (they are not in the additive per-atom form).

Implementation trick (reuse the whole model/reduce stack)
---------------------------------------------------------
The fitting sees *all* of a frame's atoms at once and receives ``fparam`` /
``aparam``. So we compute the single frame prediction ``ŷ`` here and **broadcast
it to every atom**. With ``intensive=True`` the model's mean atom-reduce
(``transform_output``) returns ``mean_i ŷ = ŷ`` unchanged. Nothing else in
``PropertyModel`` / ``DPPropertyAtomicModel`` / ``make_model`` needs to change,
and the stock ``property`` loss is used as-is.

Data contract (see ``polymer_to_dpdata.py``)
--------------------------------------------
Each polymer is one frame; its fragments are laid out > rcut apart so the shared
descriptor embeds each independently.
    aparam [nf, nloc, 1 + n_roles] :  [w, role0_onehot, role1_onehot, ...]
        w = unit_weight / n_atoms_in_unit  (RAW pooling weight; NOT standardized —
        Σ over a role's atoms = that role's pooled weight, e.g. Σ mole fractions).
    fparam [nf, numb_fparam]       :  process params, PRE-STANDARDIZED on the train
        split by the data script (DeePMD does not compute fparam input stats in
        finetune mode), so they are used here directly.

``numb_fparam`` / ``numb_aparam`` are set > 0 only so the data pipeline loads and
threads ``fparam.npy`` / ``aparam.npy``; this fitting consumes them via pooling,
NOT via the base per-atom concatenation (``forward`` is overridden, so the base
``_forward_common`` standardize-and-concat path is never taken and its
``filter_layers`` are unused).

Norm
----
With one-system-per-polymer, DeePMD trains one frame per step, so BatchNorm over
the frame batch is degenerate. ``pool_norm`` defaults to ``layer`` (per-sample
LayerNorm, batch-size independent). Use ``batch`` only with multi-frame systems.
"""

from typing import (
    Any,
)

import torch

from deepmd.pt.model.task.fitting import (
    Fitting,
)
from deepmd.pt.model.task.property import (
    PropertyFittingNet,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
)


@Fitting.register("polymer_pool")
class PolymerPoolFitting(PropertyFittingNet):
    """Property fitting that pools per-fragment embeddings then runs one head.

    Extra parameters (beyond ``PropertyFittingNet``)
    ------------------------------------------------
    n_roles : int
        Number of fragment roles pooled into separate channels (2 = end group /
        repeating unit). aparam layout is ``[w, *role_onehot(n_roles)]`` so
        ``numb_aparam`` must equal ``n_roles + 1``.
    head_neuron : list[int]
        Hidden widths of the readout head that maps ``[norm(E) ; fparam]`` -> task.
    head_activation : str
        Head activation (``silu`` by default, matching the reference head).
    head_dropout : float
        Dropout probability in the head.
    pool_norm : str
        ``layer`` (default, batch-independent) | ``batch`` | ``none`` applied to the
        pooled embedding E before concatenating fparam.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        property_name: str,
        task_dim: int = 1,
        n_roles: int = 2,
        head_neuron: list[int] = [256, 256],
        head_activation: str = "silu",
        head_dropout: float = 0.1,
        pool_norm: str = "layer",
        intensive: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        precision: str = DEFAULT_PRECISION,
        seed: int | None = None,
        # captured from the fitting dict the standard-model builder injects
        # (descriptor.mixed_types(), etc.) so they do not collide with the values
        # we pass to super().
        mixed_types: bool = True,
        distinguish_types: bool = False,
        **kwargs: Any,
    ) -> None:
        # aparam carries [w, role0, ..., role_{n_roles-1}]
        if numb_aparam != n_roles + 1:
            raise ValueError(
                f"polymer_pool requires numb_aparam == n_roles + 1 "
                f"(got numb_aparam={numb_aparam}, n_roles={n_roles}); aparam layout "
                f"is [w, *role_onehot]."
            )
        if not intensive:
            raise ValueError(
                "polymer_pool broadcasts the frame prediction to every atom and "
                "relies on the intensive mean atom-reduce; set intensive=true."
            )
        super().__init__(
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            property_name=property_name,
            task_dim=task_dim,
            intensive=True,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            precision=precision,
            seed=seed,
            mixed_types=mixed_types,
            distinguish_types=False,
            **kwargs,
        )
        self.n_roles = int(n_roles)
        self.head_neuron = list(head_neuron)
        self.head_activation = str(head_activation)
        self.head_dropout = float(head_dropout)
        self.pool_norm = str(pool_norm)
        # self.prec is resolved by GeneralFitting.__init__ (handles "default")

        pooled_dim = self.n_roles * dim_descrpt
        if self.pool_norm == "layer":
            self.norm: torch.nn.Module = torch.nn.LayerNorm(pooled_dim)
        elif self.pool_norm == "batch":
            self.norm = torch.nn.BatchNorm1d(pooled_dim)
        elif self.pool_norm == "none":
            self.norm = torch.nn.Identity()
        else:
            raise ValueError(f"unknown pool_norm={self.pool_norm}")

        act = {"silu": torch.nn.SiLU, "gelu": torch.nn.GELU,
               "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}[self.head_activation]
        layers: list[torch.nn.Module] = []
        in_dim = pooled_dim + numb_fparam
        for h in self.head_neuron:
            layers += [torch.nn.Linear(in_dim, h), act(),
                       torch.nn.Dropout(self.head_dropout)]
            in_dim = h
        layers += [torch.nn.Linear(in_dim, task_dim)]
        self.head = torch.nn.Sequential(*layers)
        self.head.to(self.prec)
        self.norm.to(self.prec)

    def forward(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        gr: torch.Tensor | None = None,
        g2: torch.Tensor | None = None,
        h2: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        return_atomic_feature: bool = False,
    ) -> dict[str, torch.Tensor]:
        xx = descriptor.to(self.prec)
        nf, nloc, nd = xx.shape
        assert aparam is not None, "polymer_pool needs aparam (pooling weights/roles)"
        aparam = aparam.view(nf, nloc, self.numb_aparam).to(self.prec)
        w = aparam[..., 0:1]                              # [nf, nloc, 1] pooling weight
        role = aparam[..., 1 : 1 + self.n_roles]          # [nf, nloc, n_roles] one-hot
        weighted = w * xx                                 # [nf, nloc, nd]
        # per-role weighted sum over atoms: [nf, n_roles, nd]
        pooled = torch.matmul(role.transpose(1, 2), weighted)
        E = pooled.reshape(nf, self.n_roles * nd)         # [nf, n_roles*nd]
        E = self.norm(E)
        if self.numb_fparam > 0:
            assert fparam is not None, "polymer_pool needs fparam (process params)"
            E = torch.cat([E, fparam.view(nf, self.numb_fparam).to(self.prec)], dim=-1)
        yhat = self.head(E)                               # [nf, task_dim]
        # broadcast to every atom so the intensive mean atom-reduce returns yhat
        outs = yhat.unsqueeze(1).expand(nf, nloc, self.dim_out).contiguous()
        return {self.var_name: outs}

    def serialize(self) -> dict:
        dd = super().serialize()
        dd["type"] = "polymer_pool"
        dd["n_roles"] = self.n_roles
        dd["head_neuron"] = self.head_neuron
        dd["head_activation"] = self.head_activation
        dd["head_dropout"] = self.head_dropout
        dd["pool_norm"] = self.pool_norm
        dd["@polymer_state"] = {
            k: v.detach().cpu().numpy()
            for k, v in {**self.norm.state_dict(prefix="norm."),
                         **self.head.state_dict(prefix="head.")}.items()
        }
        return dd

    @classmethod
    def deserialize(cls, data: dict) -> "PolymerPoolFitting":
        data = data.copy()
        state = data.pop("@polymer_state", None)
        for k in ("n_roles", "head_neuron", "head_activation", "head_dropout",
                  "pool_norm"):
            data.pop(k, None)
        # PropertyFittingNet.deserialize rebuilds the base; then reattach the head.
        # Re-run our __init__ path via the stored config instead:
        raise NotImplementedError(
            "polymer_pool round-trips through torch checkpoints (training) and "
            "torch.jit (dp --pt freeze); backend-agnostic .dp serialize/deserialize "
            "is not implemented."
        )
