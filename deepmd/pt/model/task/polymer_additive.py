# SPDX-License-Identifier: LGPL-3.0-or-later
"""Mole-fraction-weighted *additive* property fitting (ablation of ``polymer_pool``).

Motivation
----------
``polymer_pool`` pools per-fragment *embeddings* and runs one head on the pooled
representation, so comonomer interactions and coupling with the process parameters
are representable. This fitting is its deliberate **ablation**: it keeps the same
data, the same weights, and the same reduce, but projects *first* and weights
*after* — exactly like the energy fitting, only with mole-fraction weights instead
of a plain sum:

    ŷ = Σ_i w_i · f(g_i, fparam)        (w_i in aparam[..., 0])

Since Σ over a role's atoms of w_i = that role's pooled weight (e.g. its mole
fraction), this is "atomic contributions averaged by composition". It is additive
over fragments *by construction*, so it cannot represent comonomer interaction —
that is the point: it isolates how much of ``polymer_pool``'s accuracy comes from
the non-additive head rather than from the descriptor.

Run it against ``polymer_pool`` on identical data to make the non-additivity claim
measurable. See ``examples/property/polymer_cloud_point/README.md``.

``role`` is unused here
-----------------------
The role one-hot in ``aparam[..., 1:]`` is *ignored*: in the additive form every
role's contribution lands in the same Σ, so there are no per-role channels to
separate. ``numb_aparam == n_roles + 1`` is still required so the exact same
``aparam.npy`` feeds both fittings and the two configs are swappable on one dataset.
(A middle-ground variant — one head per role, ``Σ_r Σ_{i∈r} w_i f_r(g_i)`` — is
still additive but role-aware; not implemented, as the single-head form is the
cleaner ablation.)

Implementation trick (reuse the base per-atom stack *and* the intensive reduce)
-------------------------------------------------------------------------------
``GeneralFitting._forward_common`` already does the per-atom projection (descriptor
[+ fparam] -> ``filter_layers`` -> ŷ_i), so this class only rescales its output.

Two non-obvious choices:

1. ``use_aparam_as_mask=True``. ``numb_aparam > 0`` is needed so the data pipeline
   loads ``aparam.npy``, but the base would then standardize aparam and concatenate
   it per atom — feeding the composition weight ``w`` into the head as a feature,
   which breaks additivity in ``w``. This flag keeps aparam loaded but out of the
   input (``fitting.py``: ``in_dim`` and the concat gate). Its documented purpose is
   ``se_a_mask``; the reuse here is deliberate.

2. ``intensive=True`` with the weight pre-multiplied by ``nloc``. The obvious route
   is ``intensive=False`` (the extensive reduce *is* a sum), but the property loss
   then divides pred *and* label by ``natoms``, which rescales the metric to
   per-atom units and reweights each frame by ``1/natoms`` — and with one polymer per
   system, ``natoms`` varies per frame. Instead we emit ``nloc · w_i · ŷ_i`` and let
   the intensive mean-reduce (``1/nloc)Σ``) return ``Σ_i w_i ŷ_i``. Loss, stats, and
   config then match ``polymer_pool`` exactly, so the A/B is apples-to-apples.

Bias
----
Unlike ``polymer_pool`` (which overrides ``forward`` outright, so ``bias_atom_e`` is
never applied), this fitting goes through ``_forward_common`` and the per-type bias
*is* applied, then scaled by ``w``. The bias term is therefore ``b · Σ_i w_i``, which
is a clean global offset iff the pooling weights sum to 1 over the frame — as they do
when they are mole fractions. That matches what the intensive stat solve assumes, so
the stat-computed bias stays meaningful. Data whose weights do not sum to 1 breaks
this correspondence.
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


@Fitting.register("polymer_additive")
class PolymerAdditiveFitting(PropertyFittingNet):
    """Property fitting that projects per atom then sums weighted by mole fraction.

    The additive ablation of :class:`PolymerPoolFitting`. Consumes the identical
    ``aparam``/``fparam`` layout; see the module docstring for the design.

    Extra parameters (beyond ``PropertyFittingNet``)
    ------------------------------------------------
    n_roles : int
        Number of fragment roles in the aparam layout (2 = end group / repeating
        unit). Only used to validate ``numb_aparam == n_roles + 1``; the role
        one-hot itself is ignored by the additive form.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        property_name: str,
        task_dim: int = 1,
        n_roles: int = 2,
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
        # same aparam contract as polymer_pool, so one dataset feeds both
        if numb_aparam != n_roles + 1:
            raise ValueError(
                f"polymer_additive requires numb_aparam == n_roles + 1 "
                f"(got numb_aparam={numb_aparam}, n_roles={n_roles}); aparam layout "
                f"is [w, *role_onehot] (the role one-hot is unused here, but the "
                f"layout is kept identical to polymer_pool)."
            )
        if not intensive:
            raise ValueError(
                "polymer_additive folds nloc into the pooling weight and relies on "
                "the intensive mean atom-reduce to recover Σ_i w_i·ŷ_i; set "
                "intensive=true. (intensive=false would sum correctly but makes the "
                "property loss divide pred/label by natoms.)"
            )
        super().__init__(
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            property_name=property_name,
            task_dim=task_dim,
            intensive=True,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            # keep aparam loaded by the data pipeline but out of the per-atom input:
            # w must not reach the head as a feature or the form is no longer
            # additive in the composition weight.
            use_aparam_as_mask=True,
            precision=precision,
            seed=seed,
            mixed_types=mixed_types,
            distinguish_types=False,
            **kwargs,
        )
        self.n_roles = int(n_roles)

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
        nf = descriptor.shape[0]
        nloc = descriptor.shape[1]
        assert aparam is not None, "polymer_additive needs aparam (pooling weights)"
        # per-atom projection, incl. the per-type bias (see module docstring)
        outs = self._forward_common(
            descriptor,
            atype,
            gr,
            g2,
            h2,
            fparam,
            aparam,
            return_atomic_feature,
        )
        w = aparam.view(nf, nloc, self.numb_aparam)[..., 0:1].to(self.prec)
        # nloc·w_i·ŷ_i, so the intensive mean-reduce returns Σ_i w_i·ŷ_i
        outs[self.var_name] = outs[self.var_name] * (w * float(nloc))
        return outs

    def serialize(self) -> dict:
        dd = super().serialize()
        dd["type"] = "polymer_additive"
        dd["n_roles"] = self.n_roles
        return dd

    @classmethod
    def deserialize(cls, data: dict) -> "PolymerAdditiveFitting":
        raise NotImplementedError(
            "polymer_additive round-trips through torch checkpoints (training) and "
            "torch.jit (dp --pt freeze); backend-agnostic .dp serialize/deserialize "
            "is not implemented (same as polymer_pool)."
        )
