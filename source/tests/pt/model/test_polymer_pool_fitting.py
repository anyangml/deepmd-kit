# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the weighted per-fragment embedding-pool property fitting.

These test the fitting in isolation (a toy descriptor tensor), so they need no
descriptor C++ ops. They verify the three properties the design relies on:
  1. the frame prediction is broadcast to every atom, so the model's intensive
     mean atom-reduce returns it unchanged;
  2. the pooling equals the explicit reference E_r = Σ_i w_i·role_ir·g_i;
  3. the module is TorchScript-scriptable (needed for `dp --pt freeze`).
"""
import unittest

import torch

from deepmd.pt.model.task.polymer_pool import (
    PolymerPoolFitting,
)
from deepmd.pt.utils import (
    env,
)

dtype = torch.float64
device = env.DEVICE


def _make(nr=2, D=8, Fdim=3, task_dim=1):
    return PolymerPoolFitting(
        ntypes=3,
        dim_descrpt=D,
        property_name="cloud_point",
        task_dim=task_dim,
        n_roles=nr,
        head_neuron=[16, 16],
        numb_fparam=Fdim,
        numb_aparam=nr + 1,
        pool_norm="layer",
        precision="float64",
        seed=0,
    ).to(device)


def _inputs(f, nf=3, nloc=6, D=8, Fdim=3, nr=2):
    torch.manual_seed(1)
    desc = torch.randn(nf, nloc, D, dtype=dtype, device=device)
    atype = torch.zeros(nf, nloc, dtype=torch.long, device=device)
    roles = torch.randint(0, nr, (nf, nloc), device=device)
    onehot = torch.nn.functional.one_hot(roles, nr).to(dtype)
    w = torch.rand(nf, nloc, 1, dtype=dtype, device=device)
    aparam = torch.cat([w, onehot], dim=-1)
    fparam = torch.randn(nf, Fdim, dtype=dtype, device=device)
    return desc, atype, fparam, aparam, w, onehot


class TestPolymerPoolFitting(unittest.TestCase):
    def test_output_def_intensive(self):
        f = _make()
        od = f.output_def()[f.var_name]
        self.assertTrue(od.reducible)
        self.assertTrue(od.intensive)

    def test_requires_intensive_and_aparam_dim(self):
        with self.assertRaises(ValueError):
            _make()  # ok
            PolymerPoolFitting(ntypes=3, dim_descrpt=8, property_name="p",
                               n_roles=2, numb_aparam=2)  # != n_roles+1
        with self.assertRaises(ValueError):
            PolymerPoolFitting(ntypes=3, dim_descrpt=8, property_name="p",
                               n_roles=2, numb_aparam=3, intensive=False)

    def test_broadcast_is_constant_across_atoms(self):
        f = _make().eval()
        desc, atype, fparam, aparam, _, _ = _inputs(f)
        out = f(desc, atype, fparam=fparam, aparam=aparam)[f.var_name]
        # every atom carries the same frame value -> intensive mean-reduce is exact
        self.assertTrue(torch.allclose(out, out[:, :1, :].expand_as(out)))

    def test_pooling_matches_reference(self):
        f = _make().eval()
        desc, atype, fparam, aparam, w, onehot = _inputs(f)
        out = f(desc, atype, fparam=fparam, aparam=aparam)[f.var_name]
        nr = f.n_roles
        Eref = torch.cat(
            [(w * onehot[..., r : r + 1] * desc).sum(dim=1) for r in range(nr)],
            dim=-1,
        )
        yref = f.head(torch.cat([f.norm(Eref), fparam], dim=-1))
        self.assertTrue(torch.allclose(out[:, 0, :], yref, atol=1e-10))

    def test_torchscript(self):
        f = _make().eval()
        desc, atype, fparam, aparam, _, _ = _inputs(f)
        sm = torch.jit.script(f)
        o1 = f(desc, atype, fparam=fparam, aparam=aparam)[f.var_name]
        o2 = sm(desc, atype, fparam=fparam, aparam=aparam)[f.var_name]
        self.assertTrue(torch.allclose(o1, o2, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
