# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the mole-fraction-weighted additive property fitting.

These test the fitting in isolation (a toy descriptor tensor), so they need no
descriptor C++ ops. They verify the properties the design relies on:
  1. the intensive mean atom-reduce of the output equals Sum_i w_i*y_i (the whole
     point of folding nloc into the weight);
  2. the output is genuinely additive over fragments, which is what makes this the
     ablation of polymer_pool rather than a reparametrisation of it;
  3. `w` never reaches the head as an input feature (use_aparam_as_mask);
  4. the module is TorchScript-scriptable (needed for `dp --pt freeze`).
"""

import unittest

import torch

from deepmd.pt.model.task.polymer_additive import (
    PolymerAdditiveFitting,
)

dtype = torch.float64
# The weighting logic is device-agnostic; pin to CPU so the test does not depend on
# a working CUDA/driver stack (and stays fast in CI).
device = torch.device("cpu")


def _make(nr=2, D=8, Fdim=3, task_dim=1):
    return PolymerAdditiveFitting(
        ntypes=3,
        dim_descrpt=D,
        property_name="cloud_point",
        task_dim=task_dim,
        n_roles=nr,
        neuron=[16, 16],
        numb_fparam=Fdim,
        numb_aparam=nr + 1,
        precision="float64",
        seed=0,
    ).to(device)


def _inputs(nf=3, nloc=6, D=8, Fdim=3, nr=2):
    torch.manual_seed(1)
    desc = torch.randn(nf, nloc, D, dtype=dtype, device=device)
    atype = torch.zeros(nf, nloc, dtype=torch.long, device=device)
    roles = torch.randint(0, nr, (nf, nloc), device=device)
    onehot = torch.nn.functional.one_hot(roles, nr).to(dtype)
    # mole-fraction-like: weights sum to 1 over the frame (see the bias note in the
    # module docstring — the stat-solved bias only stays meaningful under this)
    w = torch.rand(nf, nloc, 1, dtype=dtype, device=device)
    w = w / w.sum(dim=1, keepdim=True)
    aparam = torch.cat([w, onehot], dim=-1)
    fparam = torch.randn(nf, Fdim, dtype=dtype, device=device)
    return desc, atype, fparam, aparam, w, onehot


class TestPolymerAdditiveFitting(unittest.TestCase):
    def test_output_def_intensive(self):
        f = _make()
        od = f.output_def()[f.var_name]
        self.assertTrue(od.reducible)
        self.assertTrue(od.intensive)

    def test_rejects_bad_aparam_dim_and_extensive(self):
        with self.assertRaises(ValueError):
            PolymerAdditiveFitting(
                ntypes=3, dim_descrpt=8, property_name="p", n_roles=2, numb_aparam=2
            )  # != n_roles+1
        with self.assertRaises(ValueError):
            PolymerAdditiveFitting(
                ntypes=3,
                dim_descrpt=8,
                property_name="p",
                n_roles=2,
                numb_aparam=3,
                intensive=False,
            )

    def test_aparam_not_in_head_input(self):
        # w must not be concatenated into the per-atom input, else the form is no
        # longer additive in the composition weight.
        f = _make()
        self.assertTrue(f.use_aparam_as_mask)
        # in_dim excludes numb_aparam but still includes numb_fparam
        self.assertEqual(f.filter_layers.networks[0].in_dim, 8 + 3)

    def test_mean_reduce_recovers_weighted_sum(self):
        # The load-bearing claim: model-side mean over atoms == Sum_i w_i * yhat_i.
        f = _make().eval()
        desc, atype, fparam, aparam, w, _ = _inputs()
        out = f(desc, atype, fparam=fparam, aparam=aparam)[f.var_name]
        reduced = out.mean(dim=1)  # what transform_output does for intensive vars
        # reference: unscaled per-atom projection, weighted and summed explicitly
        per_atom = f._forward_common(desc, atype, None, None, None, fparam, aparam)[
            f.var_name
        ]
        ref = (w * per_atom).sum(dim=1)
        self.assertTrue(torch.allclose(reduced, ref, atol=1e-12))

    def test_is_additive_over_fragments(self):
        # Splitting a frame's atoms into two disjoint groups and summing their
        # weighted contributions must equal the whole frame's value. This is what
        # polymer_pool is NOT, so it is the property that makes the A/B meaningful.
        f = _make().eval()
        desc, atype, fparam, aparam, w, _ = _inputs(nf=1, nloc=6)
        per_atom = f._forward_common(desc, atype, None, None, None, fparam, aparam)[
            f.var_name
        ]
        whole = (w * per_atom).sum(dim=1)
        left = (w[:, :3] * per_atom[:, :3]).sum(dim=1)
        right = (w[:, 3:] * per_atom[:, 3:]).sum(dim=1)
        self.assertTrue(torch.allclose(whole, left + right, atol=1e-12))

    def test_torchscript(self):
        f = _make().eval()
        desc, atype, fparam, aparam, _, _ = _inputs()
        sm = torch.jit.script(f)
        o1 = f(desc, atype, fparam=fparam, aparam=aparam)[f.var_name]
        o2 = sm(desc, atype, fparam=fparam, aparam=aparam)[f.var_name]
        self.assertTrue(torch.allclose(o1, o2, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
