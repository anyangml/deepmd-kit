# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    Optional,
)

import torch

from deepmd.pt.model.atomic_model import (
    DPDipoleAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)

from .make_model import (
    make_model,
)


@BaseModel.register("standard")
class DipoleModel(make_model(DPDipoleAtomicModel)):
    model_type = "dipole"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        coord,
        atype,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        if self.get_fitting_net() is not None:
            model_predict = {}
            model_predict["dipole"] = model_ret["dipole"]
            model_predict["global_dipole"] = model_ret["dipole_redu"]
            if self.do_grad_r("dipole"):
                model_predict["force"] = model_ret["dipole_derv_r"].squeeze(-2)
            if self.do_grad_c("dipole"):
                model_predict["virial"] = model_ret["dipole_derv_c_redu"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["atom_virial"] = model_ret["dipole_derv_c"].squeeze(
                        -3
                    )
            if "mask" in model_ret:
                model_predict["mask"] = model_ret["mask"]
        else:
            model_predict = model_ret
            model_predict["updated_coord"] += coord
        return model_predict

    @classmethod
    def update_sel(cls, global_jdata: dict, local_jdata: dict):
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        global_jdata : dict
            The global data, containing the training section
        local_jdata : dict
            The local data refer to the current class
        """
        local_jdata_cpy = local_jdata.copy()
        local_jdata_cpy["descriptor"] = cls.get_descriptor().update_sel(
            global_jdata, local_jdata["descriptor"]
        )
        return local_jdata_cpy

    def get_fitting_net(self):
        """Get the fitting network."""
        return self.atomic_model.fitting_net

    def get_descriptor(self):
        """Get the descriptor."""
        return self.atomic_model.descriptor

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ):
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        if self.get_fitting_net() is not None:
            model_predict = {}
            model_predict["dipole"] = model_ret["dipole"]
            model_predict["global_dipole"] = model_ret["dipole_redu"]
            if self.do_grad_r("dipole"):
                model_predict["force"] = model_ret["dipole_derv_r"].squeeze(-2)
            if self.do_grad_c("dipole"):
                model_predict["virial"] = model_ret["dipole_derv_c_redu"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["atom_virial"] = model_ret["dipole_derv_c"].squeeze(
                        -3
                    )
        else:
            model_predict = model_ret
        return model_predict
