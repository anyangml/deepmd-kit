# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.pt.model.descriptor.base_descriptor import (
    BaseDescriptor,
)

@BaseModel.register("standard")
class DPModel:
    """A base class to implement common methods for all the Models."""
   
    def __new__(
        cls,
        descriptor=None,
        fitting=None,
        *args,
        # disallow positional atomic_model_
        atomic_model_: Optional[DPAtomicModel] = None,
        **kwargs,
    ):
        from deepmd.pt.model.model.dipole_model import (
            DipoleModel,
        )
        from deepmd.pt.model.model.dos_model import (
            DOSModel,
        )
        from deepmd.pt.model.model.ener_model import (
            EnergyModel,
        )
        from deepmd.pt.model.model.polar_model import (
            PolarModel,
        )

        if atomic_model_ is not None:
            fitting = atomic_model_.fitting_net
        else:
            assert fitting is not None, "fitting network is not provided"

        # according to the fitting network to decide the type of the model
        if cls is DPModel:
            # map fitting to model
            if isinstance(fitting, EnergyFittingNet) or isinstance(
                fitting, EnergyFittingNetDirect
            ):
                cls = EnergyModel
            elif isinstance(fitting, DipoleFittingNet):
                cls = DipoleModel
            elif isinstance(fitting, PolarFittingNet):
                cls = PolarModel
            elif isinstance(fitting, DOSFittingNet):
                cls = DOSModel
            # else: unknown fitting type, fall back to DPModel
        return super().__new__(cls)

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
        local_jdata_cpy["descriptor"] = BaseDescriptor.update_sel(
            global_jdata, local_jdata["descriptor"]
        )
        return local_jdata_cpy

    def get_fitting_net(self):
        """Get the fitting network."""
        return self.atomic_model.fitting_net

    def get_descriptor(self):
        """Get the descriptor."""
        return self.atomic_model.descriptor
