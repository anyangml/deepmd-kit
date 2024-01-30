# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

import torch
from torch import (
    nn,
)

from deepmd.model_format import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.utils.pair_tab import (
    PairTab,
)

from .atomic_model import (
    AtomicModel,
)


class PairTabModel(nn.Module, AtomicModel):
    """Pairwise tabulation energy model.

    This model can be used to tabulate the pairwise energy between atoms for either
    short-range or long-range interactions, such as D3, LJ, ZBL, etc. It should not
    be used alone, but rather as one submodel of a linear (sum) model, such as
    DP+D3.

    Do not put the model on the first model of a linear model, since the linear
    model fetches the type map from the first model.

    At this moment, the model does not smooth the energy at the cutoff radius, so
    one needs to make sure the energy has been smoothed to zero.

    Parameters
    ----------
    tab_file : str
        The path to the tabulation file.
    rcut : float
        The cutoff radius.
    sel : int or list[int]
        The maxmum number of atoms in the cut-off radius.
    """

    def __init__(
        self, tab_file: str, rcut: float, sel: Union[int, List[int]], **kwargs
    ):
        super().__init__()
        self.tab_file = tab_file
        self.rcut = rcut

        self.tab = PairTab(self.tab_file, rcut=rcut)
        self.ntypes = self.tab.ntypes

        tab_info, tab_data = self.tab.get()  # this returns -> Tuple[np.array, np.array]
        self.tab_info = torch.from_numpy(tab_info)
        self.tab_data = torch.from_numpy(tab_data)

        # self.model_type = "ener"
        # self.model_version = MODEL_VERSION ## this shoud be in the parent class

        if isinstance(sel, int):
            self.sel = sel
        elif isinstance(sel, list):
            self.sel = sum(sel)
        else:
            raise TypeError("sel must be int or list[int]")

    def get_fitting_output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    name="energy", shape=[1], reduciable=True, differentiable=True
                )
            ]
        )

    def get_rcut(self) -> float:
        return self.rcut

    def get_sel(self) -> int:
        return self.sel

    def distinguish_types(self) -> bool:
        # to match DPA1 and DPA2.
        return False

    def forward_atomic(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        self.nframes, self.nloc, self.nnei = nlist.shape

        # this will mask all -1 in the nlist
        masked_nlist = torch.clamp(nlist, 0)

        atype = extended_atype[:, : self.nloc]  # (nframes, nloc)
        pairwise_dr = self._get_pairwise_dist(
            extended_coord
        )  # (nframes, nall, nall, 3)
        pairwise_rr = pairwise_dr.pow(2).sum(-1).sqrt()  # (nframes, nall, nall)

        self.tab_data = self.tab_data.reshape(
            self.tab.ntypes, self.tab.ntypes, self.tab.nspline, 4
        )

        # to calculate the atomic_energy, we need 3 tensors, i_type, j_type, rr
        # i_type : (nframes, nloc), this is atype.
        # j_type : (nframes, nloc, nnei)
        j_type = extended_atype[
            torch.arange(extended_atype.size(0))[:, None, None], masked_nlist
        ]

        # slice rr to get (nframes, nloc, nnei)
        rr = torch.gather(pairwise_rr[:, : self.nloc, :], 2, masked_nlist)

        raw_atomic_energy = self._pair_tabulated_inter(nlist, atype, j_type, rr)

        atomic_energy = 0.5 * torch.sum(
            torch.where(
                nlist != -1, raw_atomic_energy, torch.zeros_like(raw_atomic_energy)
            ),
            dim=-1,
        )

        return {"energy": atomic_energy}

    def _pair_tabulated_inter(
        self,
        nlist: torch.Tensor,
        i_type: torch.Tensor,
        j_type: torch.Tensor,
        rr: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise tabulated energy.

        Parameters
        ----------
        nlist : torch.Tensor
            The unmasked neighbour list. (nframes, nloc)
        i_type : torch.Tensor
            The integer representation of atom type for all local atoms for all frames. (nframes, nloc)
        j_type : torch.Tensor
            The integer representation of atom type for all neighbour atoms of all local atoms for all frames. (nframes, nloc, nnei)
        rr : torch.Tensor
            The salar distance vector between two atoms. (nframes, nloc, nnei)

        Returns
        -------
        torch.Tensor
            The masked atomic energy for all local atoms for all frames. (nframes, nloc, nnei)

        Raises
        ------
        Exception
            If the distance is beyond the table.

        Notes
        -----
        This function is used to calculate the pairwise energy between two atoms.
        It uses a table containing cubic spline coefficients calculated in PairTab.
        """
        rmin = self.tab_info[0]
        hh = self.tab_info[1]
        hi = 1.0 / hh

        self.nspline = int(self.tab_info[2] + 0.1)

        uu = (rr - rmin) * hi  # this is broadcasted to (nframes,nloc,nnei)

        # if nnei of atom 0 has -1 in the nlist, uu would be 0.
        # this is to handle the nlist where the mask is set to 0, so that we don't raise exception for those atoms.
        uu = torch.where(nlist != -1, uu, self.nspline + 1)

        if torch.any(uu < 0):
            raise Exception("coord go beyond table lower boundary")

        idx = uu.to(torch.int)

        uu -= idx

        table_coef = self._extract_spline_coefficient(
            i_type, j_type, idx, self.tab_data, self.nspline
        )
        table_coef = table_coef.reshape(self.nframes, self.nloc, self.nnei, 4)
        ener = self._calcualte_ener(table_coef, uu)

        if self.tab.rmax <= self.rcut:
            # here we need to overwrite energy to zero beyond rcut.
            mask_beyond_rcut = rr > self.rcut
            ener[mask_beyond_rcut] = 0

            # here we use smooth extrapolation to replace linear extrapolation.
            extrapolation = self._extrapolate_rmax_rcut()
            if extrapolation is not None:
                uu_extrapolate = (rr - self.tab.rmax) / (self.rcut - self.tab.rmax)
                clipped_uu = torch.clamp(uu_extrapolate, 0, 1)  # clip rr within rmax.
                extrapolate_coef = self._extract_spline_coefficient(
                    i_type, j_type, torch.zeros_like(idx), extrapolation, 1
                )
                extrapolate_coef = extrapolate_coef.reshape(
                    self.nframes, self.nloc, self.nnei, 4
                )
                ener_extrpolate = self._calcualte_ener(extrapolate_coef, clipped_uu)
                mask_rmax_to_rcut = (self.tab.rmax < rr) & (rr <= self.rcut)
                ener[mask_rmax_to_rcut] = ener_extrpolate[mask_rmax_to_rcut]
        return ener

    def _extrapolate_rmax_rcut(self) -> torch.Tensor:
        """Soomth extrapolation between table upper boundary and rcut.

        This method should only be used when the table upper boundary `rmax` is smaller than `rcut`, and
        the table upper boundary values are not zeros. To simplify the problem, we use a single
        cubic spline between `rmax` and `rcut` for each pair of atom types. One can substitute this extrapolation
        to higher order polynomials if needed.

        There are two scenarios:
            1. `ruct` - `rmax` >= hh:
                Set values at the grid point right before `rcut` to 0, and perform exterapolation between
                the grid point and `rmax`, this allows smooth decay to 0 at `rcut`.
            2. `rcut` - `rmax` < hh:
                Set values at `rmax + hh` to 0, and perform extrapolation between `rmax` and `rmax + hh`.

        Returns
        -------
        torch.Tensor
            The cubic spline coefficients for each pair of atom types. (ntype, ntype, 1, 4)
        """
        rmax_val = torch.from_numpy(
            self.tab.vdata[self.tab.vdata[:, 0] == self.tab.rmax]
        )
        pre_rmax_val = torch.from_numpy(
            self.tab.vdata[self.tab.vdata[:, 0] == self.tab.rmax - self.tab.hh]
        )

        # check if decays to `0` at rmax, if yes, no extrapolation is needed.
        if torch.all(rmax_val[:, 1:] == 0):
            return
        else:
            if self.rcut - self.tab.rmax >= self.tab.hh:
                rcut_idx = int(self.rcut / self.tab.hh - self.tab.rmin / self.tab.hh)
                rcut_val = torch.tensor(self.tab.vdata[rcut_idx, :]).reshape(1, -1)
                grid = torch.concatenate([rmax_val, rcut_val], axis=0)
            else:
                # the last two rows will be the rmax, and rmax+hh
                grid = torch.from_numpy(self.tab.vdata[-2:, :])
            passin_slope = (
                ((rmax_val - pre_rmax_val) / self.tab.hh)[:, 1:].squeeze(0)
                if self.tab.rmax > self.tab.hh
                else torch.zeros_like(rmax_val[:, 1:]).squeeze(0)
            )  # the slope at the end of table for each ntype pairs (ntypes,ntypes,1)
            extrapolate_coef = torch.from_numpy(
                self.tab._make_data(self.ntypes, 1, grid, self.tab.hh, passin_slope)
            ).reshape(self.ntypes, self.ntypes, 4)
            return extrapolate_coef.unsqueeze(2)

    @staticmethod
    def _get_pairwise_dist(coords: torch.Tensor) -> torch.Tensor:
        """Get pairwise distance `dr`.

        Parameters
        ----------
        coords : torch.Tensor
            The coordinate of the atoms shape of (nframes * nall * 3).

        Returns
        -------
        torch.Tensor
            The pairwise distance between the atoms (nframes * nall * nall * 3).

        Examples
        --------
        coords = torch.tensor([[
                [0,0,0],
                [1,3,5],
                [2,4,6]
            ]])

        dist = tensor([[
            [[ 0,  0,  0],
            [-1, -3, -5],
            [-2, -4, -6]],

            [[ 1,  3,  5],
            [ 0,  0,  0],
            [-1, -1, -1]],

            [[ 2,  4,  6],
            [ 1,  1,  1],
            [ 0,  0,  0]]
            ]])
        """
        return coords.unsqueeze(2) - coords.unsqueeze(1)

    @staticmethod
    def _extract_spline_coefficient(
        i_type: torch.Tensor,
        j_type: torch.Tensor,
        idx: torch.Tensor,
        tab_data: torch.Tensor,
        nspline: int,
    ) -> torch.Tensor:
        """Extract the spline coefficient from the table.

        Parameters
        ----------
        i_type : torch.Tensor
            The integer representation of atom type for all local atoms for all frames. (nframes, nloc)
        j_type : torch.Tensor
            The integer representation of atom type for all neighbour atoms of all local atoms for all frames. (nframes, nloc, nnei)
        idx : torch.Tensor
            The index of the spline coefficient. (nframes, nloc, nnei)
        tab_data : torch.Tensor
            The table storing all the spline coefficient. (ntype, ntype, nspline, 4)
        nspline : int
            The number of splines in the table.

        Returns
        -------
        torch.Tensor
            The spline coefficient. (nframes, nloc, nnei, 4), shape may be squeezed.

        """
        # (nframes, nloc, nnei)
        expanded_i_type = i_type.unsqueeze(-1).expand(-1, -1, j_type.shape[-1])

        # (nframes, nloc, nnei, nspline, 4)
        expanded_tab_data = tab_data[expanded_i_type, j_type]

        # (nframes, nloc, nnei, 1, 4)
        expanded_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 4)

        # handle the case where idx is beyond the number of splines
        clipped_indices = torch.clamp(expanded_idx, 0, nspline - 1).to(torch.int64)

        # (nframes, nloc, nnei, 4)
        final_coef = torch.gather(expanded_tab_data, 3, clipped_indices).squeeze()

        # when the spline idx is beyond the table, all spline coefficients are set to `0`, and the resulting ener corresponding to the idx is also `0`.
        final_coef[expanded_idx.squeeze() >= nspline] = 0

        return final_coef

    @staticmethod
    def _calcualte_ener(coef: torch.Tensor, uu: torch.Tensor) -> torch.Tensor:
        """Calculate energy using spline coeeficients.

        Parameters
        ----------
        coef : torch.Tensor
            The spline coefficients. (nframes, nloc, nnei, 4)
        uu : torch.Tensor
            The atom displancemnt used in interpolation and extrapolation (nframes, nloc, nnei)

        Returns
        -------
        torch.Tensor
            The atomic energy for all local atoms for all frames. (nframes, nloc, nnei)
        """
        a3, a2, a1, a0 = torch.unbind(coef, dim=-1)
        etmp = (a3 * uu + a2) * uu + a1  # this should be elementwise operations.
        ener = (
            etmp * uu + a0
        )  # this energy has the linear extrapolated value when rcut > rmax
        return ener
