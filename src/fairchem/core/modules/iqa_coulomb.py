"""Point-charge Coulomb energies between atom pairs, in the IQA label conventions.

IQA edge labels are stored per *directed* edge as ``X(A, B) / 2`` (summing both
directions recovers the pair value), positions are in Angstrom and energies in eV
after ``IQAPKLDataset`` unit conversion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fairchem.core.datasets.iqa_pkl_dataset import Ht_to_eV, angstrom_to_bohr

if TYPE_CHECKING:
    import torch


def point_charge_pair_energy(
    charge_src: torch.Tensor,
    charge_dst: torch.Tensor,
    distance_ang: torch.Tensor,
    pair_scale: float = 0.5,
    output_in_ev: bool = True,
) -> torch.Tensor:
    """``pair_scale * q_src * q_dst / R`` per edge, in eV (or Hartree).

    Charges in elementary charges, ``distance_ang`` in Angstrom. Computed in the
    dtype of ``distance_ang``.
    """
    energy = (
        pair_scale
        * charge_src.to(distance_ang.dtype)
        * charge_dst.to(distance_ang.dtype)
        / angstrom_to_bohr(distance_ang)
    )
    return Ht_to_eV(energy) if output_in_ev else energy
