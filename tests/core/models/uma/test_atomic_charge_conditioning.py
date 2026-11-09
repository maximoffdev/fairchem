from __future__ import annotations

import pytest
import torch
from ase.build import molecule as get_molecule

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.models.uma.escn_md import eSCNMDBackbone
from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone
from fairchem.core.models.uma.nn.embedding_dev import ChgSpinEmbedding

BACKBONE_KWARGS = {
    "max_num_elements": 100,
    "sphere_channels": 8,
    "lmax": 2,
    "mmax": 2,
    "otf_graph": True,
    "edge_channels": 8,
    "num_distance_basis": 8,
    "num_layers": 2,
    "hidden_channels": 8,
    "dataset_list": ["iqa_pkl"],
    "always_use_pbc": False,
}


def _atomic_data(name: str, atomic_charges: list[float] | None) -> AtomicData:
    data = AtomicData.from_ase(
        input_atoms=get_molecule(name),
        max_neigh=25,
        radius=6,
        task_name="iqa_pkl",
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )
    if atomic_charges is not None:
        data.atomic_charges = torch.tensor(atomic_charges)
    return data


def _batch(with_atomic_charges: bool) -> AtomicData:
    return atomicdata_list_to_batch(
        [
            _atomic_data("H2O", [-0.8, 0.4, 0.4] if with_atomic_charges else None),
            _atomic_data(
                "NH3", [-1.0, 0.3, 0.35, 0.35] if with_atomic_charges else None
            ),
        ]
    )


@pytest.mark.parametrize("embedding_type", ["pos_emb", "lin_emb"])
def test_atomic_charge_embedding_is_per_atom_and_continuous(
    embedding_type: str,
) -> None:
    emb = ChgSpinEmbedding(embedding_type, "atomic_charge", 8, grad=True)
    out = emb(torch.tensor([-0.8, 0.4, 0.41]))
    assert out.shape == (3, 8)
    assert not torch.allclose(out[1], out[2])


def test_atomic_charge_embedding_rejects_rand_emb() -> None:
    with pytest.raises(ValueError, match="rand_emb"):
        ChgSpinEmbedding("rand_emb", "atomic_charge", 8, grad=True)


def test_atomic_charge_conditioning_is_per_atom() -> None:
    torch.manual_seed(0)
    backbone = eSCNMDBackbone(charge_conditioning="atomic", **BACKBONE_KWARGS)
    batch = _batch(with_atomic_charges=True)
    sys_emb, node_emb = backbone.csd_embedding(batch)
    assert sys_emb.shape == (2, 8)
    assert node_emb.shape == (7, 8)
    # the system embedding is the mean over that system's atoms
    torch.testing.assert_close(sys_emb[0], node_emb[:3].mean(dim=0))

    out = backbone(batch.clone())["node_embedding"]
    perturbed = batch.clone()
    perturbed.atomic_charges[0] += 0.1
    out_perturbed = backbone(perturbed)["node_embedding"]
    # changing one atom's charge in H2O must not leak into NH3
    assert not torch.allclose(out[:3], out_perturbed[:3])
    torch.testing.assert_close(out[3:], out_perturbed[3:])


def test_atomic_charge_conditioning_requires_atomic_charges() -> None:
    backbone = eSCNMDBackbone(charge_conditioning="atomic", **BACKBONE_KWARGS)
    with pytest.raises(KeyError, match="atomic_charges"):
        backbone(_batch(with_atomic_charges=False))


def test_total_charge_conditioning_rejects_atomic_charges() -> None:
    backbone = eSCNMDBackbone(charge_conditioning="total", **BACKBONE_KWARGS)
    with pytest.raises(ValueError, match="atomic_charges"):
        backbone(_batch(with_atomic_charges=True))


def test_moe_backbone_routes_on_pooled_atomic_charge_embedding() -> None:
    torch.manual_seed(0)
    backbone = eSCNMDMoeBackbone(
        num_experts=2,
        use_composition_embedding=True,
        charge_conditioning="atomic",
        **BACKBONE_KWARGS,
    )
    out = backbone(_batch(with_atomic_charges=True))
    assert out["node_embedding"].shape[0] == 7
    assert backbone.global_mole_tensors.expert_mixing_coefficients.shape == (2, 2)


def test_no_charge_conditioning_ignores_charge() -> None:
    torch.manual_seed(0)
    backbone = eSCNMDBackbone(charge_conditioning="none", **BACKBONE_KWARGS)
    assert backbone.charge_embedding is None
    # spin + dataset only
    assert backbone.mix_csd.in_features == 2 * BACKBONE_KWARGS["sphere_channels"]
    batch = _batch(with_atomic_charges=False)
    out = backbone(batch.clone())["node_embedding"]
    charged = batch.clone()
    charged.charge = charged.charge + 1
    torch.testing.assert_close(out, backbone(charged)["node_embedding"])


def test_no_charge_conditioning_rejects_atomic_charges() -> None:
    backbone = eSCNMDBackbone(charge_conditioning="none", **BACKBONE_KWARGS)
    with pytest.raises(ValueError, match="atomic_charges"):
        backbone(_batch(with_atomic_charges=True))
