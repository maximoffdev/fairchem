from __future__ import annotations
import logging
import os
import random
import pickle
from typing import Any, Dict, Iterable, List, Literal, Optional

import torch
import torch_geometric
from fairchem.core.datasets.base_dataset import BaseDataset
from fairchem.core.common.registry import registry
from pathlib import Path
import numpy as np
from fairchem.core.datasets.atomic_data import AtomicData

def angstrom_to_bohr(x: torch.Tensor) -> torch.Tensor:  # 1 Å = 1.8897261245650618 Bohr
    return x * 1.8897261245650618


def bohr_to_angstrom(x: torch.Tensor) -> torch.Tensor:
    return x / 1.8897261245650618


def eV_to_Ht(x: torch.Tensor) -> torch.Tensor:          # 1 Ha = 27.211386245988 eV
    return x / 27.211386245988


def Ht_to_eV(x: torch.Tensor) -> torch.Tensor:
    return x * 27.211386245988

def Ht_per_Bohr_to_eV_per_A(x: torch.Tensor) -> torch.Tensor:
    # AIMAll forces are atomic units (Hartree/Bohr), matching the Bohr positions;
    # the model works in eV/Angstrom. 1 Ha/Bohr = 27.211386... eV / 0.529177... A.
    return x * 27.211386245988 * 1.8897261245650618


# 1 atomic unit of dipole moment (e * Bohr) = 2.5417464519... Debye
AU_TO_DEBYE = 2.5417464519485975


def au_to_debye(x: torch.Tensor) -> torch.Tensor:
    return x * AU_TO_DEBYE


def debye_to_au(x: torch.Tensor) -> torch.Tensor:
    return x / AU_TO_DEBYE


# Labels whose unit is not a plain energy, keyed by the *out_key* of key_mapping.
# Everything not listed here is treated as Hartree (see __getitem__).
FORCE_KEYS = frozenset({"iqa_forces_direct", "iqa_forces_grad"})
DIPOLE_KEYS = frozenset(
    {"dipole_intra", "dipole_vector", "dipole_bond", "dipole_scalar"}
)
# Labels that carry no unit and must be passed through untouched. Atomic charges
# q(A) are in electrons; without this they fall through to the Hartree -> eV
# branch in __getitem__ and come out scaled by 27.2.
DIMENSIONLESS_KEYS = frozenset({"iqa_charge"})
# Max deviation of sum(q(A)) from an integer before a sample is rejected as broken.
ATOMIC_CHARGE_SUM_TOL = 0.1
# Per-atom AIM integral of the Laplacian of the density. It vanishes for an exactly
# integrated basin, so |L(A)| measures the integration error of the atom's labels.
LAGRANGIAN_KEY = "L(A)"


def label_level(in_key: str) -> Literal["system", "atom", "edge"]:
    """Level of a pkl label, read off the AIMAll naming: 'X(A,B)' labels are per
    directed edge, other 'X(A...)' labels per atom, and the rest per system."""
    if "(A,B)" in in_key:
        return "edge"
    if "(A" in in_key:
        return "atom"
    return "system"

def _to_mapping(sample: Any) -> Dict[str, Any]:
    if isinstance(sample, dict):
        return sample
    if hasattr(sample, "keys") and callable(getattr(sample, "keys")):
        try:
            keys = list(sample.keys())
            return {k: getattr(sample, k) for k in keys}
        except Exception:
            pass
    out = {}
    for k in dir(sample):
        if k.startswith("_"):
            continue
        try:
            v = getattr(sample, k)
        except Exception:
            continue
        if callable(v):
            continue
        out[k] = v
    return out

def _first_present(d: Dict[str, Any], *cands: str):
    for k in cands:
        if k in d:
            return d[k]
    return None

def _require(d: Dict[str, Any], logical: str, *cands: str):
    v = _first_present(d, *cands)
    if v is None:
        avail = sorted(list(d.keys()))
        raise KeyError(
            f"Required key '{logical}' missing. Tried aliases {cands}. "
            f"Available keys (first 50): {avail[:50]}{' ...' if len(avail) > 50 else ''}"
        )
    return v

def _tensor1d(x: Any) -> Optional[torch.Tensor]:
    if x is None:
        return None
    t = torch.as_tensor(x)
    if t.ndim == 2 and t.shape[1] == 1:
        t = t.squeeze(-1)
    return t


@registry.register_dataset("iqa_pkl")
class IQAPKLDataset(BaseDataset):
    def __init__(
        self,
        src: str,
        key_mapping: Optional[Dict[str, str]] = None,
        name: str = "iqa_pkl",
        allow_missing_labels: bool = False,
        bohr2ang: bool = True,
        ht2ev: bool = True,
        force_ht2ev: bool = True,
        au2debye: bool = True,
        charge_key: str | None = None,
        atomic_charge_key: str | None = None,
        sigma_atomic_charge: float = 0.0,
        lagrangian_cutoff: float | None = None,
    ) -> None:
        super().__init__({})  # BaseDataset wants a config object; empty is fine
        self.src = Path(src)
        self.key_mapping = key_mapping or {}
        self.allow_missing_labels = allow_missing_labels
        self.bohr2ang = bohr2ang
        self.ht2ev = ht2ev
        self.force_ht2ev = force_ht2ev
        self.au2debye = au2debye
        # Charge conditioning is either on the total molecular charge (charge_key,
        # default 'q_total') -> AtomicData.charge, or on per-atom charges
        # (atomic_charge_key, e.g. 'q(A)') -> AtomicData.atomic_charges, which requires
        # charge_conditioning='atomic' on the backbone. Note the pkls also carry a
        # 'Charge' key, but that is the per-atom nuclear charge Z -- do not point
        # either key at it.
        if charge_key is not None and atomic_charge_key is not None:
            raise ValueError(
                f"charge_key ('{charge_key}') and atomic_charge_key "
                f"('{atomic_charge_key}') are mutually exclusive"
            )
        if charge_key is None and atomic_charge_key is None:
            charge_key = "q_total"
        self.charge_key = charge_key
        self.atomic_charge_key = atomic_charge_key
        # Std (e) of Gaussian noise on the atomic charge input, for robustness to
        # imperfect charges at inference. Set it on the training dataset only.
        if sigma_atomic_charge < 0:
            raise ValueError(
                f"sigma_atomic_charge must be >= 0, got {sigma_atomic_charge}"
            )
        if sigma_atomic_charge > 0 and atomic_charge_key is None:
            raise ValueError("sigma_atomic_charge > 0 requires atomic_charge_key")
        self.sigma_atomic_charge = sigma_atomic_charge
        # Atoms with |L(A)| above this (a.u.) get NaN atom labels, and so do the
        # labels of their edges; the torch.isfinite output masks in mlip_unit then
        # drop them from loss and metrics. None disables the masking.
        if lagrangian_cutoff is not None and lagrangian_cutoff <= 0:
            raise ValueError(
                f"lagrangian_cutoff must be > 0 or None, got {lagrangian_cutoff}"
            )
        self.lagrangian_cutoff = lagrangian_cutoff
        self._warned_missing_charge = False
        self.name = name
        self.dataset_name = name
        self.dataset_names = [name]

        self.paths: List[Path] = [self.src]
        self.file_paths: List[str] = []
        for root, _, fnames in os.walk(self.src):
            for fn in fnames:
                if fn.endswith(".pkl"):
                    p = Path(root) / fn
                    if p.stat().st_size > 0:
                        self.file_paths.append(str(p))
        self.file_paths.sort()
        if not self.file_paths:
            raise FileNotFoundError(f"No .pkl files found under {self.src}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> AtomicData:
        path = self.file_paths[idx]
        with open(path, "rb") as f:
            raw = pickle.load(f)
        d = _to_mapping(raw)

        # --- base graph (strict shapes/dtypes) ---
        pos = torch.as_tensor(_require(d, "pos", "pos", "positions", "R"),
                              dtype=torch.get_default_dtype())          # (N,3)
        if self.bohr2ang:
            pos = bohr_to_angstrom(pos)
        Z = torch.as_tensor(_require(d, "atomic_numbers", "atomic_numbers", "Z", "z", "numbers"),
                            dtype=torch.long).view(-1)                  # (N,)
        edge_index = torch.as_tensor(_require(d, "edge_index", "edge_index", "edges"),
                                     dtype=torch.long)                  # (2,E)
        N = int(pos.shape[0]); E = int(edge_index.shape[1])

        # --- labels (system + edge) ---
        labels: Dict[str, torch.Tensor] = {}
        #new logic for tensor labels
        for out_key, in_key in self.key_mapping.items():
            if in_key not in d:
                # Silently dropping the label yields an AtomicData whose key set differs
                # from its batch mates; atomicdata_list_to_batch() takes the key set from
                # the first sample only, so this resurfaces much later as an opaque
                # "AtomicData object has no attribute '<out_key>'" during collation.
                if not self.allow_missing_labels:
                    raise KeyError(
                        f"key_mapping entry '{out_key}' -> '{in_key}' is missing from {path}. "
                        f"All samples must carry the same labels. Either drop the entry from "
                        f"the dataset config, remove/repair the file, or pass "
                        f"allow_missing_labels=true to skip it (batching will then fail if "
                        f"other samples in the same batch do have the label)."
                    )
                continue

            val = d[in_key]
            t = torch.as_tensor(val, dtype=pos.dtype)

            # If per-atom vector (N,3) -> keep
            if t.ndim == 2 and t.shape[1] == 3:
                labels[out_key] = t
            #if scalar -> tensor with shape (1,)
            elif t.ndim == 1 or (t.ndim == 2 and t.shape[1] == 1):
                labels[out_key] = _tensor1d(t)
            # single scalar
            else:
                labels[out_key] = t.view(1) if t.ndim == 0 else t

        if self.lagrangian_cutoff is not None:
            self._mask_poorly_integrated_atoms(d, path, labels, edge_index, N)

        # user key mapping (e.g., {"energy": "e_total"})
        # for out_key, in_key in self.key_mapping.items():          #old version, not suited for tensors
        #     if in_key in d:
        #         t = torch.as_tensor(d[in_key], dtype=pos.dtype)
        #         labels[out_key] = t.view(1) if t.ndim == 0 else _tensor1d(t)

        # --- build a VALID AtomicData (constructor accepts only fixed fields) ---
        # For non-PBC molecules, give zeros cell/pbc/offsets and fillers for required fields:
        cell = torch.zeros(1, 3, 3, dtype=pos.dtype)
        pbc  = torch.zeros(1, 3, dtype=torch.bool)
        cell_offsets = torch.zeros(E, 3, dtype=pos.dtype)
        nedges = torch.tensor([E], dtype=torch.long)
        natoms = torch.tensor([N], dtype=torch.long)
        
        atomic_charges = None
        if self.atomic_charge_key is not None:
            atomic_charges = self._load_atomic_charges(d, path, N, pos.dtype)
            charge = self._total_charge_from_atomic(atomic_charges, path)
            if self.sigma_atomic_charge > 0:
                # zero-mean per molecule, so the noisy charges keep the total charge
                noise = torch.randn_like(atomic_charges) * self.sigma_atomic_charge
                atomic_charges = atomic_charges + noise - noise.mean()
        elif (q_total := _first_present(d, self.charge_key)) is not None:
            # Total molecular charge -> AtomicData.charge, consumed by ChgSpinEmbedding.
            charge = torch.tensor([int(q_total)], dtype=torch.long)
        else:
            # Silently treating charged systems as neutral would poison the
            # conditioning, so make the fallback loud (once per worker).
            if not self._warned_missing_charge:
                self._warned_missing_charge = True
                logging.warning(
                    f"charge_key '{self.charge_key}' not found in {path}; "
                    f"defaulting charge to 0. Available keys: {sorted(d.keys())[:50]}"
                )
            charge = torch.zeros(1, dtype=torch.long)   # default to neutral

        # All systems in this dataset are closed-shell singlets.
        spin   = torch.zeros(1, dtype=torch.long)   # system spin (int)
        fixed  = torch.zeros(N, dtype=torch.long)   # per-node flags
        tags   = torch.zeros(N, dtype=torch.long)   # per-node tags

        energy = labels.get("energy", None)

        if energy is not None:
            energy = Ht_to_eV(energy) if self.ht2ev else energy  # (E,)
        
        ad = AtomicData(
            pos=pos,
            atomic_numbers=Z,
            cell=cell,
            pbc=pbc,
            natoms=natoms,
            edge_index=edge_index,
            cell_offsets=cell_offsets,
            nedges=nedges,
            charge=charge,
            spin=spin,
            fixed=fixed,
            tags=tags,
            energy=energy if energy is not None else None,  # (1,)
            forces=None,
            stress=None,
            batch=None,
            sid=str(idx),
            dataset=self.name,
        )

        ad.dataset_name = self.name
        if atomic_charges is not None:
            ad.atomic_charges = atomic_charges

        for out_key, val in labels.items():
            if out_key == "energy":
                continue  # Already handled

            # Apply unit conversion if requested. Dispatch on the label's *name*, not
            # its shape: energy labels are Hartree at every level (system (1,),
            # per-atom (N,) or per-edge (E,)), and dipoles come as both vectors
            # (N, 3) and magnitudes (N,), so a shape test would mislabel both.
            if out_key in DIMENSIONLESS_KEYS:
                pass
            elif out_key in FORCE_KEYS:
                val = Ht_per_Bohr_to_eV_per_A(val) if self.force_ht2ev else val
            elif out_key in DIPOLE_KEYS:
                # AIMAll dipoles (Mu(A), Mu_Intra(A), Mu_Bond(A) and their
                # magnitudes) are atomic units, e * Bohr -> Debye.
                val = au_to_debye(val) if self.au2debye else val
            elif val.ndim == 2 and val.shape[1] == 3:
                # Some other per-atom vector (e.g. the EhF(A) electric field). Its
                # unit is not known here, so leave it in the pkl's atomic units;
                # add it to FORCE_KEYS/DIPOLE_KEYS if it ever becomes a target.
                pass
            else:
                val = Ht_to_eV(val) if self.ht2ev else val

            setattr(ad, out_key, val)

        return ad

    def _mask_poorly_integrated_atoms(
        self,
        d: dict[str, Any],
        path: str,
        labels: dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        natoms: int,
    ) -> None:
        """Set to NaN, in place in `labels`, the atom labels of atoms with
        |L(A)| > lagrangian_cutoff and the edge labels of edges touching them.
        System labels (e.g. e_total) do not come from the AIM integration and are
        kept."""
        if LAGRANGIAN_KEY not in d:
            raise KeyError(
                f"lagrangian_cutoff is set but '{LAGRANGIAN_KEY}' is missing from "
                f"{path}. Available keys: {sorted(d.keys())[:50]}"
            )
        lagrangian = torch.as_tensor(d[LAGRANGIAN_KEY]).reshape(-1)
        if lagrangian.shape[0] != natoms:
            raise ValueError(
                f"'{LAGRANGIAN_KEY}' in {path} has {lagrangian.shape[0]} entries, "
                f"expected one per atom ({natoms})"
            )
        bad_atoms = lagrangian.abs() > self.lagrangian_cutoff
        if not bad_atoms.any():
            return
        masks = {
            "atom": bad_atoms,
            "edge": bad_atoms[edge_index[0]] | bad_atoms[edge_index[1]],
        }
        for out_key, in_key in self.key_mapping.items():
            level = label_level(in_key)
            if level == "system" or out_key not in labels:
                continue
            mask = masks[level]
            val = labels[out_key]
            if val.shape[0] != mask.shape[0]:
                raise ValueError(
                    f"'{in_key}' in {path} is named as a per-{level} label but has "
                    f"{val.shape[0]} rows, expected {mask.shape[0]}"
                )
            val = val.clone()
            val[mask] = float("nan")
            labels[out_key] = val

    def _load_atomic_charges(
        self, d: Dict[str, Any], path: str, natoms: int, dtype: torch.dtype
    ) -> torch.Tensor:
        if self.atomic_charge_key not in d:
            raise KeyError(
                f"atomic_charge_key '{self.atomic_charge_key}' not found in {path}. "
                f"Available keys: {sorted(d.keys())[:50]}"
            )
        q = torch.as_tensor(d[self.atomic_charge_key], dtype=dtype).reshape(-1)
        if q.shape[0] != natoms:
            raise ValueError(
                f"atomic_charge_key '{self.atomic_charge_key}' in {path} has "
                f"{q.shape[0]} entries, expected one per atom ({natoms})"
            )
        return q

    @staticmethod
    def _total_charge_from_atomic(
        atomic_charges: torch.Tensor, path: str
    ) -> torch.Tensor:
        # AtomicData.charge is a required field; derive it from the atomic charges,
        # which sum to the integer total charge up to the AIM integration error.
        q_sum = float(atomic_charges.sum())
        q_total = round(q_sum)
        if abs(q_sum - q_total) > ATOMIC_CHARGE_SUM_TOL:
            raise ValueError(
                f"Atomic charges in {path} sum to {q_sum:.4f}, which is more than "
                f"{ATOMIC_CHARGE_SUM_TOL} e off an integer total charge"
            )
        return torch.tensor([q_total], dtype=torch.long)

    @property
    def metadata(self):
        # Look for metadata.npz in the data directory
        if not self.file_paths:
            raise RuntimeError("No PKL files found for metadata lookup.")
        first_dir = os.path.dirname(self.file_paths[0])
        meta_path = os.path.join(first_dir, "metadata.npz")
        if not os.path.exists(meta_path):
            natoms = []
            filenames = []
            for p in self.file_paths:
                try:
                    with open(p, "rb") as f:
                         s = pickle.load(f)
                    m = _to_mapping(s)
                    if hasattr(s, 'natoms'):
                        n = int(s.natoms)
                    else:
                        pos = _require(m, "pos", "pos", "positions", "R")
                        n = int(torch.as_tensor(pos).shape[0])
                    if n == 0:
                        print(f"Warning: file {p} has zero atoms. Check if it's a valid PKL file.")
                except Exception as e:
                    print(f"Error loading {p} for metadata: {e}. Setting natoms=0.")
                    n = 0
                natoms.append(n)
                filenames.append(os.path.relpath(p, first_dir))
            np.savez(meta_path, natoms=np.array(natoms, dtype=np.int64), filenames=np.array(filenames))
        meta = np.load(meta_path)
        if ("natoms" not in getattr(meta, "files", [])) or len(meta["natoms"]) == 0:
            raise RuntimeError(
                f"metadata.npz at {meta_path} is missing 'natoms' or is empty. Please check your PKL files and rerun."
            )
        return meta

    def metadata_hasattr(self, attr: str) -> bool:
        """Return True if metadata.npz contains `name`. Ensures metadata is created if missing."""
        if not self.file_paths:
            return False
        first_dir = os.path.dirname(self.file_paths[0])
        meta_path = os.path.join(first_dir, "metadata.npz")
        if not os.path.exists(meta_path):
            try:
                _ = self.metadata
            except Exception:
                return False
        try:
            meta = np.load(meta_path)
            return attr in getattr(meta, "files", [])
        except Exception:
            return False

    def get_metadata(self, attr: str, idx: Optional[Iterable[int]] = None):
        """Return metadata[name] or metadata[name][indices] (numpy array)."""
        meta = self.metadata  # ensures file exists and validated
        if attr not in getattr(meta, "files", []):
            raise KeyError(f"metadata has no key '{attr}'")
        arr = np.array(meta[attr])
        if idx is None:
            return arr
        idx = np.asarray(list(idx), dtype=int)
        return arr[idx]
