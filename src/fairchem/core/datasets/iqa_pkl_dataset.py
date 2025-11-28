from __future__ import annotations
import os
import random
import pickle
from typing import Any, Dict, Iterable, List, Optional

import torch
from fairchem.core.datasets.base_dataset import BaseDataset
from fairchem.core.common.registry import registry
from pathlib import Path
import numpy as np

def angstrom_to_bohr(x: torch.Tensor) -> torch.Tensor:  # 1 Å = 1.8897261245650618 Bohr
    return x * 1.8897261245650618


def bohr_to_angstrom(x: torch.Tensor) -> torch.Tensor:
    return x / 1.8897261245650618


def eV_to_Ht(x: torch.Tensor) -> torch.Tensor:          # 1 Ha = 27.211386245988 eV
    return x / 27.211386245988


def Ht_to_eV(x: torch.Tensor) -> torch.Tensor:
    return x * 27.211386245988

from fairchem.core.datasets.atomic_data import AtomicData

# ---- Canonical key mappings ----

PAIR_KEYS_PKL_TO_CANON = {
    # "V_IQA_Inter(A,B)/2": "pair_E_inter_2",  # (E,) total interatomic / 2
    # "Vne(A,B)/2":         "pair_Vne_2",
    # "Ven(A,B)/2":         "pair_Ven_2",
    # "Vnn(A,B)/2":         "pair_Vnn_2",
    # "VeeC(A,B)/2":        "pair_VeeC_2",
    # "VeeX(A,B)/2":        "pair_VeeX_2",
    # "E_IQA(A)":           "e_iqa_a",

}

SYSTEM_KEYS_PKL_TO_CANON = {
    "e_total": "energy",  # 0-D scalar per system (will become shape [1])
}

# ---- helpers (self-contained) ----

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
    """
    PKL dataset for IQA training:
    - Loads files from a directory.
    - Exposes system-level & per-edge labels under canonical, slash-free keys.
    - Returns AtomicData with exactly the fields it expects.
    """

    def __init__(
        self,
        src: str,
        enforce_consistent_keys: bool = True,
        key_mapping: Optional[Dict[str, str]] = None,
        name: str = "iqa_pkl",
        allow_missing_labels: bool = False,
        bohr2ang: bool = True,
        ht2ev: bool = True,
    ) -> None:
        super().__init__({})  # BaseDataset wants a config object; empty is fine
        self.src = Path(src)
        self.enforce_consistent_keys = enforce_consistent_keys
        self.key_mapping = key_mapping or {}
        self.allow_missing_labels = allow_missing_labels
        self.bohr2ang = bohr2ang
        self.ht2ev = ht2ev
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

        if self.enforce_consistent_keys:
            with open(self.file_paths[0], "rb") as f:
                s0 = pickle.load(f)
            _ = self._extract_available_keys(s0)

    def __len__(self) -> int:
        return len(self.file_paths)

    def _extract_available_keys(self, sample: Any) -> Dict[str, bool]:
        d = _to_mapping(sample)
        _ = _require(d, "pos", "pos", "positions", "R")
        _ = _require(d, "atomic_numbers", "atomic_numbers", "Z", "z", "numbers")
        _ = _require(d, "edge_index", "edge_index", "edges")
        return {k: True for k in d.keys()}

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

        # optional helpers the head may use
        edge_vec    = _first_present(d, "edge_vec")
        edge_length = _first_present(d, "edge_length")
        if edge_vec is not None:    edge_vec = torch.as_tensor(edge_vec)
        if edge_length is not None: edge_length = torch.as_tensor(edge_length)

        # --- labels (system + edge) ---
        labels: Dict[str, torch.Tensor] = {}

        # system energy (scalar -> [1])
        for src_key, canon in SYSTEM_KEYS_PKL_TO_CANON.items():
            if src_key in d:
                labels[canon] = torch.as_tensor(d[src_key], dtype=pos.dtype).view(1)

        # user key mapping (e.g., {"energy": "e_total"})
        for out_key, in_key in self.key_mapping.items():
            if in_key in d:
                t = torch.as_tensor(d[in_key], dtype=pos.dtype)
                labels[out_key] = t.view(1) if t.ndim == 0 else _tensor1d(t)

        # per-edge IQA components -> (E,)
        for pkl_key, canon in PAIR_KEYS_PKL_TO_CANON.items():
            if pkl_key in d:
                t1 = _tensor1d(d[pkl_key])
                if t1 is None:
                    continue
                if t1.numel() != E:
                    raise ValueError(
                        f"Edge label '{pkl_key}' len={t1.numel()} != E={E} for {os.path.basename(path)}"
                    )
                labels[canon] = t1.to(pos.dtype)

        # --- build a VALID AtomicData (constructor accepts only fixed fields) ---
        # For non-PBC molecules, give zeros cell/pbc/offsets and fillers for required fields:
        cell = torch.zeros(1, 3, 3, dtype=pos.dtype)
        pbc  = torch.zeros(1, 3, dtype=torch.bool)
        cell_offsets = torch.zeros(E, 3, dtype=pos.dtype)
        nedges = torch.tensor([E], dtype=torch.long)
        natoms = torch.tensor([N], dtype=torch.long)
        charge = torch.zeros(1, dtype=torch.long)   # system charge (int)
        spin   = torch.zeros(1, dtype=torch.long)   # system spin (int)
        fixed  = torch.zeros(N, dtype=torch.long)   # per-node flags
        tags   = torch.zeros(N, dtype=torch.long)   # per-node tags

        energy = labels.get("energy", None)
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
        if "e_iqa_a" in labels:
            ad.e_iqa_a = Ht_to_eV(labels["e_iqa_a"]) / 1000 if self.ht2ev else labels["e_iqa_a"]  # (E,)

        return ad

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
                    pos = _require(m, "pos", "pos", "positions", "R")
                    n = int(torch.as_tensor(pos).shape[0])
                except Exception:
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
