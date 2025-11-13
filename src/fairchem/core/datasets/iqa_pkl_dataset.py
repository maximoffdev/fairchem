from __future__ import annotations
import os
import random
import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from fairchem.core.datasets.base_dataset import BaseDataset
from fairchem.core.common.registry import registry
from pathlib import Path
import numpy as np
# ---- (optional) bring your converters here or stub them if you handle units upstream) ----


def angstrom_to_bohr(x: torch.Tensor) -> torch.Tensor:  # 1 Å = 1.8897261245650618 Bohr
    return x * 1.8897261245650618


def bohr_to_angstrom(x: torch.Tensor) -> torch.Tensor:
    return x / 1.8897261245650618


def eV_to_Ht(x: torch.Tensor) -> torch.Tensor:          # 1 Ha = 27.211386245988 eV
    return x / 27.211386245988


def Ht_to_eV(x: torch.Tensor) -> torch.Tensor:
    return x * 27.211386245988


def write_minimal_metadata(src_dir: str, file_paths: list[str]):
    import pickle
    import torch
    import numpy as np
    from torch_geometric.data import Data
    natoms_list = []
    for fp in file_paths:
        with open(fp, "rb") as f:
            raw = pickle.load(f)
        if isinstance(raw, Data):
            d = raw
        elif isinstance(raw, dict):
            d = Data(**raw)
        else:
            d = Data(**dict(raw))
        # prefer explicit natoms; else infer from pos
        if "natoms" in d:
            n = int(d["natoms"])
        elif "pos" in d and isinstance(d["pos"], torch.Tensor):
            n = int(d["pos"].shape[0])
        else:
            n = 0
        natoms_list.append(n)
    out = Path(src_dir) / "metadata.npz"
    np.savez_compressed(out, natoms=np.array(natoms_list, dtype=np.int32))
    return str(out)

# =========================================================================================
#                                 OnTheFlyPKLDataset
# =========================================================================================


class OnTheFlyPKLDataset(TorchDataset):
    """
    A robust .pkl directory dataset with:
      - full key preservation (no info loss)
      - optional 'enforce_consistent_keys' across all files with auto-filling
      - canonicalization of per-atom scalar shapes (N,1) -> (N,)
      - safe, opt-in unit conversions: pos [Å]<->[Bohr], energy [eV]<->[Ha]

    Each .pkl file must contain either a dict or a torch_geometric.data.Data-like object
    convertible to a dict of tensors/python scalars. 'pos' and 'atomic_numbers' are expected.
    """

    # === Normalization that guarantees FAIRChem batching contract ===
    def _normalize_required(self, container, *, idx=None):
        import torch, numpy as np
        # --- ensure we know the floating dtype to use (match pos if present) ---
        pos = container.get("pos", None)
        if isinstance(pos, torch.Tensor) and pos.is_floating_point():
            _fdtype = pos.dtype
        else:
            _fdtype = torch.float32
        # Ensure natoms as 0-D torch.long
        nat = container.get("natoms")
        if nat is None and isinstance(container.get("pos"), torch.Tensor):
            nat = container["pos"].shape[0]
        if isinstance(nat, torch.Tensor):
            nat = nat.item() if nat.dim() == 0 else int(nat)
        elif nat is None:
            nat = 0
        else:
            nat = int(nat)
        container["natoms"] = nat
        # Ensure energy is present and not None
        if container.get("energy", None) is None:
            container["energy"] = torch.tensor(0.0)
        return container


    def _ensure_sid_vector(self, container, *, idx=None):
        """Ensure container['sid'] is a 1-D torch.long tensor of length natoms.
        Accepts container as a dict-like (sample or Data). Mutates in place.
        """
        import torch, numpy as np
        # infer natoms
        n = None
        nat = container.get("natoms")
        if isinstance(nat, torch.Tensor) and nat.dim()==0:
            n = int(nat.item())
        elif isinstance(nat, (int, np.integer)):
            n = int(nat)
        elif "pos" in container and isinstance(container["pos"], torch.Tensor):
            n = int(container["pos"].shape[0])
        if n is None:
            return  # can't size sid without natoms
        
        # If sid exists, normalize it
        if "sid" in container:
            sid = container["sid"]
            if isinstance(sid, torch.Tensor):
                if sid.dim()==0:
                    container["sid"] = torch.full((n,), int(sid.item()), dtype=torch.long)
                elif sid.dim()==1:
                    if sid.numel()==1:
                        container["sid"] = sid.repeat(n).to(dtype=torch.long)
                    else:
                        assert sid.numel()==n, f"sid length {sid.numel()} != natoms {n}"
                        container["sid"] = sid.to(dtype=torch.long)
                else:
                    # squeeze common (n,1) -> (n,)
                    if sid.dim()==2 and sid.shape[1]==1 and sid.shape[0]==n:
                        container["sid"] = sid.squeeze(1).to(dtype=torch.long)
                    else:
                        raise AssertionError(f"Unexpected sid shape {sid.shape}")
            elif isinstance(sid, (list, tuple)):
                if len(sid)==1:
                    container["sid"] = torch.full((n,), int(sid[0]), dtype=torch.long)
                else:
                    assert len(sid)==n, f"sid length {len(sid)} != natoms {n}"
                    container["sid"] = torch.as_tensor(sid, dtype=torch.long)
            else:
                # scalar-like
                try:
                    base = int(sid)
                    container["sid"] = torch.full((n,), base, dtype=torch.long)
                except Exception:
                    raise AssertionError("sid provided but not interpretable as scalar or vector")
            return
        
        # Else synthesize from other scalar ids or idx
        base_sid = None
        for k in ("system_id", "structure_id", "sid_scalar"):
            if k in container:
                try:
                    base_sid = int(container[k])
                    break
                except Exception:
                    pass
        if base_sid is None and idx is not None:
            base_sid = int(idx)
        if base_sid is None:
            base_sid = 0
        container["sid"] = torch.full((n,), base_sid, dtype=torch.long)


    def __init__(
        self,
        data_dir: str,
        angst2au: bool = False,
        au2angst: bool = False,
        ev2ht: bool = False,
        ht2ev: bool = False,
        elektronn_transform: bool = False,  # kept for compatibility; no-op here
        # prints differences vs first file (debug)
        key_consistency_check: bool = False,
        device: str = "cpu",
        transform=None,
        pre_transform=None,
        enforce_consistent_keys: bool = True,
        scan_limit: Optional[int] = None,
        ignore_keys: Optional[Iterable[str]] = None,
        fill_value_map: Optional[Dict[str, Any]] = None,
        only_basename: Optional[str] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.angst2au = angst2au
        self.au2angst = au2angst
        self.ev2ht = ev2ht
        self.ht2ev = ht2ev
        self.device = device
        self.elektronn_transform = elektronn_transform
        self.key_consistency_check = key_consistency_check
        self.enforce_consistent_keys = enforce_consistent_keys
        self.scan_limit = scan_limit
        self.ignore_keys = set(ignore_keys) if ignore_keys else set()
        self.fill_value_map = fill_value_map or {}

        all_files = [f for f in os.listdir(data_dir)
                     if f.lower().endswith((".pkl", ".pickle"))]
        if only_basename is not None:
            all_files = [f for f in all_files if f == only_basename]
            if not all_files:
                raise FileNotFoundError(f"[PKL] Requested file '{
                    only_basename}' not found in {data_dir}")

        # deterministic order: sort by name
        all_files.sort()
        self.file_paths = [os.path.join(data_dir, f) for f in all_files]
        if not self.file_paths:
            raise FileNotFoundError(f"[PKL] No .pkl files found in {data_dir}")

        # Meta needed for harmonizing shapes when enforce_consistent_keys=True
        self.all_keys: set = set()
        self.key_prototypes: Dict[str, Any] = {}
        # per key: {per_atom, tail_shape, shape, dtype}
        self.key_tensor_meta: Dict[str, Dict[str, Any]] = {}

        if self.enforce_consistent_keys:
            self._scan_and_build_key_metadata()

    # ---------- utils ----------
    def _data_keys(self, sample) -> List[str]:
        if sample is None:
            return []
        if isinstance(sample, dict):
            return list(sample.keys())
        if hasattr(sample, "keys"):
            k = sample.keys
            try:
                return list(k()) if callable(k) else list(k)
            except Exception:
                pass
        try:
            return [k for k in sample.__dict__.keys() if not str(k).startswith("_")]
        except Exception:
            return []

    def _infer_natoms(self, sample: Dict[str, Any]) -> Optional[int]:
        pos = sample.get("pos", None) if isinstance(
            sample, dict) else getattr(sample, "pos", None)
        if isinstance(pos, torch.Tensor):
            return pos.shape[0]
        return None

    def _record_tensor_meta(self, sample: Dict[str, Any], key: str, tensor_value: torch.Tensor):
        """Record per-atom-ness and tail shape; canonicalize (N,1)->(N) for scalar per-atom tensors."""
        natoms = self._infer_natoms(sample)
        per_atom = False
        tail_shape: Tuple[int, ...] = tuple()
        shape = tuple(tensor_value.shape)
        if tensor_value.dim() > 0 and natoms is not None and tensor_value.shape[0] == natoms:
            per_atom = True
            tail_shape = tuple(tensor_value.shape[1:])
            if len(tail_shape) == 1 and tail_shape[0] == 1:
                tensor_value = tensor_value.squeeze(-1)
                tail_shape = ()
                shape = (natoms,)
        self.key_tensor_meta[key] = {
            "per_atom": per_atom,
            "tail_shape": tail_shape,
            "shape": shape,
            "dtype": tensor_value.dtype,
        }
        # keep canonical prototype
        if key in self.key_prototypes and isinstance(self.key_prototypes[key], torch.Tensor):
            proto = self.key_prototypes[key]
            if proto.dim() == 2 and proto.shape[1] == 1 and per_atom:
                self.key_prototypes[key] = proto.squeeze(-1)

    def _maybe_update_meta(self, sample, key: str, value: Any):
        if isinstance(value, torch.Tensor):
            if key not in self.key_tensor_meta:
                self._record_tensor_meta(sample, key, value)

    def _as_data(self, d: Dict[str, Any]) -> Data:
        data = Data()
        for k, v in d.items():
            setattr(data, k, v)
        # Ensure num_graphs is set as an attribute, not a key
        object.__setattr__(data, 'num_graphs', 1)
        # Set dataset as a string attribute only (not a tensor field)
        if hasattr(data, '__dict__'):
            data.__dict__['dataset'] = 'iqa_pkl'
        # Ensure per-graph fields are 1D tensors of length 1
        for key in ["energy", "natoms", "nedges", "charge"]:
            if hasattr(data, key):
                v = getattr(data, key)
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    setattr(data, key, v.unsqueeze(0))
                elif isinstance(v, (int, float)):
                    setattr(data, key, torch.tensor([v]))
        # cell and pbc should be (1, 3, 3) and (1, 3)
        if hasattr(data, "cell"):
            v = getattr(data, "cell")
            if isinstance(v, torch.Tensor) and v.dim() == 2:
                setattr(data, "cell", v.unsqueeze(0))
        if hasattr(data, "pbc"):
            v = getattr(data, "pbc")
            if isinstance(v, torch.Tensor) and v.dim() == 1:
                setattr(data, "pbc", v.unsqueeze(0))
        return data

    def _scan_and_build_key_metadata(self):
        """Scan a subset (or all) files to discover keys and build prototypes for consistent filling."""
        ref_keys: Optional[set] = None
        limit = self.scan_limit or len(self.file_paths)
        for fp in self.file_paths[:limit]:
            with open(fp, "rb") as f:
                raw = pickle.load(f)
            sample = raw if isinstance(raw, dict) else raw.to_dict(
            ) if hasattr(raw, "to_dict") else dict(raw)
            # Record keys
            sample_keys = set()
            for k in self._data_keys(sample):
                if k in self.ignore_keys:
                    continue
                sample_keys.add(k)
                self.all_keys.add(k)
                v = sample[k] if isinstance(
                    sample, dict) else getattr(sample, k, None)
                if isinstance(v, torch.Tensor):
                    v_canon = v.squeeze(-1) if (v.dim() ==
                                                2 and v.shape[1] == 1) else v
                    self._maybe_update_meta(sample, k, v_canon)
                    self.key_prototypes.setdefault(
                        k, v_canon.detach().clone().cpu())
                else:
                    self.key_prototypes.setdefault(k, v)
            if ref_keys is None:
                ref_keys = sample_keys
            elif self.key_consistency_check and (sample_keys != ref_keys):
                missing = ref_keys - sample_keys
                extra = sample_keys - ref_keys
                if missing or extra:
                    print(f"[PKL][Key diff] {os.path.basename(fp)} missing={
                          sorted(list(missing))} extra={sorted(list(extra))}")

    # ---------- dataset protocol ----------
    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Data:
        with open(self.file_paths[idx], "rb") as f:
            raw = pickle.load(f)

        # Convert to dict
        if isinstance(raw, Data):
            sample = raw.to_dict()
        elif isinstance(raw, dict):
            sample = dict(raw)
        else:
            # Fallback: try generic conversion
            sample = dict(raw)

        # Make tensors live on CPU (Dataset level)
        for k, v in list(sample.items()):
            if isinstance(v, torch.Tensor):
                sample[k] = v.detach().cpu()

        # Canonicalize scalar per-atom (N,1)->(N)
        natoms = self._infer_natoms(sample)
        if natoms is not None:
            for k, v in list(sample.items()):
                if isinstance(v, torch.Tensor) and v.dim() == 2 and v.shape[0] == natoms and v.shape[1] == 1:
                    sample[k] = v.squeeze(-1)

        # Unit conversions (non-destructive)
        if "pos" in sample and isinstance(sample["pos"], torch.Tensor):
            if self.angst2au:
                sample["pos"] = angstrom_to_bohr(sample["pos"])
            if self.au2angst:
                sample["pos"] = bohr_to_angstrom(sample["pos"])
        if "e_total" in sample and isinstance(sample["e_total"], torch.Tensor):
            if self.ev2ht:
                sample["e_total"] = eV_to_Ht(sample["e_total"])
            if self.ht2ev:
                sample["e_total"] = Ht_to_eV(sample["e_total"])

        # Enforce consistent keys (fill missing keys with zeros/typed fillers)
        if self.enforce_consistent_keys and self.all_keys:
            data = {**sample}
            for k in self.all_keys:
                if k in self.ignore_keys:
                    continue
                if k in data:
                    # If shape differs but meta exists, try to harmonize simple cases
                    v = data[k]
                    meta = self.key_tensor_meta.get(k, {})
                    if isinstance(v, torch.Tensor) and meta:
                        expected_tail = meta.get("tail_shape", tuple())
                        if meta.get("per_atom", False) and natoms is not None:
                            # Promote 1D -> expected (N,*tail) shapes if needed
                            if expected_tail in [(), (1,)] and v.dim() == 2 and v.shape[1] == 1:
                                data[k] = v.squeeze(-1)
                            elif v.dim() == 1 and expected_tail not in [(), (1,)]:
                                # Expand last dims (best-effort)
                                size = int(expected_tail[0]) if len(
                                    expected_tail) == 1 else None
                                if size is not None and size > 1:
                                    data[k] = v.unsqueeze(-1).expand(
                                        v.shape[0], size).clone()
                    continue  # had the key, keep as-is (after harmonization)
                # Missing -> fill
                custom = self.fill_value_map.get(k, None)
                meta = self.key_tensor_meta.get(k, {})
                if custom is not None:
                    data[k] = custom(sample) if callable(custom) else custom
                else:
                    proto = self.key_prototypes.get(k, None)
                    if isinstance(proto, torch.Tensor):
                        if meta.get("per_atom", False) and natoms is not None:
                            tail = meta.get("tail_shape", tuple())
                            shape = (natoms,) if tail in [
                                (), (1,)] else (natoms, *tail)
                            data[k] = torch.zeros(
                                shape, dtype=meta.get("dtype", proto.dtype))
                        else:
                            p = proto
                            if p.dim() == 2 and p.shape[1] == 1:
                                p = p.squeeze(-1)
                            data[k] = torch.zeros_like(p)
                    elif isinstance(proto, (int, float)):
                        data[k] = type(proto)()
                    else:
                        # Unknown type -> None (preserve key)
                        data[k] = None
            sample = data

        # Ensure/Coerce 'natoms' to 0-D torch.long
        if "natoms" in sample:
            if not isinstance(sample["natoms"], torch.Tensor):
                sample["natoms"] = torch.tensor(int(sample["natoms"]), dtype=torch.long)
            elif sample["natoms"].dim() != 0:
                sample["natoms"] = sample["natoms"].reshape(()).to(dtype=torch.long)
        elif isinstance(sample.get("pos"), torch.Tensor):
            sample["natoms"] = torch.tensor(sample["pos"].shape[0], dtype=torch.long)

        # Ensure sid vector
        self._ensure_sid_vector(sample, idx=idx)
        self._normalize_required(sample, idx=idx)
        # Ensure energy is always present (default to 0.0 if missing)
        if 'energy' not in sample or sample['energy'] is None:
            sample['energy'] = 0.0
        # Return PyG Data (preserves every key)
        data = self._as_data(sample)
        # Ensure num_graphs is set as an attribute, not a key
        if hasattr(data, '__dict__') and 'num_graphs' in data.__dict__:
            pass  # already set
        else:
            object.__setattr__(data, 'num_graphs', 1)
        return data


import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# from fairchem.core.datasets import BaseDataset, registry
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
    ) -> None:
        super().__init__({})  # BaseDataset wants a config object; empty is fine
        self.src = Path(src)
        self.enforce_consistent_keys = enforce_consistent_keys
        self.key_mapping = key_mapping or {}
        self.allow_missing_labels = allow_missing_labels

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

        # stack (E,5) for one-task training
        #want = ["pair_Vne_2", "pair_Ven_2", "pair_Vnn_2", "pair_VeeC_2", "pair_VeeX_2"]
        # want = ["e_iqa_a"]
        # if all(k in labels for k in want):
        #     labels["pair_components"] = torch.stack([labels[k] for k in want], dim=-1)  # (E,5)

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
        # forces/stress aren’t in your PKLs; leave None
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

        # # --- NOW attach extras (constructor won’t take them) ---
        # if edge_vec is not None:
        #     ad.edge_vec = edge_vec
        # if edge_length is not None:
        #     ad.edge_length = edge_length

        # # pairwise labels (ok to add arbitrary fields after init)
        # if "pair_E_inter_2" in labels:
        #     ad.pair_E_inter_2 = labels["pair_E_inter_2"]   # (E,)
        # if "pair_components" in labels:
        #     ad.pair_components = labels["pair_components"] # (E,5)
        
        # ad.pair_Vne_2 = labels["pair_Vne_2"]
        # ad.pair_Ven_2 = labels["pair_Ven_2"]
        # ad.pair_Vnn_2 = labels["pair_Vnn_2"]
        # ad.pair_VeeC_2 = labels["pair_VeeC_2"]
        # ad.pair_VeeX_2 = labels["pair_VeeX_2"]

        ad.dataset_name = self.name
        if "e_iqa_a" in labels:
            ad.e_iqa_a = labels["e_iqa_a"]  # (E,)

        # If you still want the 5 individual scalars accessible (optional):
        # for k in want:
        #     if k in labels:
        #         setattr(ad, k, labels[k])

        # # sanity: at least one label unless allow_missing_labels
        # if not self.allow_missing_labels:
        #     has_any = ("energy" in ad.__dict__) or ("pair_components" in ad.__dict__) or ("pair_E_inter_2" in ad.__dict__)
        #     if not has_any:
        #         raise KeyError(f"No labels found in {path}")

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

    # --- Added helpers so BalancedBatchSampler can probe/generate metadata ---
    def metadata_hasattr(self, attr: str) -> bool:
        """Return True if metadata.npz contains `name`. Ensures metadata is created if missing."""
        if not self.file_paths:
            return False
        first_dir = os.path.dirname(self.file_paths[0])
        meta_path = os.path.join(first_dir, "metadata.npz")
        if not os.path.exists(meta_path):
            # property will generate metadata.npz or raise
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
