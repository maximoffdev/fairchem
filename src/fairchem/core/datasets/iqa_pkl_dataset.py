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
        # N = int(container["natoms"].item()) if isinstance(container["natoms"], torch.Tensor) else int(container["natoms"])
        # # fixed
        # fx = container.get("fixed", None)
        # if fx is None:
        #     fx = torch.zeros(N, dtype=torch.long)
        # else:
        #     fx = torch.as_tensor(fx)
        #     if fx.dim() == 0:
        #         fx = fx.view(1)
        #     if fx.dim() == 2 and fx.shape[-1] == 1:
        #         fx = fx.view(-1)
        #     if fx.dim() > 1 and fx.shape[0] != N and fx.numel() == N:
        #         fx = fx.view(-1)
        #     fx = fx.to(dtype=torch.long)
        #     if fx.numel() != N:
        #         if fx.numel() > N:
        #             fx = fx[:N]
        #         else:
        #             fx = torch.cat([fx, torch.zeros(N - fx.numel(), dtype=torch.long)], dim=0)
        # container["fixed"] = fx
        # # tags
        # tg = container.get("tags", None)
        # if tg is None:
        #     tg = torch.zeros(N, dtype=torch.long)
        # else:
        #     tg = torch.as_tensor(tg)
        #     if tg.dim() == 0:
        #         tg = tg.view(1)
        #     if tg.dim() == 2 and tg.shape[-1] == 1:
        #         tg = tg.view(-1)
        #     if tg.dim() > 1 and tg.shape[0] != N and tg.numel() == N:
        #         tg = tg.view(-1)
        #     tg = tg.to(dtype=torch.long)
        #     if tg.numel() != N:
        #         if tg.numel() > N:
        #             tg = tg[:N]
        #         else:
        #             tg = torch.cat([tg, torch.zeros(N - tg.numel(), dtype=torch.long)], dim=0)
        # container["tags"] = tg
        # # --- final dtype guard for validators: ensure long dtypes ---
        # for _k in ("fixed", "tags"):
        #     _v = container.get(_k, None)
        #     if isinstance(_v, torch.Tensor) and _v.dtype is not torch.long:
        #         container[_k] = _v.to(torch.long)
        # z = container.get("atomic_numbers", container.get("z", None))
        # if z is not None:
        #     z = torch.as_tensor(z, dtype=torch.long).view(-1)
        #     if z.numel() != N:
        #         if z.numel() > N:
        #             z = z[:N]
        #         else:
        #             pad = torch.zeros(N - z.numel(), dtype=torch.long)
        #             z = torch.cat([z, pad], dim=0)
        #     container["atomic_numbers"] = z
        # n = int(nat.item())
        # # pbc (1,3) bool
        # pbc = container.get("pbc", torch.tensor([False, False, False], dtype=torch.bool))
        # pbc = torch.as_tensor(pbc, dtype=torch.bool)
        # if pbc.dim()==1:
        #     pbc = pbc.unsqueeze(0)
        # elif pbc.dim()==2 and pbc.shape[0] != 1:
        #     pbc = pbc.view(1, -1)
        # container["pbc"] = pbc
        # # cell (1,3,3) float32
        # cell = container.get("cell", torch.zeros(3,3, dtype=_fdtype))
        # cell = torch.as_tensor(cell, dtype=_fdtype)
        # if cell.dim()==2: cell = cell.unsqueeze(0)
        # container["cell"] = cell
        # # charge (1,) float32
        # charge = container.get("charge", 0.0)
        # charge = torch.as_tensor(float(charge), dtype=torch.float32)
        # if charge.dim()==0: charge = charge.unsqueeze(0)
        # container["charge"] = charge
        # # nedges (1,) long
        # nedges = container.get("nedges", 0)
        # nedges = torch.as_tensor(int(nedges), dtype=torch.long)
        # if nedges.dim()==0: nedges = nedges.unsqueeze(0)
        # container["nedges"] = nedges
        # # cell_offsets (E,3) float
        # co = container.get("cell_offsets", None)
        # if co is None:
        #     container["cell_offsets"] = torch.zeros(0,3, dtype=_fdtype)
        # else:
        #     co = torch.as_tensor(co, dtype=_fdtype)
        #     if co.dim()==1 and co.numel()==3:
        #         co = co.unsqueeze(0)
        #     container["cell_offsets"] = co
        # # --- ensure graph-edge shapes are internally consistent BEFORE enforcing keys ---
        # ei = container.get("edge_index", None)
        # co = container.get("cell_offsets", None)
        # E = None
        # if isinstance(ei, torch.Tensor):
        #     if ei.dtype != torch.long:
        #         ei = ei.to(torch.long)
        #     if ei.dim() == 1:
        #         if ei.numel() == 0:
        #             ei = ei.reshape(2, 0)
        #         else:
        #             if ei.numel() % 2 == 0:
        #                 ei = ei.view(2, -1)
        #             else:
        #                 ei = torch.zeros(2, 0, dtype=torch.long)
        #     elif ei.dim() == 2:
        #         if ei.shape[0] == 2:
        #             pass
        #         elif ei.shape[1] == 2:
        #             ei = ei.t().contiguous()
        #         elif ei.numel() == 0:
        #             ei = ei.reshape(2, 0)
        #         else:
        #             ei = torch.zeros(2, 0, dtype=torch.long)
        #     else:
        #         ei = torch.zeros(2, 0, dtype=torch.long)
        #     container["edge_index"] = ei
        #     E = int(ei.shape[1])
        # elif ei is None:
        #     pass
        # else:
        #     container["edge_index"] = torch.zeros(2, 0, dtype=torch.long)
        #     E = 0
        # if isinstance(co, torch.Tensor):
        #     if co.dtype != _fdtype:
        #         co = co.to(_fdtype)
        #     if co.dim() == 1:
        #         if co.numel() == 3:
        #             co = co.unsqueeze(0)
        #         elif co.numel() == 0:
        #             co = torch.zeros(0, 3, dtype=_fdtype)
        #         else:
        #             co = torch.zeros(0, 3, dtype=_fdtype)
        #     elif co.dim() == 2:
        #         if co.shape[-1] != 3:
        #             co = torch.zeros(co.shape[0], 3, dtype=_fdtype)
        #     else:
        #         co = torch.zeros(0, 3, dtype=_fdtype)
        #     container["cell_offsets"] = co
        #     if E is None:
        #         E = int(co.shape[0])
        # elif co is None:
        #     pass
        # else:
        #     container["cell_offsets"] = torch.zeros(0, 3, dtype=_fdtype)
        #     E = 0
        # if "edge_index" not in container and "cell_offsets" not in container:
        #     container["edge_index"] = torch.zeros(2, 0, dtype=torch.long)
        #     container["cell_offsets"] = torch.zeros(0, 3, dtype=_fdtype)
        #     E = 0
        # else:
        #     if "edge_index" not in container:
        #         E = int(container["cell_offsets"].shape[0]) if E is None else E
        #         container["edge_index"] = torch.zeros(2, E, dtype=torch.long)
        #     if "cell_offsets" not in container:
        #         E = int(container["edge_index"].shape[1]) if E is None else E
        #         container["cell_offsets"] = torch.zeros(E, 3, dtype=_fdtype)
        #     Ei = int(container["edge_index"].shape[1])
        #     Ec = int(container["cell_offsets"].shape[0])
        #     if Ei != Ec:
        #         E = Ei if (Ei and not Ec) else Ec if (Ec and not Ei) else 0
        #         container["edge_index"] = torch.zeros(2, E, dtype=torch.long)
        #         container["cell_offsets"] = torch.zeros(E, 3, dtype=_fdtype)
        # E_final = int(container["edge_index"].shape[1])
        # container["nedges"] = torch.tensor([E_final], dtype=torch.long)
        # sid = container.get("sid", None)
        # if sid is None:
        #     sid = [str(idx) if idx is not None else "0"]
        # elif isinstance(sid, (int, float)):
        #     sid = [str(int(sid))]
        # elif isinstance(sid, (list, tuple)):
        #     sid = [str(sid[0])] if len(sid) > 0 else ["0"]
        # elif isinstance(sid, torch.Tensor):
        #     if sid.numel() > 0:
        #         sid = [str(sid.flatten()[0].item())]
        #     else:
        #         sid = ["0"]
        # else:
        #     sid = [str(sid)]
        # container["sid"] = sid
        # for key, dtype, default in (("spin", torch.float32, 0.0), ("tags", torch.long, 0), ("fixed", torch.long, 0)):
        #     v = container.get(key, None)
        #     if v is None:
        #         container[key] = torch.full((n,), default, dtype=dtype)
        #         continue
        #     v = torch.as_tensor(v, dtype=dtype)
        #     if v.dim()==0:
        #         v = v.repeat(n)
        #     elif v.dim()==2 and v.shape[1]==1:
        #         v = v.squeeze(1)
        #     if v.numel()==1:
        #         v = v.repeat(n)
        #     assert v.numel()==n, f"{key} length {v.numel()} != natoms {n}"
        #     container[key] = v
        # if "num_graphs" in container:
        #     del container["num_graphs"]


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


# =========================================================================================
#                                   IQAPKLDataset (adapter)
# =========================================================================================
@registry.register_dataset("iqa_pkl")
class IQAPKLDataset(BaseDataset):
    """
    ASE-style PKL dataset: reads .pkl files, returns AtomicData objects, fills missing fields with zeros/defaults.
    """
    def __init__(self, src, key_mapping=None, keep_in_memory=False, **kwargs):
        super().__init__({})
        import os
        src_path = os.path.abspath(os.path.expanduser(os.path.expandvars(src)))
        if os.path.isdir(src_path):
            self.file_paths = sorted([
                os.path.join(src_path, f) for f in os.listdir(src_path)
                if f.lower().endswith((".pkl", ".pickle"))
            ])
            self.paths = [Path(src_path)]  # <-- ensure paths is set for metadata
        elif os.path.isfile(src_path) and src_path.lower().endswith((".pkl", ".pickle")):
            self.file_paths = [src_path]
            self.paths = [Path(os.path.dirname(src_path))]  # <-- ensure paths is set for metadata
        else:
            raise ValueError(f"[IQAPKLDataset] bad 'src': {src}")
        if not self.file_paths:
            raise FileNotFoundError(f"[PKL] No .pkl files found in {src}")
        self.num_samples = len(self.file_paths)
        self.key_mapping = key_mapping or {}
        self.keep_in_memory = bool(keep_in_memory)
        self._cache = None
        if self.keep_in_memory:
            self._cache = [self._get_atomicdata(i) for i in range(self.num_samples)]
        
        self.name = kwargs.get("name", "iqa_pkl")
        self.dataset_name = self.name
        self.dataset_names = [self.name]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self._cache is not None:
            return self._cache[idx]
        return self._get_atomicdata(idx)

    def _get_atomicdata(self, idx):
        # Load raw sample
        import torch
        import numpy as np
        import pickle
        from fairchem.core.datasets._utils import rename_data_object_keys
        from fairchem.core.datasets.atomic_data import AtomicData
        raw = None
        with open(self.file_paths[idx], "rb") as f:
            raw = pickle.load(f)
        # Convert to dict if needed
        if hasattr(raw, "to_dict"):
            sample = raw.to_dict()
        elif isinstance(raw, dict):
            sample = dict(raw)
        else:
            sample = dict(raw)
        # Apply key mapping (like ASE)
        if self.key_mapping:
            sample = rename_data_object_keys(sample, self.key_mapping)
        # Ensure 'energy' is present if 'e_total' is present
        if 'energy' not in sample and 'e_total' in sample:
            sample['energy'] = sample['e_total']
        # Ensure energy is always present (default to 0.0 if missing)
        if 'energy' not in sample or sample['energy'] is None:
            sample['energy'] = 0.0
        # Required fields
        natoms = sample["pos"].shape[0] if "pos" in sample else 1
        def get(key, default):
            v = sample.get(key, None)
            if v is not None:
                return v
            if callable(default):
                return default()
            return default
        pos = torch.as_tensor(get("pos", lambda: torch.zeros((natoms, 3), dtype=torch.float32)))
        atomic_numbers = torch.as_tensor(get("atomic_numbers", lambda: torch.ones(natoms, dtype=torch.long)), dtype=torch.long)
        cell = torch.as_tensor(get("cell", lambda: torch.zeros((1, 3, 3), dtype=pos.dtype)))
        pbc = torch.as_tensor(get("pbc", lambda: torch.zeros((1, 3), dtype=torch.bool)))
        natoms_t = torch.as_tensor(get("natoms", lambda: torch.tensor([natoms], dtype=torch.long)))
        edge_index = torch.as_tensor(get("edge_index", lambda: torch.zeros((2, 0), dtype=torch.long)))
        cell_offsets = torch.as_tensor(get("cell_offsets", lambda: torch.zeros((0, 3), dtype=pos.dtype)))
        # Ensure cell_offsets and edge_index have matching number of edges
        num_edges = edge_index.shape[1]
        if cell_offsets.shape[0] != num_edges:
            if num_edges == 0:
                cell_offsets = torch.zeros((0, 3), dtype=cell_offsets.dtype)
            else:
                if cell_offsets.shape[0] > num_edges:
                    cell_offsets = cell_offsets[:num_edges]
                else:
                    pad = torch.zeros((num_edges - cell_offsets.shape[0], 3), dtype=cell_offsets.dtype)
                    cell_offsets = torch.cat([cell_offsets, pad], dim=0)
        nedges = torch.as_tensor(get("nedges", lambda: torch.tensor([edge_index.shape[1]], dtype=torch.long)))
        charge = torch.as_tensor(get("charge", lambda: torch.zeros(1, dtype=torch.float32)))
        spin = torch.as_tensor(get("spin", lambda: torch.zeros(1, dtype=torch.float32)))
        fixed = torch.as_tensor(get("fixed", lambda: torch.zeros(natoms, dtype=torch.long)), dtype=torch.long)
        tags = torch.as_tensor(get("tags", lambda: torch.zeros(natoms, dtype=torch.long)), dtype=torch.long)
        energy = sample.get("energy", None)
        if energy is not None:
            energy = torch.as_tensor(energy)
            if energy.dim() == 0:
                energy = energy.unsqueeze(0)
        forces = sample.get("forces", None)
        if forces is not None:
            forces = torch.as_tensor(forces)
        stress = sample.get("stress", None)
        if stress is not None:
            stress = torch.as_tensor(stress)
            if stress.dim() == 2:
                stress = stress.unsqueeze(0)
        batch = None  # let AtomicData handle default
        sid = sample.get("sid", [str(idx)])
        if isinstance(sid, str):
            sid = [sid]
        elif not isinstance(sid, list):
            sid = [str(sid)]
        dataset = sample.get("dataset", None)
        if dataset is None:
            dataset = 'iqa_pkl'
        if not isinstance(dataset, str):
            dataset = str(dataset)
        dataset_name = dataset
        data = AtomicData(
            pos=pos,
            atomic_numbers=atomic_numbers,
            cell=cell,
            pbc=pbc,
            natoms=natoms_t,
            edge_index=edge_index,
            cell_offsets=cell_offsets,
            nedges=nedges,
            charge=charge,
            spin=spin,
            fixed=fixed,
            tags=tags,
            energy=energy,
            forces=forces,
            stress=stress,
            batch=batch,
            sid=sid,
            dataset=dataset,
        )
        data.dataset_name = dataset_name
        data.validate()
        return data

    @property
    def metadata(self):
        # Look for metadata.npz in the data directory
        if not self.file_paths:
            raise RuntimeError("No PKL files found for metadata lookup.")
        meta_path = os.path.join(os.path.dirname(self.file_paths[0]), "metadata.npz")
        if not os.path.exists(meta_path):
            # Create metadata.npz if it doesn't exist
            write_minimal_metadata(os.path.dirname(self.file_paths[0]), self.file_paths)
        meta = np.load(meta_path)
        if "natoms" not in meta or len(meta["natoms"]) == 0:
            raise RuntimeError(f"metadata.npz at {meta_path} is missing 'natoms' or is empty. Please check your PKL files and rerun.")
        return meta


# class SafeMSELoss(torch.nn.Module):
#     """A wrapper for torch.nn.MSELoss that ignores extra keyword arguments (e.g., mult_mask) and avoids ambiguous bools."""
#     def __init__(self, *args, **kwargs):
#         # Only pass valid args to MSELoss and avoid any torch.Tensor bools or ambiguous bools
#         valid_keys = [
#             'size_average', 'reduce', 'reduction'
#         ]
#         filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys and not isinstance(v, torch.Tensor) and not isinstance(v, (list, tuple, dict))}
#         super().__init__()
#         self.loss = torch.nn.MSELoss(*args, **filtered_kwargs)

#     def forward(self, input, target, *args, **kwargs):
#         # Ignore extra args/kwargs, and avoid ambiguous bools
#         return self.loss(input, target)

