import glob, os, pickle, sys
import numpy as np
from multiprocessing import Pool

DIR = "/media/data/qm_dataset/data/Pipeline/pkl/M062X_Jun-cc-pVDZ_HCNOSPClF/HCNOSPClF_combined_datasets_filtered0.001/sumformulasplit/test"
CUTOFF = 8.0


def one(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    pos = np.asarray(d.pos, dtype=np.float64)
    n = pos.shape[0]
    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    deg = (dist <= CUTOFF).sum(1)          # neighbors per atom at 8 A
    full = (n - 1) * np.ones(n)            # complete-graph degree (pkl edge set)
    return n, deg.sum(), int(deg.max()), np.bincount(deg, minlength=256)[:256], full.sum(), d.edge_index.shape[1]


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(DIR, "*.pkl")))
    tot_atoms = tot_deg = tot_full = tot_edges = 0
    gmax = 0
    hist = np.zeros(256, dtype=np.int64)
    nsys = 0
    with Pool(16) as p:
        for n, s, mx, h, fs, ne in p.imap_unordered(one, files, chunksize=32):
            tot_atoms += n; tot_deg += s; gmax = max(gmax, mx)
            hist += h; tot_full += fs; tot_edges += ne; nsys += 1
    print(f"systems              : {nsys}")
    print(f"atoms                : {tot_atoms}")
    print(f"atoms/system         : {tot_atoms/nsys:.3f}")
    print(f"edges (8 A, directed): {tot_deg}")
    print(f"AVG NEIGHBORS @8A    : {tot_deg/tot_atoms:.4f}")
    print(f"max neighbors @8A    : {gmax}")
    print(f"complete-graph avg   : {tot_full/tot_atoms:.4f}  (pkl edges/atom: {tot_edges/tot_atoms:.4f})")
    frac = 1 - hist.cumsum() / tot_atoms
    for k in (20, 30, 40, 50, 60, 80, 100):
        print(f"  frac atoms with >{k} nbrs: {frac[k]:.4f}")
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "deg_hist.npy"), hist)
