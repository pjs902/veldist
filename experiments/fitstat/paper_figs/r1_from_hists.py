"""r1 input from MODEL_HISTS.npz: per-aperture data and model moments in extract_resid.py's layout.

usage: python r1_from_hists.py PACK_DIR MODEL_HISTS.npz TAG OUT.npz
Data and model moments from the same histograms and the same estimator (r11_dataset_maps: unit sum,
Sheppard-corrected dispersions): veldist v/sigma, PM pmR, pmT (counter-clockwise E->N, the extract's sign),
sigma_R, sigma_T. LVM is not in MODEL_HISTS (its model needs an orbit-library read), so it is omitted.
"""
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

from d2_maps import pm_moments

P, Z, TAG, OUT = Path(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3], sys.argv[4]
vc = np.asarray(Table.read(P / "veldist_kinematics.ecsv").meta["vcent"], float)
h = np.median(np.diff(vc))
out = {"desc": np.array([f"r1 moments from {sys.argv[2]} tag {TAG}; data and model from the same histograms."])}
for kind, H in (("data", Z["veldist|data"]), ("model", Z[f"veldist|model|{TAG}"])):
    w = H / H.sum(1, keepdims=True)
    m = w @ vc
    out[f"veldist_{kind}_v"] = m
    out[f"veldist_{kind}_sigma"] = np.sqrt(np.maximum((w * (vc - m[:, None]) ** 2).sum(1) - h ** 2 / 12, 0))
for s, stem in (("hst_pm", "hst_veldist"), ("gaia_pm", "gaia_veldist")):
    for kind, key in (("data", f"{s}|data"), ("model", f"{s}|model|{TAG}")):
        d = pm_moments(P / f"{stem}_pm_2dhist.npz", Z[key])
        out.update({f"{s}_{kind}_pmR": d["mr"], f"{s}_{kind}_pmT": -d["mt"],
                    f"{s}_{kind}_sigma_R": d["sr"], f"{s}_{kind}_sigma_T": d["st"]})
np.savez(OUT, **out)
print("saved", OUT)
