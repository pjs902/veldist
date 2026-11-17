"""Data and model velocity histograms of every aperture of every histogram set, at given orbit weights.

    python model_hists.py PRIV_CONFIG MODEL_DIRECTORY OUT.npz WEIGHTS.npz:KEY [WEIGHTS.npz:KEY ...]

One rhs-only read per weight vector (as rescore_v4.py) on a config whose sets use the diagonal statistic, so
DYNAMITE's prepared block holds the data histograms and their errors (both scaled by the aperture's projected mass)
next to the orbit histograms. Saved per set: data, err (n_ap x cells, once), model per weight tag (same scaling),
xbin/ybin; the model histogram does not depend on the statistic, so any weight vector can be evaluated here.
'v4' (the model's own orbit_weights.ecsv) is always included.
"""
import sys

sys.path.insert(0, "/nexus/posix0/MIA-astro-env/nneum/pesmith")
import grid_env  # noqa: F401,E402

import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from astropy.io import ascii  # noqa: E402

import dynamite as dyn  # noqa: E402
from dynamite import weight_solvers as wsmod  # noqa: E402

cfg, mdir, out_path = sys.argv[1:4]
os.chdir(Path(cfg).parent)
c = dyn.config_reader.Configuration(cfg, reset_logging=True, user_logfile=False, reset_existing_output=False)
rows = [i for i, d in enumerate(c.all_models.table["directory"]) if d.rstrip("/") == mdir.rstrip("/")]
assert len(rows) == 1
model = c.all_models.get_model_from_row(rows[0])
orblib = dyn.orblib.LegacyOrbitLibrary(config=c, mod_dir=model.directory_noml, parset=model.parset)
assert orblib._orblib_marker_matches_parset()
kin = c.system.get_unique_triaxial_visible_component().kinematic_data
assert all(getattr(k, "statistic", "diag") == "diag" for k in kin), "use a diagonal-statistic config"
weights = {"v4": np.asarray(ascii.read(model.directory + "orbit_weights.ecsv")["weights"], float)}
for s in sys.argv[4:]:
    f, key = s.rsplit(":", 1)
    weights[Path(f).stem + ":" + key] = np.asarray(np.load(f, allow_pickle=True)[key], float)

CAP = {}
_orig = wsmod.NNLS._prepare_kinematic_block


def _capture(self, kins, orb, tmp, prj, ap0=0):
    obs, err, o = _orig(self, kins, orb, tmp, prj, ap0=ap0)
    if kins.name in ("veldist", "hst_pm", "gaia_pm"):
        n_ap = len(prj)
        flat = o.reshape(len(W), -1)
        h = np.zeros(flat.shape[1])
        for o0 in range(0, len(W), 2048):
            h += W[o0:o0 + 2048] @ np.asarray(flat[o0:o0 + 2048], dtype=np.float64)
        CAP.setdefault(kins.name, []).append((ap0, h.reshape(n_ap, -1), np.reshape(obs, (n_ap, -1)),
                                              np.reshape(err, (n_ap, -1))))
    return obs, err, o


wsmod.NNLS._prepare_kinematic_block = _capture
ws = wsmod.NNLS(config=c, model=model)
save = {}
for tag, W in weights.items():
    CAP.clear()
    ws.construct_gram_and_rhs_blockwise(orblib, rhs_only=True, chi2_w=W)
    for name, parts in CAP.items():
        parts.sort(key=lambda t: t[0])
        save[f"{name}|model|{tag}"] = np.concatenate([p[1] for p in parts]).astype(np.float32)
        if f"{name}|data" not in save:
            save[f"{name}|data"] = np.concatenate([p[2] for p in parts]).astype(np.float32)
            save[f"{name}|err"] = np.concatenate([p[3] for p in parts]).astype(np.float32)
        m, d, e = save[f"{name}|model|{tag}"], save[f"{name}|data"], save[f"{name}|err"]
        print(f"{tag} {name}: {m.shape}, chi2/cell {np.mean(((m - d) / e) ** 2):.3f}", flush=True)
np.savez(out_path, **save)
