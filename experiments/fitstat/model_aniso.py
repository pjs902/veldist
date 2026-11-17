"""Model velocity ellipsoid (sigma_R, sigma_T, mean mu_T) at every HST and Gaia aperture, at given orbit weights.

    python model_aniso.py PRIV_CONFIG MODEL_DIRECTORY OUT.npz WEIGHTS.npz:KEY [WEIGHTS.npz:KEY ...]

One rhs-only read per weight vector (DYNAMITE's own streamed read, as rescore_v4.py) captures the model's intrinsic
PM histograms h (n_ap x K x K, mass units) of every PM set. Per aperture: normalised h on the set's cell centres
(+-vxrange half-width, pack axes pm1/pm2, x-East), its mean and covariance, rotated to (E, N) with a_hat/b_hat
(PA 104.3) and projected on r_hat, t_hat at the aperture's (xbin, ybin). 'v4' (the model's own orbit_weights.ecsv)
is always included. Saved per set and weight tag: x, y, sR, sT, muT (km/s).
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
wspecs = sys.argv[4:]
os.chdir(Path(cfg).parent)
c = dyn.config_reader.Configuration(cfg, reset_logging=True, user_logfile=False, reset_existing_output=False)
rows = [i for i, d in enumerate(c.all_models.table["directory"]) if d.rstrip("/") == mdir.rstrip("/")]
assert len(rows) == 1
model = c.all_models.get_model_from_row(rows[0])
orblib = dyn.orblib.LegacyOrbitLibrary(config=c, mod_dir=model.directory_noml, parset=model.parset)
assert orblib._orblib_marker_matches_parset()
kin = c.system.get_unique_triaxial_visible_component().kinematic_data
pms = [k for k in kin if k.name in ("hst_pm", "gaia_pm")]
INDIR = Path(c.settings.io_settings["input_directory"])
FILES = {"hst_pm": "hst_veldist_pm_2dhist.npz", "gaia_pm": "gaia_veldist_pm_2dhist.npz"}
PA = np.radians(104.3)
AH, BH = np.array([np.sin(PA), np.cos(PA)]), np.array([-np.cos(PA), np.sin(PA)])

weights = {"v4": np.asarray(ascii.read(model.directory + "orbit_weights.ecsv")["weights"], float)}
for s in wspecs:
    f, key = s.rsplit(":", 1)
    weights[Path(f).stem + ":" + key] = np.asarray(np.load(f, allow_pickle=True)[key], float)

CAP = {}
_orig = wsmod.NNLS._prepare_kinematic_block


def _capture(self, kins, orb, tmp, prj, ap0=0):
    obs, err, o = _orig(self, kins, orb, tmp, prj, ap0=ap0)
    if kins.name in ("hst_pm", "gaia_pm"):
        flat = o.reshape(len(W), -1)
        h = np.zeros(flat.shape[1])
        for o0 in range(0, len(W), 2048):
            h += W[o0:o0 + 2048] @ np.asarray(flat[o0:o0 + 2048], dtype=np.float64)
        CAP.setdefault(kins.name, []).append((ap0, h.reshape(len(prj), -1)))
    return obs, err, o


wsmod.NNLS._prepare_kinematic_block = _capture
ws = wsmod.NNLS(config=c, model=model)
save = {}
for tag, W in weights.items():
    CAP.clear()
    ws.construct_gram_and_rhs_blockwise(orblib, rhs_only=True, chi2_w=W)
    for k in pms:
        h = np.concatenate([t[1] for t in sorted(CAP[k.name], key=lambda t: t[0])])
        z = np.load(INDIR / FILES[k.name])  # the pack file the set was read from: vxrange = HALF width
        assert (z["binID_dynamite"] == np.arange(1, len(z["xbin"]) + 1)).all() and len(z["xbin"]) == len(h)
        xb, yb = np.asarray(z["xbin"], float), np.asarray(z["ybin"], float)
        K = int(round(np.sqrt(h.shape[1])))
        assert K == z["PM_2dhist"].shape[1]
        half = float(z["vxrange"])
        cc = np.linspace(-half, half, K + 1)
        cc = 0.5 * (cc[1:] + cc[:-1])
        P1, P2 = [g.ravel() for g in np.meshgrid(cc, cc, indexing="ij")]
        w = h / h.sum(1, keepdims=True)
        m1, m2 = w @ P1, w @ P2
        c11 = w @ P1 ** 2 - m1 ** 2
        c22 = w @ P2 ** 2 - m2 ** 2
        c12 = w @ (P1 * P2) - m1 * m2
        r = np.maximum(np.hypot(xb, yb), 1e-9)
        rh = np.c_[xb, yb] / r[:, None]
        th = np.c_[-yb, xb] / r[:, None]
        out = {}
        for nm, u in (("sR", rh), ("sT", th)):
            a, b = u @ AH, u @ BH  # components of u on the pm1 / pm2 axes
            out[nm] = np.sqrt(a * a * c11 + 2 * a * b * c12 + b * b * c22)
        mE, mN = m1 * AH[0] + m2 * BH[0], m1 * AH[1] + m2 * BH[1]
        out["muT"] = th[:, 0] * mE + th[:, 1] * mN
        for nm, v in dict(x=xb, y=yb, **out).items():
            save[f"{k.name}|{tag}|{nm}"] = v
        print(f"{tag} {k.name}: {len(xb)} apertures, K {K}, half {half:.2f}; median sR/sT "
              f"{np.median(out['sR'] / out['sT']):.3f}", flush=True)
np.savez(out_path, **save)
