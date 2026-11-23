"""Stage 3 of the GP run: hand-placed models along the M_IMBH profile (one profile-only iteration of 12).
The BayesOpt batches (stages 1-2) chase the minimum, so the profile's wings stay GP extrapolation. For each target
M_IMBH, take the other six parameters at the minimum of the GP mean over a shared Sobol sample (mean of three GP fits,
seeds 0-2, on kinchi2 = the per-star statistic), kinematics only: no fast-star constraint, since these map the curve.
Same method as v4's escape_restart/profile_fill.py without its RV curve.
Run in $W/gprun AFTER stage 2 has finished (its models are then in the GP); read-only, writes profile_fill.ecsv,
which `gp_run.py config.yaml --profile-fill profile_fill.ecsv` runs as one iteration.
    cd $W/gprun && PYTHONPATH=$W/dynamite-6aa90a7 python $W/veldist/experiments/fitstat/gp_profile_fill.py
"""
import copy
import logging
import sys

import numpy as np

# 12 = one iteration at ncpus 12; dense where the 74-model profile is thinnest (below 15k) and on the right wall
TARGETS = (3500, 4500, 5500, 7000, 8500, 10000, 12500, 15000, 32000, 42000, 55000, 75000)
SEEDS = (0, 1, 2)
N = 20000  # shortcut: best Sobol point, no local refinement (as v4)
OUT = "profile_fill.ecsv"


def main():
    logging.disable(logging.WARNING)
    import torch
    from astropy.table import Table

    import dynamite as dyn
    from dynamite import parameter_space as ps

    c = dyn.config_reader.Configuration("config.yaml", reset_logging=False, reset_existing_output=False)
    assert "/veldist-fitstat/gp" in c.settings.io_settings["output_directory"]
    t = c.all_models.table
    done = np.asarray(t["all_done"], bool)
    assert done.all(), f"{(~done).sum()} unfinished rows: run after stage 2"
    tab = t[np.isfinite(np.asarray(t["kinchi2"], float))]
    print(f"GP on {len(tab)} finished models")
    pss = copy.deepcopy(c.settings.parameter_space_settings)
    pss["generator_settings"].pop("escape_feasibility", None)
    gen = ps.BayesOptGenerator(c.parspace, parspace_settings=pss)
    gen.current_models = c.all_models
    free = [p for p in c.parspace if not p.fixed]
    torch.manual_seed(0)
    np.random.seed(0)
    base = gen._sobol_in_box(N)
    fits = []
    for s in SEEDS:
        torch.manual_seed(s)
        np.random.seed(s)
        fits.append(ps.fit_gp_from_table(tab, c.parspace, which_chi2="kinchi2"))
    _, _, _, names, lo, hi = fits[0]
    assert all(f[3] == names and np.allclose(f[4], lo) and np.allclose(f[5], hi) for f in fits)
    assert [p.name for p in free] == list(names), (names, [p.name for p in free])
    jb = names.index("m-bh")
    Xdone = np.asarray(fits[0][1])
    stars = c.system.get_unique_triaxial_visible_component()
    col = {n: (lo + base * (hi - lo))[:, j] for j, n in enumerate(names)}
    ok = np.array([not np.any(np.isnan(stars.triax_pqu2tpp(p, q, u)))
                   for p, q, u in zip(col["p-stars"], col["q-stars"], col["u-stars"])])
    base = base[ok]
    print(f"Sobol sample: {len(base)} of {len(ok)} have valid viewing angles")

    def at_mass(m):
        U = base.copy()
        U[:, jb] = (np.log10(m) - lo[jb]) / (hi[jb] - lo[jb])
        with torch.no_grad():
            posts = [f[0].posterior(torch.tensor(U)) for f in fits]
            mu = np.mean([-p.mean.numpy().ravel() for p in posts], 0)
            sd = np.mean([p.variance.clamp_min(0).sqrt().numpy().ravel() for p in posts], 0)
        return U, lo + U * (hi - lo), mu, sd

    rows = []
    for m in TARGETS:
        assert lo[jb] <= np.log10(m) <= hi[jb], (m, 10 ** lo[jb], 10 ** hi[jb])
        U, raw, mu, sd = at_mass(m)
        i = int(np.argmin(mu))
        phys = {p.name: float(10 ** raw[i, j] if p.logarithmic else raw[i, j]) for j, p in enumerate(free)}
        rows.append(dict(phys, pred_kinchi2=float(mu[i]), pred_sd=float(sd[i]),
                         nearest_done=float(np.min(np.linalg.norm(Xdone - U[i], axis=1)))))
    out = Table(rows=rows, names=list(rows[0]))
    for k in out.colnames:
        out[k].format = ".4g"
    out.pprint(max_width=-1, max_lines=-1)
    out.write(OUT, format="ascii.ecsv", overwrite=True)
    print("pred_kinchi2, pred_sd: GP mean and sd; nearest_done: distance to the closest finished model in the unit box")
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
