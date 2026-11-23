"""GP (BayesOpt, qLogNEI + trust region + escape feasibility, as v4) under the per-star statistic, warm-started from the
74 re-solved regrid models ($W/gprun from gp_setup.sh). Two batches of 12 new models (n_max_iter 2, n_max_mods 98).

    cd $W/gprun && MKL_NUM_THREADS=16 PYTHONPATH=$W/dynamite-6aa90a7 python gp_run.py config.yaml

Driver = PM_grid/sBH_GP_grid_v4/run_sBH_GP_grid.py without its profile-round / profile-fill / final-plot branches, and
with ModelIterator(plots=False): v4's per-iteration plots peaked the parent at ~450 GB. The fork-safety prelude is kept
verbatim in substance (single-threaded parent, CPU affinity restored after torch, per-child MKL thread count).
"""
import sys

sys.path.insert(0, "/nexus/posix0/MIA-astro-env/nneum/pesmith")
import grid_env  # noqa: F401,E402  (re-exec under ENV_mkl; OMP/OpenBLAS 1, MKL default 8)

import ctypes  # noqa: E402
import faulthandler  # noqa: E402
import os  # noqa: E402

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import dynamite as dyn  # noqa: E402

faulthandler.enable()


def _restore_affinity():
    # importing torch narrows this process's affinity to one core; forked integrators would inherit it
    try:
        if len(os.sched_getaffinity(0)) == 1 and (os.cpu_count() or 1) > 1:
            os.sched_setaffinity(0, set(range(os.cpu_count())))
    except (AttributeError, OSError):
        pass


try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
finally:
    _restore_affinity()

from threadpoolctl import threadpool_info  # noqa: E402

for _i in threadpool_info():
    if _i["internal_api"] == "mkl":
        _mkl = ctypes.CDLL(_i["filepath"])
        # explicit per-child count: BoTorch's threadpool_limits(1) otherwise leaves solves single-threaded (v4 it2-3)
        os.register_at_fork(after_in_child=lambda: _mkl.MKL_Set_Num_Threads_Local(
            int(os.environ.get("MKL_NUM_THREADS", "8"))))

cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
c = dyn.config_reader.Configuration(cfg, reset_logging=True, user_logfile=False, reset_existing_output=False)
t = c.all_models.table
assert "/veldist-fitstat/gp" in c.settings.io_settings["output_directory"], c.settings.io_settings  # gprun or a test copy
stat = {k.name: getattr(k, "statistic", "diag") for k in c.system.get_unique_triaxial_visible_component().kinematic_data}
assert all(stat[n] == "laplace" for n in ("veldist", "hst_pm", "gaia_pm")), stat
mp, stop = c.settings.multiprocessing_settings, c.settings.parameter_space_settings["stopping_criteria"]
print(f"dynamite {dyn.__version__} at {dyn.__path__}; statistic {stat}")
print(f"warm start: {len(t)} models, {int(np.sum(t['all_done']))} done, best kinchi2 {np.min(t['kinchi2']):.1f} "
      f"({t['directory'][int(np.argmin(t['kinchi2']))]})")
print(f"ncpus={mp['ncpus']} total_cores={mp['total_cores']} orblib_chunks={mp['orblib_chunks']} "
      f"ncpus_weights={mp['ncpus_weights']} MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}; "
      f"n_max_mods={stop['n_max_mods']} n_max_iter={stop['n_max_iter']}", flush=True)
dyn.model_iterator.ModelIterator(config=c, plots=False)
t = c.all_models.table
print(f"GP RUN DONE: {len(t)} models, {int(np.sum(t['all_done']))} done, best kinchi2 {np.min(t['kinchi2']):.1f} "
      f"({t['directory'][int(np.argmin(t['kinchi2']))]})", flush=True)
