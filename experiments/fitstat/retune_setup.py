"""Private copy with ONE new model: a gprun model's parameters with some changed (re-tuning the nuisance at fixed
M_IMBH, docs/imbh-status-2026-11-26.md). Lives under $W/noise/ so noise_build.py and regrid_one.py accept it.

    python retune_setup.py W SRC_MLDIR NAME NEW_MLDIR KEY=VALUE [KEY=VALUE ...]
e.g. retune_setup.py $W orblib_007_008/ml02.93 m32_ml288 orblib_100_000/ml02.88 ml=2.88

Makes $W/noise/NAME/: config.yaml = gprun's with output_directory -> NGC5139_output/, NGC5139_output/all_models.ecsv
= gprun's plus the new row (SRC's row, KEYs replaced, directory NEW_MLDIR, not done). Then, as for a noise replica:
noise_build.py config.yaml NEW_MLDIR; regrid_one.py config.yaml NEW_MLDIR result.npz.
"""
import re
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

W, src, name, new = sys.argv[1:5]
D = Path(W) / "noise" / name
assert not D.exists(), D
t = Table.read(Path(W) / "gprun/NGC5139_output/all_models.ecsv")
dirs = [str(d).strip("/") for d in t["directory"]]
assert src.strip("/") in dirs and new.strip("/") not in dirs
assert new.split("/")[0] not in {d.split("/")[0] for d in dirs}, "orblib directory exists in gprun"
row = {k: t[k][dirs.index(src.strip("/"))] for k in t.colnames}
for kv in sys.argv[5:]:
    k, v = kv.split("=")
    assert k in t.colnames and np.issubdtype(t[k].dtype, np.floating), k
    row[k] = float(v)
assert new.rstrip("/").split("/")[1] == f"ml{float(row['ml']):05.2f}", (new, row["ml"])
row["directory"] = new.rstrip("/") + "/"
for k in t.colnames:
    if k in ("all_done", "orblib_done", "weights_done"):
        row[k] = False
    elif k in ("kinchi2", "chi2", "kinmapchi2"):
        row[k] = np.nan
t.add_row([row[k] for k in t.colnames])
(D / "NGC5139_output/models").mkdir(parents=True)
cfg = (Path(W) / "gprun/config.yaml").read_text()
cfg, n = re.subn(r'output_directory: ".*"', f'output_directory: "{D}/NGC5139_output/"', cfg)
assert n == 1
(D / "config.yaml").write_text(cfg)
t.write(D / "NGC5139_output/all_models.ecsv")
print(f"ready: {D}; new row", {k: row[k] for k in ("m-bh", "mscale-sbh", "rscale-sbh", "q-stars", "ml", "directory")})
