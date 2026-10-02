import sys
from ultralytics import YOLO
m = YOLO(sys.argv[1])
r = m.val(data="datasets/weapons_merged/data.yaml", split="val", imgsz=960,
          batch=16, device="0", verbose=False, plots=False)
names = r.names if hasattr(r, "names") else {}
lines = ["=== per-class (val split) ==="]
for i, c in enumerate(r.box.ap_class_index):
    lines.append("  %-7s P=%.3f R=%.3f mAP50=%.3f mAP50-95=%.3f"
                 % (names.get(int(c), c), r.box.p[i], r.box.r[i], r.box.ap50[i], r.box.ap[i]))
lines.append("  %-7s mAP50=%.3f mAP50-95=%.3f" % ("ALL", r.box.map50, r.box.map))
open("perclass_result.txt", "w").write("\n".join(lines) + "\n")
