"""
dark_lock_audit.py
Find auto-labelled boxes (source='auto') whose interior is dark, so a human can
re-check them first. Writes a CSV (darkest first) and a contact sheet.

Run from the backend/ directory (so `app` imports and ./data paths resolve):
    cd backend
    python ../scripts/dark_lock_audit.py --project <PROJECT_ID> \
        --state closed --out ../audit_out
Omit --state to include every auto-labelled box (list states with --list-states).
"""
import argparse, csv, json, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path.cwd()))
from app.config import settings                                    # noqa: E402
from app.connectors.statedb_connector import StateDBConnector      # noqa: E402
from app.tasks.auto_annotate import _resolve_image_path            # noqa: E402


def fetch(project_id):
    db = StateDBConnector()
    with db.get_session() as conn:
        return db.execute_query(
        conn,
        "SELECT i.id AS image_id, i.filename, i.filepath, a.id AS ann_id, "
        "a.class_name, a.state, a.bbox, a.points "
        "FROM annotations a JOIN images i ON i.id = a.image_id "
        "WHERE i.project_id = :p AND a.source = 'auto'",
        {"p": project_id},
        )


def metrics(img, bbox):
    h, w = img.shape[:2]
    xc, yc, bw, bh = bbox
    x1, y1 = max(int((xc - bw / 2) * w), 0), max(int((yc - bh / 2) * h), 0)
    x2, y2 = min(int((xc + bw / 2) * w), w), min(int((yc + bh / 2) * h), h)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    inside = v[y1:y2, x1:x2]
    mask = np.ones_like(v, bool); mask[y1:y2, x1:x2] = False
    outside = v[mask] if mask.any() else v.ravel()
    mi, mo = float(inside.mean()), float(outside.mean())
    return {
        "inside_mean": round(mi, 1),
        "outside_mean": round(mo, 1),
        "ratio": round(mi / max(mo, 1.0), 3),          # <1 => box darker than surroundings
        "dark_pct": round(float((inside < 60).mean() * 100), 1),
        "box": (x1, y1, x2, y2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--state", help="only boxes with this state/class, e.g. closed")
    ap.add_argument("--list-states", action="store_true")
    ap.add_argument("--out", default="audit_out")
    ap.add_argument("--sheet-size", type=int, default=30)
    ap.add_argument("--z", type=float, default=3.5, help="robust z-score above which a box is an outlier")
    ap.add_argument("--top", type=int, default=600, help="crops to put in contact sheets")
    args = ap.parse_args()

    rows = fetch(args.project)
    if args.list_states:
        from collections import Counter
        print(Counter((r["state"] or r["class_name"]) for r in rows)); return
    if args.state:
        rows = [r for r in rows if (r["state"] or r["class_name"]) == args.state]
    print(f"{len(rows)} auto-labelled boxes to analyse")

    out = Path(args.out); (out / "sheets").mkdir(parents=True, exist_ok=True)
    res = []
    for r in rows:
        p = _resolve_image_path(r["filepath"])
        bbox = r["bbox"]; bbox = json.loads(bbox) if isinstance(bbox, str) else bbox
        if not p or not bbox or len(bbox) != 4:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        m = metrics(img, bbox)
        if m:
            res.append({**{k: r[k] for k in ("image_id", "filename", "ann_id")},
                        "state": r["state"] or r["class_name"], "path": str(p), **m})

    # darkest first: low ratio, then low inside brightness
    res.sort(key=lambda d: (d["ratio"], d["inside_mean"]))
    # Outlier flag: compare every box with the NORMAL range of the same set.
    # Robust z-score (median/MAD) so the outliers themselves don't distort it.
    # Darkness score = 255 - inside_mean (higher = darker).
    dk = np.array([255 - d["inside_mean"] for d in res]) if res else np.array([])
    if len(dk):
        med = np.median(dk); mad = np.median(np.abs(dk - med)) * 1.4826 or 1.0
        q1, q3 = np.percentile(dk, [25, 75]); fence = q3 + 1.5 * (q3 - q1)
        for d, v in zip(res, dk):
            d["dark_score"] = round(float(v), 1)
            d["z"] = round(float((v - med) / mad), 2)
            d["outlier"] = int(v > fence or (v - med) / mad > args.z)
        print(f"normal darkness: median={med:.1f}  upper fence={fence:.1f}  "
              f"outliers={sum(d['outlier'] for d in res)} of {len(res)}")
        # outliers first, then by how extreme
        res.sort(key=lambda d: (-d["outlier"], -d["z"]))
        with open(out / "outliers.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["filename", "state", "dark_score", "z", "ratio", "image_id", "ann_id"], extrasaction="ignore")
            wr.writeheader(); wr.writerows(d for d in res if d["outlier"])
    with open(out / "dark_audit.csv", "w", newline="") as f:
        cols = ["rank", "outlier", "dark_score", "z", "filename", "state", "inside_mean", "outside_mean", "ratio", "dark_pct", "image_id", "ann_id"]
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); wr.writeheader()
        for i, d in enumerate(res, 1):
            wr.writerow({"rank": i, **d})

    # contact sheets (5 columns), brightened copy shown so dark locks are visible
    T, cols_n = 240, 5
    for s in range(0, min(args.top, len(res)), args.sheet_size):
        chunk = res[s:s + args.sheet_size]
        tiles = []
        for j, d in enumerate(chunk, s + 1):
            img = cv2.imread(d["path"]); x1, y1, x2, y2 = d["box"]
            pad = int(0.15 * max(x2 - x1, y2 - y1))
            c = img[max(y1 - pad, 0):y2 + pad, max(x1 - pad, 0):x2 + pad]
            c = cv2.cvtColor(c, cv2.COLOR_BGR2LAB)
            c[:, :, 0] = cv2.createCLAHE(3.0, (4, 4)).apply(c[:, :, 0])
            c = cv2.cvtColor(c, cv2.COLOR_LAB2BGR)
            c = cv2.resize(c, (T, T))
            cv2.putText(c, f"#{j} dark={d.get('dark_score')} z={d.get('z')}", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(c, d["filename"][-28:], (4, T - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            tiles.append(c)
        while len(tiles) % cols_n:
            tiles.append(np.zeros((T, T, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[k:k + cols_n]) for k in range(0, len(tiles), cols_n)])
        cv2.imwrite(str(out / "sheets" / f"sheet_{s // args.sheet_size + 1:03d}.jpg"), grid)
    print(f"Done. {len(res)} scored. Open {out}/sheets/sheet_001.jpg first; full list: {out}/dark_audit.csv")


if __name__ == "__main__":
    main()
