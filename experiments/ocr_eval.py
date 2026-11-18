import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""Step 2 of the OCR evaluation: measure the plate reader on the hand-labelled crops.

Measures, for each OCR engine and each preprocessing variant:
  full-plate accuracy (Wilson 95% CI), CER, accuracy by one-/two-line plate and by plate height,
  syntax-valid rate, the most frequent character confusions, McNemar tests against the pipeline
  setting, and a risk-coverage curve of the OCR confidence (does a higher score mean a right read?).

  python ocr_eval.py --crops outputs/ocr_test --out outputs/reports/ocr_eval
  python ocr_eval.py --crops outputs/ocr_test --out outputs/reports/ocr_eval --engines fastplate easyocr paddle

Engines: fastplate = FastPlateOCR cct-s-v2-global-model (the pipeline's reader);
easyocr and paddle (PaddleOCR, the reader of Baseline A) are optional: pip install easyocr / paddleocr.
"""
import argparse, csv, json, math, re, time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# --- text utilities ---------------------------------------------------------------------------------
def norm(s):
    return "".join(ch for ch in str(s).upper() if ch.isalnum())


def align(ref, hyp):
    """Levenshtein distance and the substitution pairs (ref_char, hyp_char) of one optimal alignment."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), int); d[:, 0] = range(n + 1); d[0, :] = range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    subs, i, j = [], n, m
    while i and j:
        if d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                subs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + 1:
            subs.append((ref[i - 1], "∅")); i -= 1
        else:
            subs.append(("∅", hyp[j - 1])); j -= 1
    subs += [(ref[k], "∅") for k in range(i)] + [("∅", hyp[k]) for k in range(j)]
    return int(d[n, m]), subs


# Vietnamese plate after removing separators: 2-digit province code, series (letter + optional letter
# or digit), 4-5 digits. shortcut: the province code is not checked against the official list;
# import the team's own list from alpr_common for the full syntax rule.
PLATE_RE = re.compile(r"^\d{2}[A-Z][A-Z0-9]?\d{4,5}$")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


# --- preprocessing variants -------------------------------------------------------------------------
def order4(pts):
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])  # tl tr br bl


def rectify(crop):
    """Perspective rectification: Otsu mask of the plate body -> 4 corners -> warp to a rectangle."""
    g = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return crop
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.3 * g.size:          # plate body not found: keep the crop unchanged
        return crop
    poly = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    pts = poly.reshape(-1, 2).astype(np.float32) if len(poly) == 4 else cv2.boxPoints(cv2.minAreaRect(c))
    tl, tr, br, bl = src = order4(pts)
    W = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    H = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if W < 10 or H < 5:
        return crop
    M = cv2.getPerspectiveTransform(src, np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]))
    return cv2.warpPerspective(crop, M, (W, H))


def clahe(crop):
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def tight(crop):  # remove the 8% padding the export added on each side: 1/1.16 of the size
    h, w = crop.shape[:2]; dx, dy = int(w * 0.08 / 1.16), int(h * 0.08 / 1.16)
    return crop[dy:h - dy, dx:w - dx]


VARIANTS = {
    "tight": tight,                        # the bare ground-truth box
    "pad8": lambda c: c,                   # pipeline setting (8% padding)
    "pad8+rectify": rectify,
    "pad8+clahe": clahe,
    "pad8+rectify+clahe": lambda c: clahe(rectify(c)),
}


# --- OCR engines: each returns (text, confidence in [0,1] or nan) --------------------------------
def make_fastplate():
    from fast_plate_ocr import LicensePlateRecognizer
    m = LicensePlateRecognizer("cct-s-v2-global-model")

    def read(img):  # the model expects RGB; OpenCV crops are BGR
        p = m.run(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), return_confidence=True)[0]
        return norm(p.plate), float(np.mean(p.char_probs))  # mean over all 10 slots, pads included
    return read


def lines_to_text(items):
    """items: (y_center, x_left, text, conf). Group into lines by y, read top line first."""
    if not items:
        return "", float("nan")
    items.sort()
    ys = [i[0] for i in items]
    split = None
    if len(items) > 1:
        gaps = np.diff(ys)
        if gaps.max() > 0.25 * (max(ys) - min(ys) + 1e-6) and gaps.max() > 5:
            split = int(gaps.argmax()) + 1
    groups = [items] if split is None else [items[:split], items[split:]]
    text = "".join("".join(t for _, _, t, _ in sorted(g, key=lambda i: i[1])) for g in groups)
    return norm(text), float(np.mean([c for *_, c in items]))


def make_easyocr():
    import easyocr
    r = easyocr.Reader(["en"], verbose=False)
    allow = "0123456789ABCDEFGHKLMNPRSTUVXYZ-."

    def read(img):
        res = r.readtext(img, detail=1, allowlist=allow)
        return lines_to_text([(np.mean([p[1] for p in b]), min(p[0] for p in b), t, c) for b, t, c in res])
    return read


def make_paddle():
    from paddleocr import PaddleOCR   # PaddleOCR 3.x API
    o = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)

    def read(img):
        res = o.predict(img)[0]
        boxes = res["rec_polys"] if "rec_polys" in res else res["dt_polys"]
        return lines_to_text([(float(np.mean(b[:, 1])), float(b[:, 0].min()), t, float(c))
                              for b, t, c in zip(boxes, res["rec_texts"], res["rec_scores"])])
    return read


ENGINES = dict(fastplate=make_fastplate, easyocr=make_easyocr, paddle=make_paddle)


# --- evaluation -------------------------------------------------------------------------------------
def summarise(recs):
    n = len(recs)
    k = sum(r["ok"] for r in recs)
    lo, hi = wilson(k, n)
    chars = sum(len(r["gt"]) for r in recs)
    return dict(n=n, acc=k / n if n else float("nan"), acc_lo=lo, acc_hi=hi,
                cer=sum(r["dist"] for r in recs) / max(1, chars),
                syntax_valid=sum(bool(PLATE_RE.match(r["pred"])) for r in recs) / max(1, n))


def mcnemar(a, b):
    from scipy.stats import binomtest
    b01 = sum(x["ok"] and not y["ok"] for x, y in zip(a, b))
    b10 = sum(y["ok"] and not x["ok"] for x, y in zip(a, b))
    p = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    return dict(only_first=b01, only_second=b10, p=p)


def main():
    default_crops = "outputs/ocr_test" if Path("outputs/ocr_test").exists() else "ocr_test"
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default=default_crops, help="Path to cropped plates dataset directory")
    ap.add_argument("--out", default="outputs/reports/ocr_eval", help="Path to save evaluation reports")
    ap.add_argument("--engines", nargs="+", default=["fastplate"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    crops = Path(a.crops)
    labels_file = crops / "labels.csv"
    if not labels_file.exists():
        print(f"❌ Không tìm thấy '{labels_file}'. Hãy chạy 'python ocr_export_crops.py export' trước để tạo crops!")
        return
    rows = [r for r in csv.DictReader(open(crops / "labels.csv", encoding="utf-8"))
            if r["text"] and r["text"] != "ILLEGIBLE"]
    print(f"{len(rows)} labelled legible crops")
    imgs = {r["file"]: cv2.imread(str(crops / "crops" / r["file"])) for r in rows}

    results, per_crop = {}, []
    for e in a.engines:
        try:
            read = ENGINES[e]()
        except ImportError as err:  # optional readers: pip install easyocr / paddleocr
            print(f"skip {e}: {err}"); continue
        read(imgs[rows[0]["file"]])  # warm-up, so the first variant's time excludes model loading
        for v in a.variants:
            recs, t0 = [], time.perf_counter()
            for r in rows:
                pred, conf = read(VARIANTS[v](imgs[r["file"]]))
                gt = norm(r["text"]); dist, subs = align(gt, pred)
                recs.append(dict(file=r["file"], line=r["line"], h=int(r["h"]), gt=gt, pred=pred, conf=conf,
                                 ok=pred == gt, dist=dist, subs=subs))
            key = f"{e}/{v}"
            res = dict(ms_per_crop=(time.perf_counter() - t0) * 1000 / len(rows),  # includes preprocessing
                       all=summarise(recs),
                       one_line=summarise([x for x in recs if x["line"] == "one"]),
                       two_line=summarise([x for x in recs if x["line"] == "two"]),
                       h_lt25=summarise([x for x in recs if x["h"] < 25]),
                       h_25_50=summarise([x for x in recs if 25 <= x["h"] < 50]),
                       h_ge50=summarise([x for x in recs if x["h"] >= 50]),
                       acc_if_syntax_valid=summarise([x for x in recs if PLATE_RE.match(x["pred"])]),
                       top_confusions=Counter(s for x in recs for s in x["subs"]).most_common(15))
            results[key] = (res, recs)
            print(key, {k: round(v * 100, 2) if isinstance(v, float) else v for k, v in res["all"].items()})
            per_crop += [dict(engine=e, variant=v, **{k: x[k] for k in ("file", "line", "h", "gt", "pred", "conf", "ok", "dist")})
                         for x in recs]

    base = f"{a.engines[0]}/pad8"
    tests = {k: mcnemar(results[base][1], r[1]) for k, r in results.items() if k != base and base in results}
    json.dump(dict(summary={k: r[0] for k, r in results.items()}, mcnemar_vs=dict(base=base, tests=tests)),
              open(out / "ocr_eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    with open(out / "ocr_per_crop.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, per_crop[0].keys()); w.writeheader(); w.writerows(per_crop)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # risk-coverage of the confidence: keep the most confident reads first
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, (_, recs) in results.items():
        rr = sorted((x for x in recs if not math.isnan(x["conf"])), key=lambda x: -x["conf"])
        if not rr or not k.endswith("/pad8"):
            continue
        ok = np.cumsum([x["ok"] for x in rr]) / np.arange(1, len(rr) + 1)
        ax.plot(np.arange(1, len(rr) + 1) / len(rr) * 100, ok * 100, label=k)
    ax.set_xlabel("Coverage: share of plates accepted (%)"); ax.set_ylabel("Accuracy of accepted plates (%)")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(out / "risk_coverage.pdf"); fig.savefig(out / "risk_coverage.png", dpi=200)

    # accuracy per variant and line type
    keys = list(results)
    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 0.9), 4))
    xs = np.arange(len(keys))
    for i, part in enumerate(["all", "one_line", "two_line"]):
        ax.bar(xs + (i - 1) * 0.27, [results[k][0][part]["acc"] * 100 for k in keys], 0.27, label=part)
    ax.set_xticks(xs); ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Full-plate accuracy (%)"); ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out / "accuracy_by_variant.pdf"); fig.savefig(out / "accuracy_by_variant.png", dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    assert align("51A12345", "51A12345")[0] == 0
    assert align("59G307326", "59C307326") == (1, [("G", "C")])
    assert align("8B", "B")[0] == 1 and PLATE_RE.match("59G307326") and PLATE_RE.match("51A12345")
    assert not PLATE_RE.match("531111")
    main()
