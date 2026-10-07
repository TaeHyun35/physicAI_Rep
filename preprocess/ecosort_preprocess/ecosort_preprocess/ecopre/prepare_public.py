"""
공개 데이터를 에코소트 환경에 맞추는 단계 (전처리 상세 설계 9-2).

  1) 물체 오려내기   : TrashNet·폴더형(AI Hub) = GrabCut,  TACO = 제공된 분할 마스크
  2) 자체 배경에 합성: 빈 라이트박스 배경 위에 무작위 크기·위치·회전 + 그림자
  3) 같은 전처리 적용: EcoPreprocessor.process(frame, 256)  (학습-추론 동일: 원칙 P5)
  4) 목록 기록      : data/processed/index_public.csv
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .config import (BACKGROUND_DIR, CUTOUT_DIR, DATA_DIR, DROP_ZONE, OBJ_SIZE_RANGE, PROCESSED_DIR,
                     TACO_MAP, TRAIN_SIZE)
from .preprocess import EcoPreprocessor
from .compose import composite, reinhard_match, rotate_bgra, scale_to_long_side

INDEX_FIELDS = ["path", "label", "item", "object_id", "source", "set", "conditions", "flags"]


# --------------------------------------------------------------------------- 1) 오려내기
def grabcut_cutout(img: np.ndarray, inset: float = 0.04, work_size: int = 256) -> np.ndarray | None:
    """
    단순 배경 사진에서 물체만 오려내 BGRA로 반환. 실패하면 None.
      - 가장자리 inset 비율은 '확실한 배경'으로 두고 GrabCut이 나머지를 물체/배경으로 나눈다.
      - GrabCut은 느리므로 긴 변 work_size로 줄여서 마스크를 구한 뒤 원본 크기로 키운다 (약 5~10배 빠름).
    """
    H, W = img.shape[:2]
    s = min(1.0, work_size / max(H, W))
    small = cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
    h, w = small.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * inset), int(h * inset), int(w * (1 - 2 * inset)), int(h * (1 - 2 * inset)))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, mask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # 가장 큰 덩어리만 남긴다
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fg = np.where(lab == big, 255, 0).astype(np.uint8)
    ratio = fg.mean() / 255
    if not 0.05 < ratio < 0.90:   # 너무 작거나 배경까지 잡힌 경우 버림
        return None
    # 마스크를 원본 크기로 키워 원본 해상도로 오려낸다 (질감 보존)
    fg = cv2.resize(fg, (W, H), interpolation=cv2.INTER_LINEAR)
    fg = np.where(fg > 127, 255, 0).astype(np.uint8)
    x, y, bw, bh = cv2.boundingRect(fg)
    return np.dstack([img, fg])[y:y + bh, x:x + bw]


def _cut_one(args):
    """병렬 처리용: (원본 경로, 저장 경로) → 성공 여부"""
    src, dst = args
    cut = grabcut_cutout(cv2.imread(str(src)))
    if cut is None:
        return False
    cv2.imwrite(str(dst), cut)
    return True


def cutouts_from_class_folders(src: Path, mapping: dict, source: str, limit_per_class: int | None = None,
                               workers: int = 0):
    """
    클래스별 폴더 구조(TrashNet, AI Hub) → data/cutouts/<source>/<label>/<원본폴더>_<파일명>.png
    workers > 0 이면 여러 CPU 코어로 병렬 처리한다 (0 = 자동: 코어 수).
    """
    from multiprocessing import Pool, cpu_count
    jobs, per_dir = [], {}
    for cls_dir in sorted(p for p in Path(src).iterdir() if p.is_dir()):
        label = mapping.get(cls_dir.name)
        if label is None:
            print(f"  [skip] 매핑 없음: {cls_dir.name}")
            continue
        out = CUTOUT_DIR / source / label
        out.mkdir(parents=True, exist_ok=True)
        files = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
        files = files[:limit_per_class] if limit_per_class else files
        per_dir[cls_dir.name] = (label, len(jobs), len(jobs) + len(files))
        jobs += [(f, out / f"{cls_dir.name}_{f.stem}.png") for f in files]

    n_proc = workers or cpu_count()
    if n_proc > 1:
        with Pool(n_proc) as pool:
            results = pool.map(_cut_one, jobs, chunksize=4)
    else:
        results = [_cut_one(j) for j in jobs]

    stats = {}
    for name, (label, a, b) in per_dir.items():
        stats[name] = f"{sum(results[a:b])}/{b - a} → {label}"
        print(f"  {name}: {stats[name]}")
    return stats


def cutouts_from_taco(root: Path, limit_per_class: int | None = None, min_area_px: int = 2500):
    """TACO 분할 마스크로 물체를 오려낸다. 다운로드한 이미지 크기에 맞춰 좌표를 보정한다."""
    ann = json.loads((root / "annotations.json").read_text())
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    images = {im["id"]: im for im in ann["images"]}
    counts, cache = {}, {}
    for a in ann["annotations"]:
        label = TACO_MAP.get(cats[a["category_id"]], "etc")
        if limit_per_class and counts.get(label, 0) >= limit_per_class:
            continue
        im = images[a["image_id"]]
        path = root / "images" / im["file_name"]
        if not path.exists() or not a.get("segmentation"):
            continue
        if path not in cache:
            cache.clear()   # 메모리 절약: 직전 이미지 하나만 보관
            cache[path] = cv2.imread(str(path))
        img = cache[path]
        if img is None:
            continue
        s = img.shape[1] / im["width"]   # 640px 버전을 받았으면 좌표를 줄인다
        mask = np.zeros(img.shape[:2], np.uint8)
        for poly in a["segmentation"]:
            pts = (np.array(poly).reshape(-1, 2) * s).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
        if mask.sum() / 255 < min_area_px * s * s:
            continue
        x, y, bw, bh = cv2.boundingRect(mask)
        cut = np.dstack([img, mask])[y:y + bh, x:x + bw]
        out = CUTOUT_DIR / "taco" / label
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / f"taco_{a['id']}.png"), cut)
        counts[label] = counts.get(label, 0) + 1
    print(f"  TACO 오려내기: {counts}")
    return counts


# --------------------------------------------------------------------------- 2~4) 합성·전처리
def own_color_stats(pre: EcoPreprocessor) -> dict:
    """자체 촬영 학습 이미지에서 클래스별 물체 픽셀의 Lab 평균·표준편차 (색 맞춤 목표값)."""
    stats = {}
    root = PROCESSED_DIR / "own" / "train_pool"
    for cls_dir in root.glob("*"):
        pix = []
        for f in list(cls_dir.glob("*.jpg"))[:200]:
            img = cv2.imread(str(f))
            obj = np.abs(img.astype(np.int16) - np.array(pre.pad_color)).max(axis=2) > 25
            if obj.sum() > 100:
                pix.append(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[obj])
        if pix:
            allp = np.concatenate(pix).astype(np.float32)
            stats[cls_dir.name] = (allp.mean(axis=0), allp.std(axis=0))
    return stats


def build_public_processed(pre: EcoPreprocessor, per_cutout: int = 1, color_match: bool = False, seed: int = 0):
    """오려낸 물체를 배경에 합성 → 전처리 → data/processed/public/<label>/ 저장 + 목록 작성."""
    rng = np.random.default_rng(seed)
    bgs = [cv2.imread(str(p)) for p in sorted(BACKGROUND_DIR.glob("*.jpg"))]
    H, W = bgs[0].shape[:2]
    stats = own_color_stats(pre) if color_match else {}
    rows, failed = [], 0
    for src_dir in sorted(p for p in CUTOUT_DIR.iterdir() if p.is_dir()):
        source = src_dir.name
        for label_dir in sorted(p for p in src_dir.iterdir() if p.is_dir()):
            label = label_dir.name
            out_dir = PROCESSED_DIR / "public" / label
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in sorted(label_dir.glob("*.png")):
                cut = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                if cut is None or cut.ndim != 3 or cut.shape[2] != 4:
                    continue
                if label in stats:
                    cut = reinhard_match(cut, *stats[label])
                for k in range(per_cutout):
                    p = scale_to_long_side(cut, int(H * rng.uniform(*OBJ_SIZE_RANGE)))
                    p = rotate_bgra(p, rng.uniform(0, 360))
                    cx = int(W / 2 + rng.uniform(-DROP_ZONE, DROP_ZONE) * W)
                    cy = int(H / 2 + rng.uniform(-DROP_ZONE, DROP_ZONE) * H)
                    frame = composite(bgs[rng.integers(len(bgs))], p, cx, cy)
                    res = pre.process(frame, out_size=TRAIN_SIZE)
                    if not res.ok:
                        failed += 1
                        continue
                    name = f"{source}_{f.stem}_{k}.jpg"
                    cv2.imwrite(str(out_dir / name), res.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    rows.append({
                        "path": str((out_dir / name).relative_to(DATA_DIR)), "label": label,
                        "item": source, "object_id": f"{source}-{f.stem}",   # 같은 원본 = 같은 물체
                        "source": "public", "set": "pretrain", "conditions": "synthetic",
                        "flags": "|".join(res.flags),
                    })
    write_index(PROCESSED_DIR / "index_public.csv", rows)
    print(f"[public] 합성·전처리 {len(rows)}장 저장, 실패 {failed}장")
    return rows


def write_index(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(rows)
