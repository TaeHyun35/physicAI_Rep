"""
자체 촬영 데이터 전처리.

원본 프레임 파일명 규칙:  클래스_품목_물체ID_조건_번호.jpg
  예) plastic_petclear_P0042_L1-C0-S1-F_0001.jpg
  - 클래스: plastic / can / paper / etc
  - 물체ID: 같은 물건을 여러 번 찍어도 같은 ID → 학습/검증 분할을 물체 단위로 하기 위함
  - 조건  : 조명(L1/L0)-청결(C0/C1)-형태(S0/S1)-시점(F/S)

폴더(= 용도): data/raw/own/train_pool (학습·검증), test (고정 테스트셋), confuse (헷갈리는 품목)
결과       : data/processed/own/<폴더>/<클래스>/*.jpg  +  data/processed/index_own.csv
"""
from pathlib import Path

import cv2
import numpy as np

from .config import CLASSES, DATA_DIR, OWN_RAW_DIR, OWN_SETS, PROCESSED_DIR, TRAIN_SIZE
from .preprocess import EcoPreprocessor, save_pre_config
from .prepare_public import write_index


def parse_name(path: Path) -> dict | None:
    """파일명을 규칙대로 나눈다. 규칙에 맞지 않으면 None."""
    parts = path.stem.split("_")
    if len(parts) < 5 or parts[0] not in CLASSES:
        return None
    return {"label": parts[0], "item": parts[1], "object_id": parts[2], "conditions": parts[3]}


def build_own_processed(pre: EcoPreprocessor, source: str = "own"):
    """data/raw/own의 모든 프레임에 장치와 같은 전처리를 적용한다."""
    rows, skipped, sharps = [], [], []
    for set_name in OWN_SETS:
        src_dir = OWN_RAW_DIR / set_name
        if not src_dir.exists():
            continue
        for f in sorted(src_dir.glob("*.jpg")):
            meta = parse_name(f)
            if meta is None:
                skipped.append((f.name, "파일명 규칙 불일치"))
                continue
            res = pre.process(cv2.imread(str(f)), out_size=TRAIN_SIZE)
            if not res.ok:
                skipped.append((f.name, res.error))   # 물체를 못 찾은 프레임은 원인 확인 필요
                continue
            sharps.append((set_name, res.sharpness))
            out_dir = PROCESSED_DIR / "own" / set_name / meta["label"]
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f.name), res.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            # 운영 중 수집(field) 데이터는 품목 칸에 'field'가 들어간다 (active_learning.py)
            src = "field" if meta["item"] == "field" else source
            rows.append({**meta, "path": str((out_dir / f.name).relative_to(DATA_DIR)),
                         "source": src, "set": set_name, "flags": [x for x in res.flags if x != "blurry"]})

    # 학습용(train_pool) 선명도로 흐림 기준을 다시 정한 뒤, 그 기준으로 blurry 표시를 붙인다
    thresh = calibrate_blur_threshold(pre, [v for s, v in sharps if s == "train_pool"])
    for r, (_, v) in zip(rows, sharps):
        r["flags"] = "|".join(r["flags"] + (["blurry"] if v < thresh else []))
    write_index(PROCESSED_DIR / "index_own.csv", rows)
    print(f"[own] 전처리 {len(rows)}장 저장, 건너뜀 {len(skipped)}장")
    for name, why in skipped[:10]:
        print(f"  - {name}: {why}")
    return rows


def calibrate_blur_threshold(pre: EcoPreprocessor, sharps: list, pct: float = 5, factor: float = 0.8):
    """
    흐림 기준값 재설정 (전처리 상세 설계 S11).
    정상 촬영한 학습 이미지의 선명도 하위 5% 값에 0.8을 곱한 값을 기준으로 삼아
    data/pre_config.json에 저장한다. 장치도 이 파일을 읽으므로 학습과 같은 기준을 쓴다.
    """
    if len(sharps) < 20:
        print("  [흐림 기준] 표본이 20장 미만이라 기존 값 유지")
        return pre.cfg.blur_thresh
    new = float(np.percentile(sharps, pct) * factor)
    print(f"  [흐림 기준] {pre.cfg.blur_thresh:.1f} → {new:.1f} (선명도 하위 {pct}% x {factor})")
    pre.cfg.blur_thresh = new
    save_pre_config(pre.cfg)
    return new
