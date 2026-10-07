"""
자체 촬영 데이터 전처리와 점검.

원본 프레임 파일명 규칙 (현장조사 체크리스트 '샘플 이미지 수집 지침 4'):
    클래스_품목_물체ID_조건_번호.jpg
  예) plastic_petclear_P0042_L1-C0-S1-F_0001.jpg
  - 클래스: plastic / can / paper / etc  (헷갈리는 품목도 '정답 클래스'를 쓴다)
  - 품목  : 밑줄(_) 없이
  - 물체ID: 같은 물건을 여러 번 찍어도 같은 ID → 학습/검증 분할을 물체 단위로 하기 위함
  - 조건  : 조명(L1/L0)-청결(C0/C1)-형태(S0/S1)-시점(F/S)
  - 확장자: .jpg/.jpeg/.png, 대소문자 무관

폴더(= 용도): data/raw/own/train_pool (학습·검증), test (고정 테스트셋), confuse (헷갈리는 품목)
결과       : data/processed/own/<폴더>/<클래스>/*.jpg  +  data/processed/index_own.csv
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .config import CLASSES, DATA_DIR, OWN_RAW_DIR, OWN_SETS, PROCESSED_DIR, TRAIN_SIZE, list_images
from .preprocess import EcoPreprocessor, save_pre_config
from .prepare_public import write_index

COND_RE = re.compile(r"^L[01]-C[01]-S[01]-[FS]$")


def name_problem(path: Path) -> str | None:
    """파일명 규칙 위반 이유. 문제가 없으면 None."""
    parts = Path(path).stem.split("_")
    if len(parts) != 5:
        return f"밑줄로 나눈 칸이 {len(parts)}개 (5개여야 함: 클래스_품목_물체ID_조건_번호)"
    label, _, object_id, cond, _ = parts
    if label not in CLASSES:
        return f"클래스 '{label}'가 {CLASSES} 중 하나가 아님 (confuse 폴더도 정답 클래스를 씀)"
    if not object_id:
        return "물체ID가 비어 있음"
    if not COND_RE.match(cond):
        return f"조건 '{cond}'가 L?-C?-S?-F/S 형식이 아님"
    return None


def parse_name(path: Path) -> dict | None:
    """파일명을 규칙대로 나눈다. 규칙에 맞지 않으면 None (이유는 name_problem으로 확인)."""
    if name_problem(path):
        return None
    return dict(zip(("label", "item", "object_id", "conditions"), Path(path).stem.split("_")[:4]))


def build_own_processed(pre: EcoPreprocessor, source: str = "own"):
    """data/raw/own의 모든 프레임에 장치와 같은 전처리를 적용한다."""
    rows, skipped, sharps = [], [], []
    for set_name in OWN_SETS:
        for f in list_images(OWN_RAW_DIR / set_name):
            meta = parse_name(f)
            if meta is None:
                skipped.append((f.name, "파일명 규칙 불일치"))
                continue
            frame = cv2.imread(str(f))
            if frame is None:
                skipped.append((f.name, "이미지를 읽을 수 없음"))
                continue
            res = pre.process(frame, out_size=TRAIN_SIZE)
            if not res.ok:
                # L0(조명 끔) 프레임은 light_error로 빠지는 것이 정상 동작이다
                skipped.append((f.name, res.error))
                continue
            sharps.append((set_name, res.sharpness))
            out_dir = PROCESSED_DIR / "own" / set_name / meta["label"]
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{f.stem}.jpg"
            cv2.imwrite(str(out_path), res.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            # 운영 중 수집한 데이터는 품목 칸에 'field'를 넣어 구분한다
            src = "field" if meta["item"] == "field" else source
            rows.append({**meta, "path": out_path.relative_to(DATA_DIR).as_posix(),
                         "source": src, "set": set_name, "flags": [x for x in res.flags if x != "blurry"]})

    if not rows and not skipped:
        print(f"[own] 자체 촬영 이미지가 없습니다: {OWN_RAW_DIR} 아래 {OWN_SETS}")
        return rows

    # 학습용(train_pool) 선명도로 흐림 기준을 다시 정한 뒤, 그 기준으로 blurry 표시를 붙인다
    thresh = calibrate_blur_threshold(pre, [v for s, v in sharps if s == "train_pool"])
    for r, (_, v) in zip(rows, sharps):
        r["flags"] = "|".join(r["flags"] + (["blurry"] if v < thresh else []))
    write_index(PROCESSED_DIR / "index_own.csv", rows)
    print(f"[own] 전처리 {len(rows)}장 저장, 건너뜀 {len(skipped)}장")
    if skipped:
        print("  건너뛴 이유: " + ", ".join(f"{k} {v}" for k, v in Counter(w for _, w in skipped).most_common()))
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


def audit_own() -> dict:
    """
    촬영 데이터 점검 (배경 사진 없이도 실행 가능, 전처리 전에 먼저 돌린다).
      - 파일명 규칙 위반 목록
      - 용도·클래스별 '물건 수 / 사진 수'  (체크리스트 G-03, 0-5: 사진보다 물건 수가 병목)
      - train_pool과 test/confuse에 같은 물체ID가 있는지 (평가 누수)
      - 촬영 조건 분포 (L0 비중, 오염·구김 비중)
      - 센서 값 .txt 누락 수
    """
    bad, no_txt = [], 0
    objs = defaultdict(set)          # (용도, 클래스) → {물체ID}
    imgs = Counter()                 # (용도, 클래스) → 사진 수
    where = defaultdict(set)         # 물체ID → {용도}
    conds = Counter()
    for set_name in OWN_SETS:
        for f in list_images(OWN_RAW_DIR / set_name):
            why = name_problem(f)
            if why:
                bad.append((f"{set_name}/{f.name}", why))
                continue
            label, _, oid, cond, _ = f.stem.split("_")
            objs[(set_name, label)].add(oid)
            imgs[(set_name, label)] += 1
            where[oid].add(set_name)
            conds.update(cond.split("-"))
            no_txt += not f.with_suffix(".txt").exists()

    total = sum(imgs.values())
    print(f"\n[check] {OWN_RAW_DIR}  (규칙에 맞는 사진 {total}장, 위반 {len(bad)}장)")
    if not total and not bad:
        print("  사진이 없습니다. data/raw/own/{train_pool,test,confuse}/ 에 넣어 주세요.")
        return {}
    print(f"  {'용도':10s} " + " ".join(f"{c:>13s}" for c in CLASSES) + "   (물건 수 / 사진 수)")
    for s in OWN_SETS:
        cells = [f"{len(objs[(s, c)])} / {imgs[(s, c)]}" for c in CLASSES]
        print(f"  {s:10s} " + " ".join(f"{x:>13s}" for x in cells))

    leaks = sorted(o for o, s in where.items() if "train_pool" in s and len(s) > 1)
    print("  평가 누수: " + (f"[경고] train_pool과 평가 폴더에 같은 물체 {len(leaks)}개: {leaks[:5]}"
                          if leaks else "없음"))
    if total:
        print(f"  조건 분포: 조명끔 L0 {conds['L0'] / total:.0%}, 오염 C1 {conds['C1'] / total:.0%}, "
              f"구김 S1 {conds['S1'] / total:.0%}, 측면 S {conds['S'] / total:.0%}")
        print(f"  센서 값 .txt 없음: {no_txt}장")
    for name, why in bad[:15]:
        print(f"  - {name}: {why}")
    return {"bad": bad, "leaks": leaks,
            "objects": {k: len(v) for k, v in objs.items()}, "images": dict(imgs)}
