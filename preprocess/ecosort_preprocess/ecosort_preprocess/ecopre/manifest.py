"""
학습 목록(manifest.csv) 생성과 데이터 분할.

분할 원칙 (데이터 계획 2-3):
  - 같은 물체(object_id)의 사진은 반드시 같은 쪽(train 또는 val)에 들어간다 → 정확도 부풀림 방지
  - 클래스별로 따로 나눠 클래스 비율을 유지한다 (층화 분할)
  - test / confuse 폴더는 분할하지 않고 그대로 고정 평가용으로 쓴다

stage 열:  pretrain = 공개 데이터 (사전학습),  finetune = 자체·운영 데이터 (미세조정)
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict

import numpy as np

from .config import MANIFEST_PATH, PROCESSED_DIR

FIELDS = ["path", "label", "object_id", "source", "stage", "split", "conditions", "item", "flags"]


def _read(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _group_split(rows, val_ratio, rng):
    """클래스별로 물체 ID를 섞어 val_ratio 만큼을 검증용으로 지정."""
    by_label = defaultdict(set)
    for r in rows:
        by_label[r["label"]].add(r["object_id"])
    val_ids = set()
    for label, ids in by_label.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * val_ratio))) if len(ids) > 1 else 0
        val_ids.update(ids[:n_val])
    for r in rows:
        r["split"] = "val" if r["object_id"] in val_ids else "train"


def build_manifest(val_ratio_own: float = 0.2, val_ratio_public: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)
    public = _read(PROCESSED_DIR / "index_public.csv")
    own = _read(PROCESSED_DIR / "index_own.csv")

    for r in public:
        r["stage"] = "pretrain"
    _group_split(public, val_ratio_public, rng)

    pool = [r for r in own if r["set"] == "train_pool"]
    for r in pool:
        r["stage"] = "finetune"
    _group_split(pool, val_ratio_own, rng)
    for r in own:
        if r["set"] in ("test", "confuse"):
            r["stage"], r["split"] = "eval", r["set"]

    rows = public + own
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # 요약 출력: 단계/분할별 클래스 수
    summary = Counter((r["stage"], r["split"], r["label"]) for r in rows)
    print(f"[manifest] {MANIFEST_PATH} ({len(rows)}장)")
    for stage, split in sorted({(s, p) for s, p, _ in summary}):
        cnt = {lbl: n for (s, p, lbl), n in summary.items() if s == stage and p == split}
        print(f"  {stage:9s} {split:8s} {dict(sorted(cnt.items()))}")
    _check_leak(rows)
    return rows


def _check_leak(rows):
    """같은 물체가 학습과 검증/테스트에 동시에 있으면 경고."""
    where = defaultdict(set)
    for r in rows:
        where[r["object_id"]].add(r["split"])
    leaks = [oid for oid, s in where.items() if "train" in s and len(s) > 1]
    if leaks:
        print(f"  [경고] 학습과 평가에 동시에 들어간 물체 {len(leaks)}개: {leaks[:5]}")
    else:
        print("  물체 단위 분할 확인: 누수 없음")


def load_manifest():
    return _read(MANIFEST_PATH)
