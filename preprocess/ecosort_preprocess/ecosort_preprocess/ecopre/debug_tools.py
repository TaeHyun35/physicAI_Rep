"""
전처리 점검 도구.

  visualize : 한 장의 프레임이 S1~S14를 거치며 어떻게 바뀌는지 한 장의 그림으로 저장
  benchmark : 폴더 전체를 처리해 단계별 시간(H-03), 실패 원인, 플래그 비율을 표로 출력
  sweep     : 파라미터 하나를 여러 값으로 바꿔 가며 결과 비교 (실험 E1, E2 등)
"""
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .augment import augment
from .config import DEBUG_DIR, MODEL_SIZE, TRAIN_SIZE
from .preprocess import EcoPreprocessor, center_crop


def _list_images(path: Path, limit: int | None = None) -> list:
    path = Path(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.suffix.lower() in (".jpg", ".png"))
    return files[:limit] if limit else files


def _tile(img, size=240, label=""):
    """패널 한 칸: 비율 유지 리사이즈 + 회색 여백 + 제목."""
    if img is None:
        img = np.full((size, size, 3), 60, np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    cell = np.full((size + 22, size, 3), 40, np.uint8)
    y, x = 22 + (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    cell[y:y + img.shape[0], x:x + img.shape[1]] = img
    cv2.putText(cell, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return cell


# --------------------------------------------------------------------------- 1) 단계별 시각화
def visualize(pre: EcoPreprocessor, image_path: Path, out_dir: Path = DEBUG_DIR, n_aug: int = 4, seed: int = 0):
    """원본 → 마스크 → 크롭 → 보정 → 256 결과 → 224 모델 입력 → 증강 예시를 한 장으로 저장."""
    frame = cv2.imread(str(image_path))
    dbg = {}
    res = pre.process(frame, out_size=TRAIN_SIZE, debug=dbg)
    im = dbg.get("images", {})

    view = frame.copy()
    if res.box:   # 원본 위에 크롭 영역 표시
        x0, y0, x1, y1 = res.box
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 0, 255), max(2, frame.shape[1] // 300))

    row1 = [_tile(view, label="0 frame + box"),
            _tile(im.get("color_mask"), label="S3-S6 color mask"),
            _tile(im.get("edge_mask"), label="S8 edge mask" if "edge_mask" in im else "S8 (skipped)"),
            _tile(im.get("crop_raw"), label="S10 crop (raw)")]
    row2 = [_tile(im.get("crop_corrected"), label="S12 color corrected"),
            _tile(res.image, label=f"S13 output {TRAIN_SIZE}"),
            _tile(center_crop(res.image, MODEL_SIZE) if res.ok else None, label=f"model input {MODEL_SIZE}"),
            _tile(None, label="")]
    rng = np.random.default_rng(seed)
    row3 = [_tile(augment(res.image, rng) if res.ok else None, label=f"train aug #{i + 1}") for i in range(n_aug)]

    panel = np.vstack([np.hstack(row1), np.hstack(row2), np.hstack(row3)])
    # 마지막 빈 칸에 결과 요약과 단계별 시간 표시
    lines = [f"error: {res.error}", f"flags: {','.join(res.flags) or '-'}", f"sharpness: {res.sharpness:.0f}"]
    lines += [f"{k}: {v:.2f}ms" for k, v in dbg.get("times_ms", {}).items()]
    lines.append(f"total: {sum(dbg.get('times_ms', {}).values()):.2f}ms")
    x0, y0 = 3 * 240 + 6, 262 + 40
    for i, t in enumerate(lines):
        cv2.putText(panel, t, (x0, y0 + i * 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 200), 1, cv2.LINE_AA)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"viz_{Path(image_path).stem}.jpg"
    cv2.imwrite(str(out), panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[visualize] {out}  ({res.error or 'ok'}, flags={res.flags})")
    return out


# --------------------------------------------------------------------------- 2) 성능·품질 점검
def benchmark(pre: EcoPreprocessor, folder: Path, limit: int | None = None, budget_ms: float = 50.0):
    """
    폴더 전체를 전처리해 다음을 출력한다.
      - 단계별 평균·p95 처리 시간과 합계 (예산 50ms 대비)
      - 실패 원인(no_object / multi_object / light_error) 비율
      - 플래그(edge_assist / edge_fallback / blurry / saturated) 비율
      - 클래스별 실패 수 (파일명 첫 단어를 클래스로 간주)
    """
    files = _list_images(folder, limit)
    times = defaultdict(list)
    totals, errors, flags, cls_fail = [], Counter(), Counter(), Counter()
    for f in files:
        frame = cv2.imread(str(f))
        dbg = {}
        t0 = time.perf_counter()
        res = pre.process(frame, out_size=TRAIN_SIZE, debug=dbg)
        totals.append((time.perf_counter() - t0) * 1000)
        for k, v in dbg["times_ms"].items():
            times[k].append(v)
        errors[res.error or "ok"] += 1
        flags.update(res.flags)
        if not res.ok:
            cls_fail[f.name.split("_")[0]] += 1

    n = len(files)
    print(f"\n[benchmark] {folder}  ({n}장, 프레임 {pre.frame_shape[1]}x{pre.frame_shape[0]})")
    print(f"  {'단계':22s} {'평균ms':>8s} {'p95ms':>8s} {'실행 비율':>9s}")
    for k, v in times.items():
        print(f"  {k:22s} {np.mean(v):8.2f} {np.percentile(v, 95):8.2f} {len(v) / n:9.0%}")
    tot_mean, tot_p95 = np.mean(totals), np.percentile(totals, 95)
    verdict = "예산 이내" if tot_p95 <= budget_ms else "예산 초과 → GPU 이전 검토 (H-03)"
    print(f"  {'합계':22s} {tot_mean:8.2f} {tot_p95:8.2f}   ({budget_ms:.0f}ms {verdict})")
    print("  결과: " + ", ".join(f"{k} {v / n:.1%}" for k, v in errors.most_common()))
    print("  플래그: " + (", ".join(f"{k} {v / n:.1%}" for k, v in flags.most_common()) or "없음"))
    if cls_fail:
        print("  클래스별 실패: " + ", ".join(f"{k} {v}" for k, v in cls_fail.most_common()))
    return {"n": n, "mean_ms": tot_mean, "p95_ms": tot_p95, "errors": dict(errors), "flags": dict(flags)}


# --------------------------------------------------------------------------- 3) 파라미터 비교 실험
def sweep(pre: EcoPreprocessor, folder: Path, param: str, values: list, limit: int | None = None):
    """
    전처리 파라미터 하나를 바꿔 가며 성공률·플래그 비율을 비교한다.
    예) thresh 12/20/30 (실험 E1),  w_l 1.0/0.5/0.3 (실험 E2)
    ※ 정확한 판단에는 정답 마스크와의 IoU 비교가 필요하며, 이 표는 1차 선별용이다.
    """
    if param in ("scale", "target", "gain_sigma"):
        raise ValueError(f"{param}는 캘리브레이션 때 쓰이므로 설정 파일을 바꾼 뒤 다시 실행해 비교하세요.")
    files = _list_images(folder, limit)
    frames = [cv2.imread(str(f)) for f in files]
    base_cfg = pre.cfg
    print(f"\n[sweep] {param} ∈ {values}  ({len(files)}장)")
    print(f"  {param:>8s} {'성공':>7s} {'no_obj':>7s} {'multi':>7s} {'edge보조':>8s} {'edge대체':>8s} {'평균ms':>7s}")
    rows = []
    for v in values:
        pre.cfg = replace(base_cfg, **{param: type(getattr(base_cfg, param))(v)})
        cnt, t = Counter(), []
        for fr in frames:
            t0 = time.perf_counter()
            r = pre.process(fr, out_size=TRAIN_SIZE)
            t.append((time.perf_counter() - t0) * 1000)
            cnt[r.error or "ok"] += 1
            cnt.update(r.flags)
        n = len(frames)
        rows.append((v, cnt))
        print(f"  {v:>8} {cnt['ok'] / n:7.1%} {cnt['no_object'] / n:7.1%} {cnt['multi_object'] / n:7.1%}"
              f" {cnt['edge_assist'] / n:8.1%} {cnt['edge_fallback'] / n:8.1%} {np.mean(t):7.2f}")
    pre.cfg = base_cfg
    return rows
