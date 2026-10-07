"""
에코소트 전처리 실행기.

준비물
  data/backgrounds/          빈 라이트박스 프레임 10장 (같은 카메라 설정으로 촬영)
  data/raw/own/<set>/        자체 촬영 원본 (파일명: 클래스_품목_물체ID_조건_번호.jpg)
  data/raw/public/...        공개 데이터 (download 명령으로 받기)

순서
  1) python run_preprocess.py download trashnet          # 공개 데이터 받기 (taco, aihub도 가능)
  2) python run_preprocess.py own                         # 자체 촬영 → 전처리 (+ 흐림 기준 자동 설정)
  3) python run_preprocess.py public --trashnet           # 공개 데이터 → 오려내기 → 배경 합성 → 전처리
  4) python run_preprocess.py manifest                    # 물체 ID 단위 학습/검증 분할
  (2~4를 한 번에: python run_preprocess.py all --trashnet)

점검
  python run_preprocess.py visualize data/raw/own/train_pool/plastic_xxx.jpg   # 단계별 그림
  python run_preprocess.py benchmark data/raw/own/train_pool                     # 시간·실패율
  python run_preprocess.py sweep data/raw/own/train_pool --param thresh --values 12 20 30
"""
import argparse
import json
import shutil
from pathlib import Path

from ecopre.config import BACKGROUND_DIR, CUTOUT_DIR, PROCESSED_DIR, PUBLIC_RAW_DIR, TRASHNET_MAP


def load_pre():
    """빈 배경 사진으로 캘리브레이션한 전처리기 (data/pre_config.json 설정 반영)."""
    from ecopre.preprocess import EcoPreprocessor
    return EcoPreprocessor.from_dir(BACKGROUND_DIR)


def cmd_download(a):
    from ecopre import download
    {"trashnet": download.download_trashnet,
     "taco": lambda: download.download_taco(a.max_images),
     "aihub": download.aihub_guide}[a.dataset]()


def cmd_own(a):
    from ecopre.prepare_own import build_own_processed
    shutil.rmtree(PROCESSED_DIR / "own", ignore_errors=True)
    build_own_processed(load_pre())


def cmd_public(a):
    from ecopre import prepare_public as pp
    if a.clean:
        shutil.rmtree(CUTOUT_DIR, ignore_errors=True)
    # 1) 물체 오려내기 (원본 → data/cutouts/<출처>/<클래스>/*.png)
    if a.trashnet:
        src = Path(a.trashnet) if a.trashnet != "auto" else PUBLIC_RAW_DIR / "trashnet" / "dataset-resized"
        print(f"[public] TrashNet 오려내기 (GrabCut): {src}")
        pp.cutouts_from_class_folders(src, TRASHNET_MAP, "trashnet", a.limit_per_class, a.workers)
    if a.taco:
        print("[public] TACO 오려내기 (분할 마스크)")
        pp.cutouts_from_taco(PUBLIC_RAW_DIR / "taco", a.limit_per_class)
    if a.folder:
        mapping = json.loads(Path(a.map).read_text(encoding="utf-8"))
        print(f"[public] 폴더형 데이터 오려내기: {a.folder}")
        pp.cutouts_from_class_folders(Path(a.folder), mapping, a.name, a.limit_per_class, a.workers)
    # 2) 배경 합성 → 같은 전처리 → data/processed/public/
    shutil.rmtree(PROCESSED_DIR / "public", ignore_errors=True)
    pp.build_public_processed(load_pre(), a.per_cutout, a.color_match)


def cmd_manifest(a):
    from ecopre.manifest import build_manifest
    build_manifest()


def cmd_all(a):
    cmd_own(a)
    cmd_public(a)
    cmd_manifest(a)


def cmd_visualize(a):
    from ecopre.debug_tools import _list_images, visualize
    pre = load_pre()
    for f in _list_images(Path(a.path), a.limit):
        visualize(pre, f)


def cmd_benchmark(a):
    from ecopre.debug_tools import benchmark
    benchmark(load_pre(), Path(a.folder), a.limit, a.budget)


def cmd_sweep(a):
    from ecopre.debug_tools import sweep
    sweep(load_pre(), Path(a.folder), a.param, a.values, a.limit)


def build_parser():
    p = argparse.ArgumentParser(description="에코소트 전처리 파이프라인")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download", help="공개 데이터 다운로드")
    s.add_argument("dataset", choices=["trashnet", "taco", "aihub"])
    s.add_argument("--max-images", type=int, default=None, help="TACO 일부만 받기")
    s.set_defaults(fn=cmd_download)

    s = sub.add_parser("own", help="자체 촬영 원본 전처리")
    s.set_defaults(fn=cmd_own)

    def public_args(s):
        s.add_argument("--trashnet", nargs="?", const="auto", help="TrashNet 폴더 (생략 시 기본 위치)")
        s.add_argument("--taco", action="store_true")
        s.add_argument("--folder", help="클래스별 폴더 구조 데이터 (예: AI Hub)")
        s.add_argument("--map", help="--folder용 클래스 매핑 JSON")
        s.add_argument("--name", default="aihub")
        s.add_argument("--limit-per-class", type=int, default=None)
        s.add_argument("--per-cutout", type=int, default=2, help="물체 하나당 합성 장수")
        s.add_argument("--color-match", action="store_true", help="자체 데이터 색 분포에 맞춤 (실험 E4)")
        s.add_argument("--clean", action="store_true", help="기존 오려낸 물체를 지우고 시작")
        s.add_argument("--workers", type=int, default=0, help="오려내기 병렬 프로세스 수 (0=CPU 코어 수)")

    s = sub.add_parser("public", help="공개 데이터 오려내기 → 합성 → 전처리")
    public_args(s)
    s.set_defaults(fn=cmd_public)

    s = sub.add_parser("manifest", help="학습/검증/테스트 분할 목록 생성")
    s.set_defaults(fn=cmd_manifest)

    s = sub.add_parser("all", help="own → public → manifest 한 번에")
    public_args(s)
    s.set_defaults(fn=cmd_all)

    s = sub.add_parser("visualize", help="단계별 처리 결과 그림 저장 (debug/)")
    s.add_argument("path", help="이미지 파일 또는 폴더")
    s.add_argument("--limit", type=int, default=5)
    s.set_defaults(fn=cmd_visualize)

    s = sub.add_parser("benchmark", help="처리 시간·실패율 점검")
    s.add_argument("folder")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--budget", type=float, default=50.0, help="전처리 시간 예산 (ms)")
    s.set_defaults(fn=cmd_benchmark)

    s = sub.add_parser("sweep", help="파라미터 비교 실험")
    s.add_argument("folder")
    s.add_argument("--param", required=True, help="예: thresh, w_l, close_k, edge_thresh")
    s.add_argument("--values", nargs="+", required=True)
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(fn=cmd_sweep)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.fn(args)
