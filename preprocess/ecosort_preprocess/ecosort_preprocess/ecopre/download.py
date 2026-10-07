"""
공개 데이터셋 다운로드.

  TrashNet : GitHub 저장소의 dataset-resized.zip (약 43MB, 6개 클래스 약 2,500장)
  TACO     : GitHub의 annotations.json + Flickr에 올라간 원본 이미지 (약 1,500장)
  AI Hub   : 회원가입·승인이 필요해 자동 다운로드 불가 → 내려받은 뒤 public.py의 폴더 가져오기 사용

※ 사용 전 각 데이터셋의 라이선스를 반드시 확인하세요.
"""
import json
import shutil
import zipfile
from pathlib import Path

import requests

from .config import PUBLIC_RAW_DIR

TRASHNET_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
TACO_ANN_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"


def _download(url: str, dst: Path, chunk=1 << 20):
    """스트리밍 다운로드 (진행률 표시)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done, shown = 0, -10
        with open(dst, "wb") as f:
            for part in r.iter_content(chunk):
                f.write(part)
                done += len(part)
                pct = done / total * 100 if total else 0
                if total and pct - shown >= 10:   # 10% 단위로만 표시
                    shown = pct
                    print(f"  {dst.name}: {pct:5.1f}%")


def download_trashnet() -> Path:
    """TrashNet 다운로드 후 압축 해제 → data/raw/public/trashnet/dataset-resized/<class>/*.jpg"""
    root = PUBLIC_RAW_DIR / "trashnet"
    out = root / "dataset-resized"
    if out.exists() and any(out.iterdir()):
        print(f"[download] TrashNet 이미 있음: {out}")
        return out
    zpath = root / "dataset-resized.zip"
    print("[download] TrashNet 다운로드 중...")
    _download(TRASHNET_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(root)
    zpath.unlink()
    # 압축 안에 들어 있는 macOS 메타데이터 폴더 정리
    for junk in list(root.rglob("__MACOSX")):
        shutil.rmtree(junk, ignore_errors=True)
    counts = {d.name: len(list(d.glob("*.jpg"))) for d in sorted(out.iterdir()) if d.is_dir()}
    print(f"[download] TrashNet 완료: {counts}")
    return out


def download_taco(max_images: int | None = None) -> Path:
    """
    TACO 주석 파일과 이미지 다운로드 → data/raw/public/taco/{annotations.json, images/...}
    이미지는 Flickr 서버에서 받으므로 네트워크 정책에 따라 막힐 수 있다.
    max_images로 일부만 받아 먼저 테스트할 수 있다.
    """
    root = PUBLIC_RAW_DIR / "taco"
    ann_path = root / "annotations.json"
    if not ann_path.exists():
        print("[download] TACO 주석 다운로드 중...")
        _download(TACO_ANN_URL, ann_path)
    ann = json.loads(ann_path.read_text())
    images = ann["images"][:max_images] if max_images else ann["images"]
    ok = fail = 0
    for i, im in enumerate(images):
        dst = root / "images" / im["file_name"]
        if dst.exists():
            ok += 1
            continue
        url = im.get("flickr_640_url") or im.get("flickr_url")   # 640px 버전이 가볍다
        try:
            _download(url, dst)
            ok += 1
        except Exception as e:  # noqa: BLE001  (네트워크 오류는 건너뛰고 계속)
            fail += 1
            print(f"  실패 {im['file_name']}: {e}")
        if (i + 1) % 50 == 0:
            print(f"[download] TACO {i + 1}/{len(images)}")
    print(f"[download] TACO 이미지 완료: 성공 {ok}, 실패 {fail}")
    return root


def aihub_guide():
    print("""
[AI Hub 생활 폐기물 이미지]
1) https://aihub.or.kr 에서 회원가입 후 '생활 폐기물 이미지' 데이터 이용 신청
2) 내려받은 이미지를 클래스별 폴더로 정리:  data/raw/public/aihub/<원본클래스명>/*.jpg
3) 원본 클래스명 → 에코소트 클래스 매핑 JSON 작성 (예: {"페트병": "plastic", "캔류": "can"})
4) python pipeline.py prepare-public --folder data/raw/public/aihub --map mapping.json
""")
