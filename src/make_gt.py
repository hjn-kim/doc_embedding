"""PDF/DOCX → data/gt/<언어>/<파일명>.txt 본문 텍스트 생성.

config 의 chunking.source: gt 에서 data/gt 는 **벤치마크가 색인하는 코퍼스 그 자체**다
(참조용 사본이 아니다). 여기 있는 텍스트가 곧 청크가 되고, questions.json 의
must_include 도 여기서 베낀다. 즉 '색인 텍스트 = 베끼는 텍스트' 라서 문구 불일치로
gold 를 못 찾는 사고가 없다. 대신 gt 를 손으로 고치면 벤치마크 대상이 바뀐다.

이 스크립트는 PyMuPDF 추출본을 만든다. 다단 편집 PDF 는 읽는 순서가 뒤섞이고
스캔 이미지 PDF 는 한 글자도 안 나오므로, 그런 문서는 생성 결과를 그대로 쓰지 말고
OCR·수작업으로 고친 텍스트를 넣어라. 고친 텍스트가 코퍼스가 되므로 문제되지 않는다.

블록(=청크 묶음 단위)은 빈 줄로 나뉜다. 페이지마다 빈 줄을 남겨 두면 청크 경계가
페이지를 넘지 않고 locator 도 살아난다.

    python -m src.make_gt                 # gt 가 없는 문서만 생성
    python -m src.make_gt ru              # 특정 언어 폴더만
    python -m src.make_gt ru --force      # 이미 있어도 덮어쓰기
    python -m src.make_gt --check         # 쓰지 않고 기존 gt 와 어긋난 곳만 보고
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path

from .loaders import load_docx, load_pdf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
GT = DATA / "gt"

# 블록(페이지/문단) 사이 구분자. 빈 줄이라 _key() 비교에는 영향이 없고,
# 사람이 페이지 경계를 알아볼 수 있다. 페이지 번호 같은 표시를 넣으면
# 그 줄을 걸친 must_include 가 깨지므로 넣지 않는다.
SEP = "\n\n"


def extract(path: Path, source: str) -> str:
    load = load_docx if path.suffix.lower() == ".docx" else load_pdf
    return SEP.join(b.text for b in load(path, source))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("langs", nargs="*", help="언어 폴더명 (없으면 전체)")
    ap.add_argument("--force", action="store_true", help="기존 gt 를 덮어쓴다")
    ap.add_argument("--check", action="store_true",
                    help="쓰지 않고 기존 gt 와 파이프라인 추출의 차이만 보고")
    args = ap.parse_args()

    langs = args.langs or sorted(
        p.name for p in DATA.iterdir()
        if p.is_dir() and p.name != "gt" and not p.name.startswith(("_", "."))
    )

    made = skipped = 0
    drift: list[tuple[str, float, int, int]] = []
    for lang in langs:
        src_dir = DATA / lang
        if not src_dir.is_dir():
            print(f"[건너뜀] {src_dir} 없음")
            continue
        for path in sorted(src_dir.iterdir()):
            if path.suffix.lower() not in {".pdf", ".docx"} or path.name.startswith("~$"):
                continue
            out = GT / lang / f"{path.stem}.txt"
            text = extract(path, f"{lang}/{path.name}")
            if not text.strip():
                print(f"  [경고] {lang}/{path.name}: 텍스트 추출 실패 "
                      f"(스캔 이미지 PDF 일 수 있음 → OCR 필요)")
                continue

            if out.exists():
                cur = out.read_text(encoding="utf-8")
                ratio = difflib.SequenceMatcher(None, cur, text).ratio()
                if args.check:
                    drift.append((f"{lang}/{path.name}", ratio, len(cur), len(text)))
                    continue
                if not args.force:
                    print(f"  [있음] {lang}/{path.stem}.txt (유사도 {ratio:.3f}) "
                          f"— 덮어쓰려면 --force")
                    skipped += 1
                    continue

            if args.check:
                drift.append((f"{lang}/{path.name}", 0.0, 0, len(text)))
                continue

            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(f"  [생성] {out.relative_to(ROOT)}  {len(text):,}자")
            made += 1

    if args.check:
        print(f"{'source':52}{'유사도':>8}{'기존':>8}{'추출':>8}")
        for name, ratio, a, b in sorted(drift, key=lambda x: x[1]):
            flag = "  ← gt 없음" if a == 0 else ("  ← 어긋남" if ratio < 0.99 else "")
            print(f"{name[:50]:52}{ratio:8.3f}{a:8,}{b:8,}{flag}")
        return

    print(f"\n생성 {made}건 / 건너뜀 {skipped}건")


if __name__ == "__main__":
    main()
