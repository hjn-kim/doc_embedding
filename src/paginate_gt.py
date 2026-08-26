"""gt 텍스트에 원본 PDF 의 페이지 경계를 빈 줄로 표시한다.

chunking.source: gt 에서 data/gt 는 색인 코퍼스이고, loaders.load_gt 는 빈 줄로
Block 을 나눈다. 빈 줄이 하나도 없는 gt 는 문서 전체가 한 블록이 되어
  - locator 가 전부 "블록.1" 이 되고 (misses.md 에서 위치 추적 불가)
  - 청크 경계가 페이지를 마음대로 넘나든다
이 스크립트는 PDF 페이지 텍스트를 gt 본문에 정렬해 경계에 빈 줄을 끼워 넣는다.

넣는 것은 공백뿐이다. gold._key() 가 공백을 전부 지우므로 색인되는 글자는
한 글자도 바뀌지 않는다. 바뀌는 것은 블록/청크 경계와 locator 뿐이다.

정렬은 '앞 페이지의 끝 문구' 를 gt 에서 찾는 방식이라, gt 의 읽는 순서가 PDF 와
다르면(다단 편집 문서) 실패한다. 실패한 파일은 건드리지 않고 목록으로 보고한다.

    python -m src.paginate_gt              # dry-run: 무엇이 되고 안 되는지만 출력
    python -m src.paginate_gt --write      # 실제로 빈 줄 삽입
    python -m src.paginate_gt ch pil --write
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

from .loaders import load_pdf, normalize_text

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
GT = DATA / "gt"

ANCHOR = 40   # 페이지 경계를 찾을 때 쓰는 앵커 길이 (공백 제외 글자 수)
RETREAT = 400  # 앵커를 페이지 끝(또는 앞)에서 최대 몇 글자까지 물러서며 찾을지
STEP = 10      # 물러서는 간격

# 페이지 맨 끝의 푸터(연락처·페이지번호·반복 헤더)는 gt 를 만들 때 흔히 지워진다.
# 그래서 '페이지의 마지막 40자' 를 그대로 찾으면 실패한다. 끝에서 조금씩 물러서며
# gt 에 남아 있는 마지막 본문 조각을 앵커로 삼는다.


def _key_with_index(text: str) -> tuple[str, list[int]]:
    """공백을 지운 문자열과 '지운 문자열 i번째 → 원문 위치' 색인을 함께 만든다."""
    norm = unicodedata.normalize("NFKC", text).lower()
    packed, back = [], []
    for i, ch in enumerate(norm):
        if not ch.isspace():
            packed.append(ch)
            back.append(i)
    return "".join(packed), back


def _key(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).lower())


def _snap_to_line(text: str, pos: int) -> int:
    """줄 중간에 빈 줄을 넣지 않도록 가장 가까운 줄바꿈으로 옮긴다."""
    nxt = text.find("\n", pos)
    prv = text.rfind("\n", 0, pos)
    if nxt == -1:
        return prv + 1 if prv != -1 else pos
    if prv == -1:
        return nxt + 1
    return (nxt + 1) if (nxt - pos) <= (pos - prv) else (prv + 1)


def paginate(gt_text: str, pages: list[str]) -> tuple[str | None, str]:
    """gt 본문에 페이지 경계 빈 줄을 넣어 반환. 실패하면 (None, 사유)."""
    if len(pages) < 2:
        return None, "페이지 1개"
    if re.search(r"\n\s*\n", gt_text):
        return None, "이미 빈 줄 있음"

    key, back = _key_with_index(gt_text)
    cuts: list[int] = []
    cursor = 0
    for i in range(1, len(pages)):
        prev_key, next_key = _key(pages[i - 1]), _key(pages[i])
        pos = -1
        # 1순위: 앞 페이지의 끝 문구 뒤. 푸터가 지워졌을 수 있으니 끝에서 물러서며 찾는다.
        for off in range(0, RETREAT + 1, STEP):
            end = len(prev_key) - off
            if end - ANCHOR < 0:
                break
            found = key.find(prev_key[end - ANCHOR:end], cursor)
            if found != -1:
                pos = found + ANCHOR
                break
        # 2순위: 다음 페이지의 첫 문구 앞. 헤더가 지워졌을 수 있으니 앞에서 나아가며 찾는다.
        if pos == -1:
            for off in range(0, RETREAT + 1, STEP):
                if off + ANCHOR > len(next_key):
                    break
                found = key.find(next_key[off:off + ANCHOR], cursor)
                if found != -1:
                    pos = found
                    break
        if pos == -1:
            return None, f"p.{i}/{len(pages)} 경계 앵커 못 찾음 (읽는 순서 불일치)"
        cuts.append(back[min(pos, len(back) - 1)])
        cursor = pos

    if cuts != sorted(cuts):
        return None, "경계 순서가 뒤집힘 (읽는 순서 불일치)"

    out, prev = [], 0
    for c in cuts:
        c = _snap_to_line(gt_text, c)
        if c <= prev:          # 같은 줄로 몰린 경계는 건너뛴다
            continue
        out.append(gt_text[prev:c].rstrip())
        prev = c
    out.append(gt_text[prev:].rstrip())
    out = [p for p in out if p.strip()]
    if len(out) < 2:
        return None, "나뉜 조각이 1개"
    return "\n\n".join(out) + "\n", f"{len(out)}블록"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("langs", nargs="*", help="언어 폴더명 (없으면 전체)")
    ap.add_argument("--write", action="store_true", help="실제로 파일을 고친다")
    args = ap.parse_args()

    langs = args.langs or sorted(
        p.name for p in GT.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))
    )

    ok, skip, fail = [], [], []
    for lang in langs:
        for gt_path in sorted((GT / lang).glob("*.txt")):
            pdf = DATA / lang / f"{gt_path.stem}.pdf"
            rel = f"{lang}/{gt_path.stem}"
            if not pdf.exists():
                skip.append((rel, "원본 PDF 없음"))
                continue
            gt_text = gt_path.read_text(encoding="utf-8")
            pages = [normalize_text(b.text) for b in load_pdf(pdf, rel)]
            new, why = paginate(gt_text, pages)
            if new is None:
                (skip if why in ("페이지 1개", "이미 빈 줄 있음") else fail).append((rel, why))
                continue
            ok.append((rel, why, len(pages)))
            if args.write:
                gt_path.write_text(new, encoding="utf-8")

    verb = "삽입함" if args.write else "삽입 가능"
    print(f"\n■ {verb}: {len(ok)}건")
    for rel, why, npg in ok:
        print(f"   {rel[:56]:58} PDF {npg}p → {why}")
    print(f"\n■ 건너뜀: {len(skip)}건")
    for rel, why in skip:
        print(f"   {rel[:56]:58} {why}")
    print(f"\n■ 정렬 실패(손대지 않음): {len(fail)}건")
    for rel, why in fail:
        print(f"   {rel[:56]:58} {why}")
    if not args.write:
        print("\n(dry-run 입니다. 적용하려면 --write)")


if __name__ == "__main__":
    main()
