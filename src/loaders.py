"""문서 로더: .docx / .pdf 에서 텍스트와 메타데이터를 추출한다."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Block:
    """문서에서 뽑아낸 텍스트 한 덩어리 (문단 / 표 행 / 페이지)."""

    text: str
    source: str  # 파일명
    locator: str  # "p.3", "para.12", "table.2" 처럼 원문 위치
    meta: dict = field(default_factory=dict)  # src/metadata.py 가 채운다 (문서종류·기관·작성일·섹션)


def normalize_text(text: str) -> str:
    """전각/호환 문자 정리 + 공백 정규화. 한글 PDF 추출물 정리에 필요."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"[ \t 　]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_docx(path: Path, source: str | None = None) -> list[Block]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    src = source or path.name
    doc = Document(str(path))
    blocks: list[Block] = []

    # 본문 순서대로(문단/표 섞여 있음) 훑기 위해 body XML 요소를 직접 순회한다.
    para_i, table_i = 0, 0
    body = doc.element.body
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            para_i += 1
            text = normalize_text(Paragraph(child, doc).text)
            if text:
                blocks.append(Block(text, src, f"para.{para_i}"))
        elif tag == "tbl":
            table_i += 1
            table = Table(child, doc)
            for row_i, row in enumerate(table.rows, start=1):
                cells = [normalize_text(c.text) for c in row.cells]
                # 같은 셀이 병합으로 중복되는 경우 제거
                deduped, seen = [], set()
                for c in cells:
                    if c and c not in seen:
                        seen.add(c)
                        deduped.append(c)
                line = " | ".join(deduped)
                if line:
                    blocks.append(
                        Block(line, src, f"table.{table_i}.row.{row_i}")
                    )

    return blocks


def load_pdf(path: Path, source: str | None = None) -> list[Block]:
    src = source or path.name
    blocks: list[Block] = []

    try:
        import fitz  # PyMuPDF

        with fitz.open(str(path)) as doc:
            for page_i, page in enumerate(doc, start=1):
                text = normalize_text(page.get_text("text"))
                if text:
                    blocks.append(Block(text, src, f"p.{page_i}"))
        if blocks:
            return blocks
    except ImportError:
        pass

    from pypdf import PdfReader

    reader = PdfReader(str(path))
    for page_i, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if text:
            blocks.append(Block(text, src, f"p.{page_i}"))

    return blocks


# 언어 폴더가 아닌 것 (백업·작업용). 코퍼스에서 제외한다.
SKIP_DIR_PREFIXES = ("_", ".")


def load_documents(data_dir: Path) -> list[Block]:
    """data_dir 아래의 모든 .docx / .pdf 를 재귀적으로 읽어 Block 리스트로 반환.

    source 는 파일명이 아니라 data_dir 기준 상대경로다 ("ko/2015고단7004_판결문.pdf").
    언어 폴더가 다르면 파일명이 같아도 구분되고, 첫 경로 조각이 곧 언어 코드가 된다.
    """
    files = sorted(
        p
        for p in data_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".docx", ".pdf"}
        and not p.name.startswith("~$")
        and not any(part.startswith(SKIP_DIR_PREFIXES) for part in p.relative_to(data_dir).parts[:-1])
    )
    if not files:
        raise FileNotFoundError(
            f"{data_dir} 안에 .docx 또는 .pdf 파일이 없습니다. 문서를 넣어주세요."
        )

    blocks: list[Block] = []
    for path in files:
        rel = path.relative_to(data_dir).as_posix()
        # 최상위 폴더명을 언어 코드로 쓴다. 폴더 없이 바로 놓인 파일은 "(root)".
        lang = rel.split("/")[0] if "/" in rel else "(root)"
        loaded = (
            load_docx(path, rel) if path.suffix.lower() == ".docx" else load_pdf(path, rel)
        )
        if not loaded:
            print(f"  [경고] {rel} 에서 텍스트를 추출하지 못했습니다 "
                  f"(스캔 이미지 PDF일 수 있음 → OCR 필요).")
        for b in loaded:
            b.meta["lang"] = lang
        print(f"  {rel}: {len(loaded)} blocks")
        blocks.extend(loaded)

    return blocks


def load_gt(gt_dir: Path, doc_dir: Path | None = None) -> list[Block]:
    """data/gt/<언어>/<파일명>.txt 를 읽어 Block 리스트로 반환. 색인 대상 본문이다.

    PDF 에서 직접 뽑지 않고 gt 텍스트를 색인하는 이유:
      - 비교 대상 변수는 임베딩 모델이지 PDF 추출기가 아니다. 텍스트 출처를 바꿔도
        모든 모델이 똑같은 청크를 보므로 모델 간 공정성은 그대로다.
      - gt 는 다단 편집 PDF 의 읽는 순서가 바로잡혀 있고, 스캔 이미지 PDF 도 OCR 돼 있다.
        (PyMuPDF 로는 ch/(2020)_0109_488 처럼 한 글자도 못 뽑는 문서가 있다)
      - 무엇보다 '색인하는 텍스트 = must_include 를 베끼는 텍스트' 가 되어,
        문구가 원문과 미세하게 달라 gold 를 못 찾는 사고가 원천적으로 사라진다.

    그 대가로 gt 파일이 곧 코퍼스다. gt 를 손으로 고치면 벤치마크 대상이 바뀐다.

    source 는 gt 파일명이 아니라 원본 문서명이다 ("ko/2015고단7004_판결문.pdf").
    questions.json 의 source 표기를 그대로 유지하기 위해서다.

    블록은 빈 줄(\n\n) 단위로 나눈다. 빈 줄이 없는 gt 는 문서 전체가 한 블록이 되고,
    청킹은 chunker.SEPARATORS 가 "\n" → 문장부호 순으로 알아서 쪼갠다.
    """
    if not gt_dir.is_dir():
        raise FileNotFoundError(
            f"{gt_dir} 가 없습니다. 먼저 `python -m src.make_gt` 로 gt 를 만드세요."
        )

    # 원본 문서 확장자를 되찾기 위한 표: "ko/2015고단7004_판결문" → "ko/2015고단7004_판결문.pdf"
    stem_to_doc: dict[str, str] = {}
    if doc_dir and doc_dir.is_dir():
        for p in doc_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".docx", ".pdf"}:
                rel = p.relative_to(doc_dir).as_posix()
                if not any(part.startswith(SKIP_DIR_PREFIXES) for part in rel.split("/")[:-1]):
                    stem_to_doc[rel.rsplit(".", 1)[0]] = rel

    files = sorted(p for p in gt_dir.rglob("*.txt") if p.is_file())
    if not files:
        raise FileNotFoundError(f"{gt_dir} 안에 .txt 가 없습니다.")

    blocks: list[Block] = []
    for path in files:
        key = path.relative_to(gt_dir).as_posix().rsplit(".", 1)[0]
        source = stem_to_doc.get(key, f"{key}.txt")
        lang = key.split("/")[0] if "/" in key else "(root)"

        text = normalize_text(path.read_text(encoding="utf-8"))
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not parts:
            print(f"  [경고] {key}.txt 가 비어 있습니다.")
            continue

        for i, part in enumerate(parts, start=1):
            blocks.append(Block(part, source, f"블록.{i}", {"lang": lang}))
        print(f"  {source}: {len(parts)} blocks ({len(text):,}자)")

    return blocks
