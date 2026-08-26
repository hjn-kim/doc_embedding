"""공통 청킹. 모든 모델이 '똑같은 청크'를 쓰게 하는 것이 공정 비교의 전제다.

길이 단위는 두 가지다.
  char  : 글자 수. 문자 밀도가 언어마다 달라서(중국어 1.42자/토큰 vs 러시아어 3.96자/토큰)
          같은 500자라도 중국어 청크가 러시아어 청크보다 2.8배 크다. 언어 간 비교에는 못 쓴다.
  token : 고정 기준 토크나이저 하나로 잰 토큰 수. 언어와 무관하게 청크 크기가 일정해진다.
          '모델별' 토크나이저를 쓰면 모델마다 청크 경계가 달라져 위 전제가 깨지므로,
          기준 토크나이저는 반드시 하나로 고정한다(config: chunking.tokenizer).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .loaders import Block

# 큰 단위부터 차례로 시도하는 분할 기준
SEPARATORS = ["\n\n", "\n", ". ", "다. ", "? ", "! ", "; ", ", ", " ", ""]

_TOKENIZERS: dict[str, Any] = {}


def get_tokenizer(name: str):
    """기준 토크나이저를 한 번만 로드해 재사용한다."""
    if name not in _TOKENIZERS:
        from transformers import AutoTokenizer

        _TOKENIZERS[name] = AutoTokenizer.from_pretrained(name)
    return _TOKENIZERS[name]


class Measure:
    """길이 측정기. unit='char' 면 글자 수, 'token' 이면 기준 토크나이저의 토큰 수."""

    def __init__(self, unit: str = "char", tokenizer: str | None = None):
        if unit not in ("char", "token"):
            raise ValueError(f"chunking.unit 은 char 또는 token 이어야 합니다: {unit!r}")
        if unit == "token" and not tokenizer:
            raise ValueError("chunking.unit=token 이면 chunking.tokenizer 를 지정해야 합니다.")
        self.unit = unit
        self.tk = get_tokenizer(tokenizer) if unit == "token" else None

    def __call__(self, text: str) -> int:
        if self.tk is None:
            return len(text)
        return len(self.tk(text, add_special_tokens=False)["input_ids"])

    def _offsets(self, text: str):
        """토큰 경계를 '원문 문자 위치'로 돌려준다. (ids, offsets) 또는 None.

        decode() 로 잘라내면 토크나이저 어휘에 없는 글자가 <unk> 로 바뀌어
        원문이 손상된다 (예: 邛崃 → <unk>). 그래서 자를 때는 항상 토큰 개수만
        토크나이저로 세고, 실제 문자열은 원문에서 슬라이스한다.
        """
        try:
            enc = self.tk(text, add_special_tokens=False, return_offsets_mapping=True)
        except (TypeError, NotImplementedError):  # slow tokenizer 는 offset 이 없다
            return None
        offs = enc.get("offset_mapping")
        if not offs:
            return None
        return enc["input_ids"], offs

    def tail(self, text: str, n: int) -> str:
        """뒤에서 n 단위만큼 잘라 반환 (overlap 용)."""
        if self.tk is None:
            return text[-n:]
        got = self._offsets(text)
        if got is None:
            ids = self.tk(text, add_special_tokens=False)["input_ids"]
            return text if len(ids) <= n else self.tk.decode(ids[-n:])
        ids, offs = got
        return text if len(ids) <= n else text[offs[-n][0]:]

    def hard_split(self, text: str, size: int) -> list[str]:
        """구분자로 더 못 자를 때의 강제 분할."""
        if self.tk is None:
            return [text[i : i + size] for i in range(0, len(text), size)]
        got = self._offsets(text)
        if got is None:
            ids = self.tk(text, add_special_tokens=False)["input_ids"]
            return [self.tk.decode(ids[i : i + size]) for i in range(0, len(ids), size)]
        ids, offs = got
        out = []
        for i in range(0, len(ids), size):
            j = min(i + size, len(ids)) - 1
            piece = text[offs[i][0] : offs[j][1]]
            if piece.strip():
                out.append(piece)
        return out


@dataclass
class Chunk:
    id: int
    text: str
    source: str
    locator: str
    meta: dict = field(default_factory=dict)  # 부모 블록에서 물려받는다

    def to_dict(self) -> dict:
        return asdict(self)


def _split_recursive(text: str, body_size: int, seps: list[str], mz: Measure) -> list[str]:
    if mz(text) <= body_size:
        return [text]
    if not seps or seps[0] == "":
        return mz.hard_split(text, body_size)

    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    pieces: list[str] = []
    buf = ""
    for i, part in enumerate(parts):
        candidate = part if i == len(parts) - 1 else part + sep
        if mz(buf) + mz(candidate) <= body_size:
            buf += candidate
        else:
            if buf:
                pieces.append(buf)
            if mz(candidate) > body_size:
                pieces.extend(_split_recursive(candidate, body_size, rest, mz))
                buf = ""
            else:
                buf = candidate
    if buf:
        pieces.append(buf)
    return [p for p in pieces if p.strip()]


def _apply_overlap(pieces: list[str], overlap: int, mz: Measure) -> list[str]:
    if overlap <= 0 or len(pieces) < 2:
        return pieces
    out = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:]):
        out.append(mz.tail(prev, overlap) + cur)
    return out


def _block_span(starts: list[int], locs: list[str], a: int, b: int) -> tuple[int, str]:
    """[a, b) 구간이 걸치는 블록 범위 → (시작 블록 index, locator 문자열)."""
    first = 0
    for i, st in enumerate(starts):
        if st <= a:
            first = i
        else:
            break
    last = first
    for i in range(first, len(starts)):
        if starts[i] < b:
            last = i
        else:
            break
    if first == last:
        return first, locs[first]
    # "블록.1~블록.3" 은 길기만 하므로 뒤쪽은 번호만 남긴다 → "블록.1~3"
    tail = locs[last].rsplit(".", 1)[-1]
    return first, f"{locs[first]}~{tail}"


def chunk_blocks(
    blocks: list[Block],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    min_chunk_chars: int = 30,
    unit: str = "char",
    tokenizer: str | None = None,
    block_boundary: str = "soft",
) -> list[Chunk]:
    """Block 리스트 → Chunk 리스트.

    chunk_size 는 overlap 을 붙인 '최종' 청크 크기의 상한이다.
    그래서 본문은 chunk_size - chunk_overlap 까지만 채우고 앞에 이전 청크 꼬리를 붙인다.
    (이렇게 하지 않으면 512 로 맞춰도 실제 청크가 610 토큰이 되어 e5 에서 조용히 잘린다)

    block_boundary 는 블록(=페이지/문단) 경계를 어떻게 다룰지다.
      soft : 문서 전체를 이어 붙여 자른다. 청크 크기가 균일해지고, 페이지가 몇 장이든
             청크 수가 늘지 않는다. 원래 위치는 locator 에 "블록.2~3" 으로 남긴다.
      hard : 블록마다 끊는다. 청크가 페이지를 넘지 않는 대신, 페이지 끝마다 짧은
             청크가 생겨 청크 수가 20% 늘고 크기가 들쭉날쭉해진다.

    soft 가 기본이다. 청크 경계가 '원본 PDF 의 페이지가 어디서 끊겼는지' 에 좌우되면
    문서마다 조건이 달라져 언어 간 비교가 흔들리기 때문이다.
    """
    if block_boundary not in ("soft", "hard"):
        raise ValueError(
            f"chunking.block_boundary 는 soft 또는 hard 여야 합니다: {block_boundary!r}"
        )
    mz = Measure(unit, tokenizer)
    body_size = max(1, chunk_size - max(chunk_overlap, 0))
    chunks: list[Chunk] = []
    next_id = 0

    # source 별로 순서를 유지한 채 그룹핑
    groups: dict[str, list[Block]] = {}
    for b in blocks:
        groups.setdefault(b.source, []).append(b)

    for source, group in groups.items():
        buf_text, buf_locators, buf_metas = "", [], []

        def flush() -> None:
            nonlocal buf_text, buf_locators, buf_metas, next_id
            if not buf_text.strip():
                buf_text, buf_locators, buf_metas = "", [], []
                return
            pieces = _split_recursive(buf_text.strip(), body_size, SEPARATORS, mz)
            pieces = _apply_overlap(pieces, chunk_overlap, mz)
            locator = (
                buf_locators[0]
                if len(set(buf_locators)) == 1
                else f"{buf_locators[0]}~{buf_locators[-1]}"
            )
            # 여러 블록이 합쳐졌으면 첫 블록의 meta 를 대표로 쓴다.
            meta = dict(buf_metas[0]) if buf_metas else {}
            for piece in pieces:
                piece = re.sub(r"\s+", " ", piece).strip()
                if len(piece) < min_chunk_chars:
                    continue
                # overlap 꼬리는 decode 한 텍스트라, 앞 조각과 이어붙여 다시 토큰화하면
                # 경계에서 토큰이 한두 개 늘어날 수 있다. chunk_size 를 넘기면
                # 모델 입력에서 조용히 잘리므로 여기서 확실히 깎는다.
                if mz(piece) > chunk_size:
                    piece = mz.hard_split(piece, chunk_size)[0]
                chunks.append(Chunk(next_id, piece, source, locator, dict(meta)))
                next_id += 1
            buf_text, buf_locators, buf_metas = "", [], []

        if block_boundary == "hard":
            for b in group:
                if mz(buf_text) + mz(b.text) + 1 > body_size and buf_text:
                    flush()
                buf_text = f"{buf_text}\n{b.text}" if buf_text else b.text
                buf_locators.append(b.locator)
                buf_metas.append(b.meta)
                if mz(buf_text) >= body_size:
                    flush()
            flush()
            continue

        # ── soft: 문서 전체를 이어 붙여 한 번에 자르고, 조각마다 원래 블록을 되짚는다 ──
        full = "\n".join(b.text for b in group)
        if not full.strip():
            continue
        starts, off = [], 0
        for b in group:                      # 각 블록이 full 의 몇 번째 글자에서 시작하는지
            starts.append(off)
            off += len(b.text) + 1           # +1 은 이어 붙일 때 넣은 개행
        locs = [b.locator for b in group]

        pieces = _split_recursive(full, body_size, SEPARATORS, mz)
        # 조각은 full 에서 그대로 잘라낸 것이라 순서대로 위치를 되찾을 수 있다.
        piece_at, cursor = [], 0
        for piece in pieces:
            found = full.find(piece, cursor)
            if found == -1:                  # 이론상 없지만, 어긋나면 커서를 그대로 쓴다
                found = cursor
            piece_at.append(found)
            cursor = found + len(piece)

        for i, piece in enumerate(pieces):
            if i == 0 or chunk_overlap <= 0:
                text, begin = piece, piece_at[i]
            else:
                tail = mz.tail(pieces[i - 1], chunk_overlap)
                text, begin = tail + piece, max(0, piece_at[i] - len(tail))
            end = piece_at[i] + len(piece)

            text = re.sub(r"\s+", " ", text).strip()
            if len(text) < min_chunk_chars:
                continue
            if mz(text) > chunk_size:
                text = mz.hard_split(text, chunk_size)[0]

            bi, locator = _block_span(starts, locs, begin, end)
            chunks.append(Chunk(next_id, text, source, locator, dict(group[bi].meta)))
            next_id += 1

    return chunks


def whole_documents(blocks: list[Block], min_chunk_chars: int = 30) -> list[Chunk]:
    """문서 1개 = 청크 1개. 자르지 않는다 (색인 단위 'full').

    컨텍스트 상한이 512 인 모델에 이걸 주면 문서 뒷부분이 통째로 잘려나가므로,
    긴 컨텍스트를 가진 모델에만 적용해야 한다 (run_benchmark 에서 걸러낸다).
    """
    groups: dict[str, list[Block]] = {}
    for b in blocks:
        groups.setdefault(b.source, []).append(b)

    chunks: list[Chunk] = []
    for i, (source, group) in enumerate(groups.items()):
        text = re.sub(r"\s+", " ", "\n".join(b.text for b in group)).strip()
        if len(text) < min_chunk_chars:
            continue
        chunks.append(Chunk(len(chunks), text, source, "전체", dict(group[0].meta)))
    return chunks
