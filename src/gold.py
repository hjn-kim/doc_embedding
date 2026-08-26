"""질문 로드 + 정답(gold) 청크 자동 라벨링.

수작업으로 "정답 청크 번호"를 적는 건 청킹 설정을 바꿀 때마다 다시 해야 해서 비현실적이다.
대신 질문마다 '정답에 반드시 들어있는 문구'(must_include)를 적어두면,
그 문구를 포함한 청크를 자동으로 gold 로 라벨링한다.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .chunker import Chunk


@dataclass
class Question:
    id: str
    question: str
    source: str | None = None       # 어느 문서에 답이 있는지 (scope=own_doc 일 때 사용)
    must_include: list[str] = None  # 전부 포함되어야 gold
    any_include: list[str] = None   # 하나라도 포함되면 gold
    must_exclude: list[str] = None  # 하나라도 있으면 gold 에서 제외 (must_include 의 반대)
    gold_chunks: list[int] = None   # 정답 청크 id 를 직접 지정 (지정하면 위 문구 조건을 대신한다)
    note: str = ""


# questions.json 에서 허용하는 키. 오타를 조용히 무시하지 않기 위해 검사한다.
ALLOWED_KEYS = {
    "id", "question", "source",
    "must_include", "any_include", "must_exclude",
    "gold_chunks", "note",
}


def _key(text: str) -> str:
    """매칭용 정규화: 한글 PDF는 띄어쓰기가 깨지는 경우가 많아 공백을 전부 제거한다."""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", "", text)


def load_questions(path: Path) -> list[Question]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw["questions"] if isinstance(raw, dict) else raw
    questions: list[Question] = []
    seen_ids: set[str] = set()

    for i, item in enumerate(items, start=1):
        qid = str(item.get("id", f"q{i}"))

        unknown = set(item) - ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"질문 {qid}: 알 수 없는 키 {sorted(unknown)}. "
                f"사용 가능: {sorted(ALLOWED_KEYS)}"
            )
        if qid in seen_ids:
            raise ValueError(f"질문 id 가 중복됩니다: {qid}")
        seen_ids.add(qid)

        questions.append(
            Question(
                id=qid,
                question=item["question"],
                source=item.get("source"),
                must_include=item.get("must_include") or [],
                any_include=item.get("any_include") or [],
                must_exclude=item.get("must_exclude") or [],
                gold_chunks=item.get("gold_chunks") or [],
                note=item.get("note", ""),
            )
        )
    return questions


def match_context(text: str, phrases: list[str], width: int = 90) -> str | None:
    """청크에서 must_include/any_include 문구가 실제로 걸린 지점 주변을 잘라 반환.

    청크 앞부분만 보여주는 preview 로는 "왜 이게 정답 청크인지"를 확인할 수 없다.
    (예: 566자 청크의 290자 지점에 답이 있으면 preview 200자로는 안 보인다)

    매칭은 공백을 지운 문자열에서 하므로, 원문 위치로 되돌리기 위해
    '공백 제거 문자열의 i번째 글자 → 원문의 몇 번째 글자' 인덱스를 같이 만든다.
    """
    norm = unicodedata.normalize("NFKC", text)
    lowered = norm.lower()
    packed, back = [], []
    for i, ch in enumerate(lowered):
        if not ch.isspace():
            packed.append(ch)
            back.append(i)
    key = "".join(packed)

    for phrase in phrases:
        pk = _key(phrase)
        if not pk:
            continue
        pos = key.find(pk)
        if pos == -1:
            continue
        start, end = back[pos], back[min(pos + len(pk), len(back)) - 1] + 1
        left, right = max(0, start - width), min(len(norm), end + width)
        snippet = re.sub(r"\s+", " ", norm[left:right]).strip()
        return ("…" if left > 0 else "") + snippet + ("…" if right < len(norm) else "")
    return None


# gold 청크가 전체 코퍼스의 이 비율을 넘으면 조건이 너무 헐렁하다고 본다.
BROAD_RATIO = 0.25


def label_gold(
    questions: list[Question], chunks: list[Chunk]
) -> tuple[dict[str, set[int]], list[str], list[tuple[str, int]]]:
    """질문 id → 정답 청크 id 집합.

    반환: (gold, gold 를 못 찾은 질문 id, 조건이 너무 넓은 질문 [(id, gold 수)])
    """
    chunk_keys = [(c.id, _key(c.text), c.source) for c in chunks]
    gold: dict[str, set[int]] = {}
    unmatched: list[str] = []
    too_broad: list[tuple[str, int]] = []

    valid_ids = {c.id for c in chunks}
    # 문서 단위 색인(변형 full)은 청크 id 체계가 512 와 다르다. whole_documents 가
    # locator 를 "전체" 로 찍어두므로 그것으로 구분하고, 이때 gold_chunks 는
    # 'source 문서 전체' 로 해석한다 (문서 1개 = 청크 1개이므로 정확히 그 문서가 정답).
    doc_level = bool(chunks) and all(c.locator == "전체" for c in chunks)

    for q in questions:
        # gold_chunks 를 적어 두면 문구 매칭 대신 그 청크 id 를 그대로 정답으로 쓴다.
        # 문구가 원문과 미세하게 달라 못 찾는 사고가 없는 대신, 청크 id 는 청킹 설정이
        # 바뀌면 어긋나므로 chunks.json 을 다시 만들면 반드시 재검증해야 한다.
        if q.gold_chunks:
            if doc_level:
                if not q.source:
                    raise ValueError(
                        f"질문 {q.id}: gold_chunks 만 있고 source 가 없어 문서 단위 색인에서 "
                        f"정답 문서를 정할 수 없습니다. source 를 적어주세요."
                    )
                matched = {c.id for c in chunks if c.source == q.source}
                if not matched:
                    raise ValueError(
                        f"질문 {q.id}: source({q.source}) 문서가 코퍼스에 없습니다."
                    )
                gold[q.id] = matched
                continue
            missing = [i for i in q.gold_chunks if i not in valid_ids]
            if missing:
                raise ValueError(
                    f"질문 {q.id}: gold_chunks 에 없는 청크 id {missing}. "
                    f"청킹 설정을 바꿨다면 questions.json 의 정답 id 를 다시 매겨야 합니다."
                )
            if q.source:
                wrong = [c.id for c in chunks
                         if c.id in set(q.gold_chunks) and c.source != q.source]
                if wrong:
                    raise ValueError(
                        f"질문 {q.id}: gold_chunks {wrong} 가 source({q.source}) 가 아닌 "
                        f"다른 문서의 청크입니다."
                    )
            gold[q.id] = set(q.gold_chunks)
            continue

        must = [_key(s) for s in q.must_include if s.strip()]
        any_ = [_key(s) for s in q.any_include if s.strip()]
        excl = [_key(s) for s in q.must_exclude if s.strip()]
        if not must and not any_:
            raise ValueError(
                f"질문 {q.id}: gold_chunks / must_include / any_include 중 하나는 "
                f"반드시 필요합니다. "
                f"(must_exclude 는 단독으로 쓸 수 없습니다 — 제외 조건일 뿐이라 "
                f"'답이 없는 모든 청크'가 정답이 되어버립니다)"
            )

        matched = set()
        for cid, ckey, csource in chunk_keys:
            if q.source and csource != q.source:
                continue
            if must and not all(m in ckey for m in must):
                continue
            if any_ and not any(a in ckey for a in any_):
                continue
            if excl and any(e in ckey for e in excl):
                continue
            matched.add(cid)

        gold[q.id] = matched
        if not matched:
            unmatched.append(q.id)
        elif len(matched) > max(2, BROAD_RATIO * len(chunks)):
            too_broad.append((q.id, len(matched)))

    return gold, unmatched, too_broad
