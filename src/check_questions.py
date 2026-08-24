"""questions.json 검증. 벤치마크를 GPU 에 올리기 전에 로컬에서 돌린다.

    python -m src.check_questions

잡아내는 것:
  1. gold 없음        must_include 문구가 원문과 달라 정답 청크를 못 찾음
  2. gold 과다        조건이 헐렁해 문서의 절반이 정답이 됨
  3. 앵커 누수        must_include 문구가 다른 문서에도 있음 (정답이 여러 문서로 번짐)
  4. 지시어           "이 사건/이 문서" 처럼 어느 문서인지 특정 못 하는 질문
  5. 문서 미특정      질문에 대상 문서를 지목하는 고유명사가 하나도 없음
  6. 청크 중복        같은 문서 안에서 두 질문이 같은 청크만 가리킴
  7. 배분 이탈        문서별 문항 수가 권장치(청크수/2)와 크게 다름
"""

from __future__ import annotations

import io
import re
import sys
import contextlib
import collections
from pathlib import Path

import yaml

from .gold import _key, label_gold, load_questions
from .run_benchmark import ROOT, build_corpora

# 어느 문서를 가리키는지 알 수 없게 만드는 표현
DEIXIS = [
    "이 사건", "이 판결", "이 문서", "이 결정", "이 보도자료", "이 기사", "이 판결문",
    "이 결정서", "이 불기소", "이 안내문", "이 논문", "이 포스터", "이 사람",
    "본 사건", "본 판결", "해당 사건", "해당 문서", "해당 판결", "위 사건", "위 문서",
]


# 앵커로 인정하는 상한: 이 개수 이하의 문서에만 나타나는 단어라야 문서를 지목한다고 본다
ANCHOR_MAX_DOCS = 3
# 앵커로 인정할 최소 길이. 한자·가나는 한 글자가 담는 정보가 많아(喻某, 张某某) 2자부터,
# 알파벳·한글은 4자부터 본다.
ANCHOR_MIN_LEN_CJK = 2
ANCHOR_MIN_LEN = 4
CJK = re.compile(r"^[぀-ヿ㐀-䶿一-鿿]+$")


def _has_anchor(q, doc_keys: dict[str, str]) -> bool:
    target = doc_keys.get(q.source, "")
    for raw in re.split(r"[\s,·()\[\]「」『』“”\"'?!]+", q.question):
        tok = _key(raw)
        # 조사·어미가 붙어 있어도 걸리도록 뒤에서부터 한 글자씩 줄여가며 본다
        for end in range(len(tok), ANCHOR_MIN_LEN_CJK - 1, -1):
            cand = tok[:end]
            # "闫某가" 처럼 한자 뒤에 조사가 붙은 토큰은 더 줄여야 순수 한자가 나온다.
            # 여기서 멈추면 두 글자짜리 인명 앵커를 영영 못 본다.
            if len(cand) < ANCHOR_MIN_LEN and not CJK.match(cand):
                continue
            if cand not in target:
                continue
            n = sum(1 for k in doc_keys.values() if cand in k)
            if n <= ANCHOR_MAX_DOCS:
                return True
            break   # 대상 문서엔 있지만 흔한 말이다 — 더 줄여도 흔해질 뿐이다
    return False


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    with contextlib.redirect_stdout(io.StringIO()):
        corpora = build_corpora(cfg)
    questions = load_questions(ROOT / cfg["paths"]["questions_file"])

    chunks = corpora["512"]
    per_doc_chunks = collections.Counter(c.source for c in chunks)
    golds = {v: label_gold(questions, cs)[0] for v, cs in corpora.items()}

    # 문서별 전체 텍스트(정규화) — 앵커 변별력 계산용
    doc_keys = {c.source: _key(c.text) for c in corpora["full"]}

    problems: list[tuple[str, str]] = []

    # 3) 앵커 누수 — source 제한을 걷어내고 문구가 걸리는 문서를 전부 센다
    keyed = [(_key(c.text), c.source) for c in chunks]
    for q in questions:
        for phrase in (q.must_include or []) + (q.any_include or []):
            pk = _key(phrase)
            if not pk:
                continue
            owners = {src for ck, src in keyed if pk in ck}
            if len(owners) > 1:
                other = sorted(owners - {q.source})
                problems.append((q.id, f"앵커 누수 — '{phrase[:34]}' 가 {len(other)}개 다른 문서에도 있음: {other[:2]}"))

    for q in questions:
        # 1) 2) gold 개수
        for v in corpora:
            n = len(golds[v][q.id])
            if n == 0:
                problems.append((q.id, f"[{v}] gold 없음 — must_include 가 원문과 다름"))
            elif v == "512" and n > max(2, per_doc_chunks[q.source] * 0.5):
                problems.append((q.id, f"[{v}] gold 과다 {n}개 / 문서 청크 {per_doc_chunks[q.source]}개"))
        # 4) 지시어
        hit = [w for w in DEIXIS if w in q.question]
        if hit:
            problems.append((q.id, f"지시어 '{hit[0]}' — 어느 문서인지 특정 불가"))
        # 5) 문서 미특정 — 질문 안에 '대상 문서를 실제로 변별하는' 단어가 있는가.
        #    한국어 문서는 앵커도 한국어라 문자 종류로는 못 가린다. 그래서 질문의 각
        #    토큰이 코퍼스의 몇 개 문서에 나타나는지를 세서, 대상 문서를 포함해
        #    ANCHOR_MAX_DOCS 개 이하에만 걸리는 토큰이 하나라도 있으면 통과시킨다.
        #    (조사가 붙어도 걸리도록 앞에서부터 짧게 잘라가며 확인한다)
        if not _has_anchor(q, doc_keys):
            problems.append((q.id, "앵커 없음 — 대상 문서를 지목하는 고유명사·사건번호·금액을 넣을 것"))

    # 6) 같은 문서 안에서 gold 가 완전히 겹치는 질문 쌍
    by_src: dict[str, list] = {}
    for q in questions:
        by_src.setdefault(q.source, []).append(q)
    for src, qs in by_src.items():
        for i, a in enumerate(qs):
            for b in qs[i + 1:]:
                if golds["512"][a.id] and golds["512"][a.id] == golds["512"][b.id]:
                    problems.append((b.id, f"청크 중복 — {a.id} 와 정답 청크가 완전히 같음"))

    # 7) 배분
    print(f"질문 {len(questions)}개 / 문서 {len(by_src)}개 / 청크 {len(chunks)}개\n")
    print(f"{'source':66}{'청크':>5}{'문항':>5}{'권장':>5}")
    for src in sorted(per_doc_chunks):
        want = max(2, min(12, round(per_doc_chunks[src] / 2)))
        have = len(by_src.get(src, []))
        flag = "" if abs(have - want) <= 1 else "   ←"
        if have == 0:
            flag = "   ← 질문 없음"
        print(f"{src:66}{per_doc_chunks[src]:5}{have:5}{want:5}{flag}")

    print()
    if problems:
        print(f"── 문제 {len(problems)}건 " + "─" * 50)
        for qid, msg in problems:
            print(f"  {qid:14} {msg}")
        raise SystemExit(1)
    print("문제 없음 — 모든 질문이 gold 를 갖고, 앵커가 대상 문서에만 걸립니다.")


if __name__ == "__main__":
    main()
