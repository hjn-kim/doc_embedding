
from __future__ import annotations

import json
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from shared_utils import apply_common_styles

# 결과·설정 파일은 앱 파일 옆이 아니라 서버 공용 폴더에 둔다. 2-1 이
# ../document_lab 에서 rag_api 를 찾는 것과 같은 자리이고, 임베딩 벤치마크
# 몫으로 그 아래 embedding/ 하나를 쓴다.
DATA_DIR = Path(__file__).resolve().parent.parent / "document_lab" / "embedding"

# 결과2 탭이 보는 판. 같은 벤치마크를 질문 번역본으로 돌린 결과다.
#   python -m src.run_benchmark --questions questions/questions_lang.json --out results_lang
LANG_DIR = DATA_DIR / "results_lang"

# set_page_config 는 다른 st 호출보다 먼저여야 해서 공통 스타일보다 위에 둔다.
st.set_page_config(page_title="임베딩 모델 비교", page_icon="📊", layout="wide")
apply_common_styles()


# ── 색 ────────────────────────────────────────────────────────
# dataviz 레퍼런스 팔레트. 두 계열 조합은 light/dark 모두 검증기를 통과했다
# (CVD ΔE 24.7 / 26.8, 일반시야 33.6 / 31.8, 대비 3:1 이상).
# 어두운 쪽은 밝은 쪽을 그냥 뒤집은 게 아니라, 어두운 표면에 맞춰 따로 고른 단계다.
def app_theme_is_dark() -> bool:
    """앱이 지금 어두운 테마로 그려지고 있는지. 브라우저 Settings 변경을 실시간 반영."""
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return st.get_option("theme.base") == "dark"


# 히트맵용 순차 램프. 계열색과 같은 파랑 한 가지를 밝기만 바꿔 쓴다(무지개 금지).
# "0 에 가까운 값이 표면 쪽으로 물러난다"는 규칙 때문에 어두운 테마는 순서를 뒤집는다.
SEQ_LIGHT = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]
SEQ_DARK = ["#0d366b", "#184f95", "#2a78d6", "#5598e7", "#9ec5f4"]


def palette(dark: bool) -> dict[str, str]:
    return {
        "series1": "#3987e5" if dark else "#2a78d6",   # 파랑
        "series2": "#d95926" if dark else "#eb6834",   # 주황
        "surface": "#1a1a19" if dark else "#fcfcfb",
        "grid": "#2c2c2a" if dark else "#e1e0d9",
        "muted": "#898781",                             # 축·라벨 (양쪽 공용)
        "seq": SEQ_DARK if dark else SEQ_LIGHT,
        # 진한 셀 위 / 옅은 셀 위 글자색. 어느 쪽 테마든 셀 대비를 확보한다.
        "on_strong": "#f3f2ef",
        "on_weak": "#2c2c2a",
    }


# 질문 id 접두사가 곧 문서의 언어다 (ch-01-1 → ch). 질문 자체는 전부 한국어이므로
# 이 축은 "한국어로 물었을 때 그 언어 문서를 얼마나 잘 찾아오는가"를 뜻한다.
LANG_LABEL = {
    "ch": "중국어",
    "en": "영어",
    "ko": "한국어",
    "pil": "타갈로그어",
    "ru": "러시아어",
    "uz": "우즈베크어",
    "vn": "베트남어",
}


# 검색 단위 설명. run_benchmark 의 index.variants 와 같은 이름을 쓴다.
INDEX_DESC = {
    "512": "510토큰 청크 1개 = 벡터 1개. 정답 **청크**를 찾는 문제이고, "
           "모든 모델이 문서 전문을 동등하게 본다.",
    "full": "문서 1개 = 벡터 1개. 정답 **문서**를 찾는 문제이고, "
            "컨텍스트 상한이 짧은 모델은 애초에 이 검색에서 빠진다.",
}


# config.yaml 의 prefix_style 을 화면 문구로. 모델마다 질문·문서에 붙이는 접두사가
# 다른데, 이걸 모르면 같은 계열 모델의 점수 차이를 엉뚱하게 읽는다.
PREFIX_DESC = {
    "none": "없음",
    "e5": "query: / passage:",
    "e5_inst": "Instruct+Query (질문에만)",
}


# 상태 색은 고정 — 계열 색으로 재사용하지 않는다. 항상 숫자/기호와 함께 쓴다.
RANK_TINT = {
    1: "rgba(12,163,12,0.20)",     # good     — 1등으로 맞힘
    2: "rgba(250,178,25,0.20)",    # warning  — 2~3등
    3: "rgba(250,178,25,0.20)",
    4: "rgba(236,131,90,0.20)",    # serious  — 4~5등
    5: "rgba(236,131,90,0.20)",
}
MISS_TINT = "rgba(208,59,59,0.20)"  # critical — 상위 5개 밖


def themed(chart, p: dict[str, str]):
    """차트에 배경·여백·축 스타일을 입힌다.

    배경과 여백을 .configure() 로 주면 안 된다 — Altair 의 .configure() 는 config 를
    통째로 갈아끼워서 앞서 부른 .configure_axis() 설정이 조용히 사라진다.
    둘 다 Vega-Lite 최상위 속성이므로 .properties() 로 준다.
    격자는 실선 hairline 으로 눌러 데이터만 도드라지게 한다.
    """
    return (
        chart.properties(
            background=p["surface"],
            padding={"left": 12, "right": 12, "top": 12, "bottom": 12},
        )
        .configure_axis(
            labelColor=p["muted"], titleColor=p["muted"], tickColor=p["grid"],
            domainColor=p["grid"], gridColor=p["grid"], gridDash=[], grid=True,
        )
        .configure_view(strokeWidth=0)
    )


# ── 데이터 로드 ───────────────────────────────────────────────
RESULT_FILES = ("summary.csv", "details.json", "chunks.json")


def results_stamp(d: Path) -> float:
    """결과 파일들의 최신 수정 시각. load_results 의 캐시 키로 쓴다.

    폴더 mtime 을 쓰면 안 된다 — 파일이 추가·삭제될 때만 바뀌고 같은 이름으로
    덮어쓰면 그대로라, 벤치마크를 다시 돌려도 옛 결과가 계속 보인다.
    """
    times = [(d / f).stat().st_mtime for f in RESULT_FILES if (d / f).exists()]
    return max(times) if times else 0.0


@st.cache_data(show_spinner=False)
def load_results(results_dir: str, stamp: float):
    """stamp 는 캐시 키 용도로만 받는다 (본문에서 쓰지 않는 게 정상)."""
    d = Path(results_dir)
    summary = pd.read_csv(d / "summary.csv")
    # 검색 값은 512(숫자)와 full(문자)이 섞여 있다. 문자열로 통일해야 비교·groupby 가 안전하다.
    summary["검색"] = (
        summary["검색"].astype(str) if "검색" in summary.columns else "-"
    )

    details = json.loads((d / "details.json").read_text(encoding="utf-8"))

    # chunks.json 은 검색 변형별 dict — {"512": [청크...], "full": [문서...]}
    chunks_path = d / "chunks.json"
    raw = json.loads(chunks_path.read_text(encoding="utf-8")) if chunks_path.exists() else {}
    if isinstance(raw, list):          # 검색 변형이 없던 시절 결과 호환
        raw = {"-": raw}
    chunks = {str(k): v for k, v in raw.items()}
    return summary, details, chunks


@st.cache_data(show_spinner=False)
def load_model_specs(config_path: str, stamp: float) -> pd.DataFrame:
    """config.yaml 의 모델 정의를 읽어 hf_id 로 붙일 수 있는 표로 만든다.

    summary.csv 에는 점수만 있고 '어떤 조건으로 쟀는지'(프리픽스·백엔드·컨텍스트
    상한·dtype)가 없다. 그 조건이 곧 점수 차이의 원인인 경우가 많아서 함께 보여준다.
    config 가 없거나 깨져도 대시보드는 그대로 떠야 하므로 실패하면 빈 표를 준다.
    """
    try:
        import yaml
    except ImportError:
        return pd.DataFrame()
    f = Path(config_path)
    if not f.exists():
        return pd.DataFrame()
    try:
        cfg = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    except Exception:
        return pd.DataFrame()
    rows = []
    for m in cfg.get("models") or []:
        if not isinstance(m, dict) or not m.get("hf_id"):
            continue
        style = str(m.get("prefix_style", "none"))
        rows.append({
            "hf_id": m["hf_id"],
            "백엔드": m.get("backend", "-"),
            "프리픽스": PREFIX_DESC.get(style, style),
            "컨텍스트 상한": m.get("max_context"),
            "dtype": m.get("dtype") or "fp16 (기본)",
            "instruction": m.get("instruction") or "",
        })
    return pd.DataFrame(rows)


def index_sort_key(v: str):
    """검색 단위 정렬 — 숫자 변형(512, 1024…)이 먼저, full 같은 이름이 뒤."""
    return (0, int(v)) if str(v).isdigit() else (1, str(v))


# 언어별 탭 모델 나열 순서 — 점수와 무관하게 계열끼리 붙여 고정한다.
# (기본명, 하이브리드 여부) 로 구분한다: bge-m3 는 dense 와 hybrid 가 따로 있다.
LANG_MODEL_ORDER = [
    ("kure-v1", False),
    ("bge-m3", True),
    ("bge-m3", False),
    ("harrier-0.6b", False),
    ("harrier-270m", False),
    ("e5-small-ko", False),
    ("e5-large-instruct", False),
]


def lang_model_key(m: str):
    """`bge-m3 [512] [hybrid(dense+sparse)]` 같은 라벨을 고정 순서 위치로."""
    key = (m.split(" [")[0], "hybrid" in m)
    return (LANG_MODEL_ORDER.index(key) if key in LANG_MODEL_ORDER
            else len(LANG_MODEL_ORDER), m)


def build_question_frame(details: list[dict], unit_of: dict[str, str]) -> pd.DataFrame:
    """질문 × 모델 단위 표. gold_rank = 정답 단위가 몇 등으로 올라왔는지(없으면 NaN).

    details.json 에는 검색 컬럼이 없다. 모델 라벨이 `kure-v1 [512]` 꼴이라
    summary 의 model→검색 매핑으로 붙이고, 매핑에 없으면 라벨에서 직접 뽑는다.
    """
    rows = []
    for d in details:
        model = d["model"]
        rank = next((i for i, t in enumerate(d["top5"], start=1) if t["is_gold"]), None)
        unit = unit_of.get(model)
        if unit is None:
            m = re.search(r"\[([^\[\]]+)\]", model)
            unit = m.group(1) if m else "-"
        rows.append(
            {
                "model": model,
                "검색": str(unit),
                "qid": d["qid"],
                "lang": d["qid"].split("-")[0],
                "question": d["question"],
                "n_gold": d["n_gold"],
                "gold_rank": rank,
                **{k: v for k, v in d.items() if k[:4] in ("Hit@", "MRR@", "nDCG")},
            }
        )
    return pd.DataFrame(rows)


# ── 사이드바 ──────────────────────────────────────────────────
st.sidebar.header("설정")
results_dir = st.sidebar.text_input("results 폴더", value=str(DATA_DIR / "results"))

rdir = Path(results_dir)
if not (rdir / "summary.csv").exists():
    st.markdown('<h1 class="main-title">임베딩 모델 비교</h1>', unsafe_allow_html=True)
    st.warning(f"`{rdir / 'summary.csv'}` 가 없습니다.")
    st.code(f"cd {DATA_DIR}\npython -m src.run_benchmark", language="bash")
    st.caption("먼저 벤치마크를 돌려 결과 파일을 만든 뒤 새로고침하세요.")
    st.stop()

summary_all, details, chunks_all = load_results(str(rdir), results_stamp(rdir))

# 모델 조건표용 config. results 폴더 옆이 아니라 공용 폴더 바로 밑에 있다.
cfg_path = DATA_DIR / "config.yaml"
cfg_stamp = cfg_path.stat().st_mtime if cfg_path.exists() else 0.0
qdf_all = build_question_frame(
    details, dict(zip(summary_all["model"], summary_all["검색"]))
)

# ── 검색 단위 선택 — 화면 전체가 여기에 잠긴다 ────────────────
units = sorted(summary_all["검색"].unique(), key=index_sort_key)
if "후보수" in summary_all.columns:
    n_cand = summary_all.groupby("검색")["후보수"].max().to_dict()
else:
    n_cand = {u: len(chunks_all.get(u, [])) for u in units}


def unit_caption(u: str) -> str:
    n = n_cand.get(u)
    return f"{u}  (후보 {int(n)}개)" if n else str(u)


unit = st.sidebar.radio(
    "검색 단위",
    units,
    format_func=unit_caption,
    help="후보 수가 달라 무작위 기준선부터 다르므로 검색 단위를 섞어서 비교하지 않습니다. "
         "한 번에 한 단위만 봅니다.",
)

summary = summary_all[summary_all["검색"] == unit].copy()
qdf = qdf_all[qdf_all["검색"] == unit].copy()
chunks = chunks_all.get(unit) or chunks_all.get("-") or []

hit_cols = sorted([c for c in summary.columns if c.startswith("Hit@")],
                  key=lambda c: int(c.split("@")[1]))
ks = [int(c.split("@")[1]) for c in hit_cols]
lo, hi = min(ks), max(ks)
sort_col = f"nDCG@{hi}" if f"nDCG@{hi}" in summary.columns else hit_cols[0]
order = summary.sort_values(sort_col, ascending=False)["model"].tolist()

picked = st.sidebar.multiselect("비교할 모델", order, default=order)
if not picked:
    st.warning("모델을 하나 이상 선택하세요.")
    st.stop()

summary = summary[summary["model"].isin(picked)].copy()
qdf = qdf[qdf["model"].isin(picked)].copy()
order = [m for m in order if m in picked]

if st.sidebar.button("결과 다시 읽기"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
theme_choice = st.sidebar.radio(
    "차트 테마",
    ["자동", "밝게", "어둡게"],
    horizontal=True,
    help="차트의 배경과 계열 색을 함께 바꿉니다. "
         "자동은 앱 테마(우상단 ⋮ → Settings)를 따라가고, "
         "밝게/어둡게는 앱 테마와 무관하게 차트만 고정합니다 — 보고서 스크린샷용.",
)
dark = (
    app_theme_is_dark() if theme_choice == "자동" else (theme_choice == "어둡게")
)
p = palette(dark)

n_questions = qdf["qid"].nunique()
n_units = int(n_cand.get(unit) or len(chunks))
baseline = 1 / n_units if n_units else 0.0

st.sidebar.divider()
st.sidebar.caption(
    f"[{unit}] · 질문 {n_questions}개 · 후보 {n_units}개 · 모델 {len(order)}개"
)


# ── 헤더 ──────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">임베딩 모델 검색 성능 비교</h1>',
            unsafe_allow_html=True)

best = summary.sort_values(sort_col, ascending=False).iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"1위 모델 [{unit}]", best["model"])
c2.metric(f"Hit@{lo}", f"{best[f'Hit@{lo}']:.3f}",
          help=f"상위 {lo}개 안에 정답이 있던 질문 비율")
c3.metric(f"정답@{lo}", f"{int(best[f'정답@{lo}'])} / {n_questions}")
c4.metric("인코딩 속도", f"{best['chunks_per_s']:,.0f} chunks/s")

st.caption(
    f"표 정렬 기준 **{sort_col}**."
)

tab_overview, tab_res1, tab_res2, tab_data, tab_detail = st.tabs(
    ["개요", "결과1", "결과2", "데이터셋", "상세"]
)


# ── 1. 개요 ───────────────────────────────────────────────────
with tab_overview:
    st.subheader(f"요약 — 검색 [{unit}]")
    st.dataframe(
        summary.set_index("model").drop(columns=["hf_id", "검색"], errors="ignore"),
        width="stretch",
    )
    st.caption(
        f"**MRR@{hi}** — 정답이 몇 등으로 올라왔는지의 역수를 평균한 값 "
        f"(1등 1.0 · 2등 0.5 · 3등 0.33 · {hi}등 밖 0)으로, 맞혔는지와 얼마나 위에 "
        f"올렸는지를 한 숫자로 묶은 지표.  \n"
        f"**nDCG@{hi}** — 정답을 상위에 몰아놓은 정도를 0~1 로 정규화한 값으로, "
        f"첫 정답만 보는 MRR 과 달리 정답이 여러 개일 때 나머지 정답의 위치까지 반영한다."
    )

    st.subheader("정확도")
    melted = summary.melt(
        id_vars="model", value_vars=hit_cols, var_name="지표", value_name="값"
    )
    bars = (
        alt.Chart(melted)
        .mark_bar(cornerRadiusEnd=4, size=20)   # 데이터 끝만 둥글게, 24px 이하
        .encode(
            y=alt.Y("model:N", sort=order, title=None),
            x=alt.X("값:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".0%")),
            # 색은 패널당 하나뿐이라 범례 없이 패널 제목이 지표를 말해 준다
            color=alt.Color(
                "지표:N", legend=None,
                scale=alt.Scale(domain=hit_cols, range=[p["series1"], p["series2"]]),
            ),
            tooltip=[alt.Tooltip("model:N", title="모델"),
                     alt.Tooltip("지표:N"),
                     alt.Tooltip("값:Q", format=".3f")],
        )
    )
    # 막대 끝에 값 — 축을 읽지 않아도 되게. 텍스트는 계열 색이 아닌 muted 잉크.
    labels = bars.mark_text(align="left", dx=6, color=p["muted"], fontSize=11).encode(
        text=alt.Text("값:Q", format=".3f"), color=alt.value(p["muted"])
    )
    st.altair_chart(
        themed(
            (bars + labels)
            .properties(height=42 * len(order) + 20)
            .facet(column=alt.Column(
                "지표:N", title=None,
                header=alt.Header(labelColor=p["muted"], labelFontSize=13))),
            p,
        ),
        width="stretch",
    )

    st.subheader("정확도 대비 비용")
    scat = summary.copy()
    base = alt.Chart(scat).encode(
        x=alt.X("chunks_per_s:Q", title="chunks/s (클수록 빠름)",
                scale=alt.Scale(zero=False, padding=20)),
        y=alt.Y(f"{sort_col}:Q", title=sort_col,
                scale=alt.Scale(zero=False, padding=20)),
        tooltip=[alt.Tooltip("model:N", title="모델"),
                 alt.Tooltip(f"{sort_col}:Q", format=".3f"),
                 alt.Tooltip("chunks_per_s:Q", format=",.0f", title="chunks/s"),
                 alt.Tooltip("dim:Q", title="차원"),
                 alt.Tooltip("index_MB:Q", title="인덱스 MB")],
    )
    dots = base.mark_circle(
        opacity=1, stroke=p["surface"], strokeWidth=2   # 겹칠 때를 대비한 표면색 링
    ).encode(
        size=alt.Size("index_MB:Q", legend=None, scale=alt.Scale(range=[120, 700])),
        color=alt.value(p["series1"]),                  # 단일 계열 → 범례 불필요
    )
    names = base.mark_text(align="left", dx=14, fontSize=11, color=p["muted"]).encode(
        text="model:N"
    )
    st.altair_chart(
        themed((dots + names).properties(height=380), p), width="stretch"
    )


# ── 2.결과1 ─────────────────────────
with tab_res1:
    st.subheader(f"언어별 정확도 — 검색 [{unit}]")
    st.caption(
        "질문은 전부 한국어이고 문서만 언어가 다릅니다. 즉 이 탭은 교차 언어 검색 성능입니다. "
    )

    # Hit 은 "맞혔나", MRR·nDCG 는 "얼마나 위에 올렸나"를 본다.
    # 셋 다 0~1 범위라 같은 히트맵 색 눈금을 그대로 쓸 수 있다.
    lang_metrics = hit_cols + [
        c for c in (f"MRR@{hi}", f"nDCG@{hi}") if c in qdf.columns
    ]
    lang_default = (lang_metrics.index(f"nDCG@{hi}")
                    if f"nDCG@{hi}" in lang_metrics else 0)
    klang = st.radio("기준", lang_metrics, horizontal=True, index=lang_default,
                     key="lang_k")
    is_hit = klang.startswith("Hit@")   # 이때만 "맞힌 문항 수"가 말이 된다

    lang_models = sorted(order, key=lang_model_key)
    lang_order = [l for l in LANG_LABEL if l in set(qdf["lang"])]
    label_order = [LANG_LABEL[l] for l in lang_order]

    g = qdf.groupby(["model", "lang"])[klang].agg(["mean", "size"]).reset_index()
    g.columns = ["model", "lang", "값", "문항수"]
    # Hit@k 는 0/1 이라 평균 x 문항수 = 맞힌 개수지만, MRR·nDCG 는 부분점수라
    # 같은 계산이 "맞힌 문항 수"를 뜻하지 않는다. 그래서 Hit 일 때만 만든다.
    if is_hit:
        g["맞힌수"] = (g["값"] * g["문항수"]).round().astype(int)
    g["언어"] = g["lang"].map(LANG_LABEL)

    # 램프가 테마에 따라 뒤집히므로 셀 글자색 조건도 함께 뒤집는다
    hi_ink = p["on_weak"] if dark else p["on_strong"]
    lo_ink = p["on_strong"] if dark else p["on_weak"]

    # 히트맵 맨 끝 평균 칸 — 표와 같은 매크로 평균(언어별 값의 합 / 언어 수).
    mean_rows = (
        g.groupby("model")
        .agg(**{"값": ("값", "mean"), "문항수": ("문항수", "sum"),
                **({"맞힌수": ("맞힌수", "sum")} if is_hit else {})})
        .reset_index()
        .assign(lang="__mean__", 언어="평균")
    )
    gh = pd.concat([g, mean_rows], ignore_index=True)
    heat_order = label_order + ["평균"]

    cells = (
        alt.Chart(gh)
        .mark_rect(stroke=p["surface"], strokeWidth=2)   # 셀 사이 2px 표면 간격
        .encode(
            x=alt.X("언어:N", sort=heat_order, title=None,
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y("model:N", sort=lang_models, title=None),
            color=alt.Color(
                "값:Q",
                scale=alt.Scale(range=p["seq"], domain=[0, 1]),
                legend=alt.Legend(title=klang, format=".0%", gradientLength=150),
            ),
            tooltip=[alt.Tooltip("model:N", title="모델"),
                     alt.Tooltip("언어:N"),
                     alt.Tooltip("값:Q", format=".3f", title=klang)]
                    + ([alt.Tooltip("맞힌수:Q", title="맞힌 문항")] if is_hit else [])
                    + [alt.Tooltip("문항수:Q", title="전체 문항")],
        )
    )
    cell_labels = cells.mark_text(fontSize=11).encode(
        text=alt.Text("값:Q", format=".2f"),
        color=alt.condition(alt.datum["값"] >= 0.5, alt.value(hi_ink), alt.value(lo_ink)),
    )
    st.altair_chart(
        themed((cells + cell_labels).properties(height=44 * len(lang_models) + 20), p),
        width="stretch",
    )

    meaning = (
        f"{klang} — 상위 {klang.split('@')[1]}개 안에 정답이 들어온 질문 비율"
        if is_hit else
        (f"{klang} — 정답이 몇 등으로 올라왔는지의 역수 평균 (1등 1.0 · 2등 0.5)"
         if klang.startswith("MRR@") else
         f"{klang} — 정답을 얼마나 위쪽에 몰아놨는지 (정답이 여러 개일 때도 반영)")
    )

    st.subheader("표로 보기")
    pivot = g.pivot(index="model", columns="언어", values="값")
    pivot = pivot.reindex(index=[m for m in lang_models if m in pivot.index],
                          columns=[c for c in label_order if c in pivot.columns])
    # 맨 끝 평균 열 — 화면에 보이는 언어 칸들의 단순 평균(언어당 가중치 동일).
    # 문항 수가 언어마다 달라서 전체 문항 평균(개요 탭 값)과는 일치하지 않는다.
    pivot["평균"] = pivot.mean(axis=1)
    counts = g.drop_duplicates("lang").set_index("언어")["문항수"]
    st.dataframe(
        pivot.style.format("{:.3f}"),
        width="stretch",
    )
    st.caption(
        "언어별 문항 수 — "
        + " · ".join(f"{c} {int(counts[c])}개" for c in pivot.columns if c in counts.index)
    )

    st.subheader("언어별 1위")
    winners = (
        g.loc[g.groupby("lang")["값"].idxmax()]
        .assign(순서=lambda x: x["lang"].map({l: i for i, l in enumerate(lang_order)}))
        .sort_values("순서")
    )
    st.dataframe(
        winners[["언어", "model", "값"] + (["맞힌수"] if is_hit else []) + ["문항수"]]
        .rename(columns={"model": "1위 모델", "값": klang})
        .set_index("언어")
        .style.format({klang: "{:.3f}"}),
        width="stretch",
    )

    st.subheader("모델 하나의 언어별 성적")
    st.caption("한 모델이 어떤 언어에서 강하고 어디서 무너지는지 한 줄로 봅니다.")
    mpick = st.selectbox("모델", lang_models, key="lang_model")
    one = g[g["model"] == mpick].sort_values("값", ascending=False)
    obars = (
        alt.Chart(one)
        .mark_bar(cornerRadiusEnd=4, size=20, color=p["series1"])
        .encode(
            y=alt.Y("언어:N", sort=one["언어"].tolist(), title=None),
            x=alt.X("값:Q", title=klang, scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".0%")),
            tooltip=[alt.Tooltip("언어:N"),
                     alt.Tooltip("값:Q", format=".3f", title=klang)]
                    + ([alt.Tooltip("맞힌수:Q", title="맞힌 문항")] if is_hit else [])
                    + [alt.Tooltip("문항수:Q", title="전체 문항")],
        )
    )
    olabels = obars.mark_text(align="left", dx=6, fontSize=11).encode(
        text=alt.Text("값:Q", format=".3f"), color=alt.value(p["muted"])
    )
    st.altair_chart(
        themed((obars + olabels).properties(height=36 * len(one) + 20), p),
        width="stretch",
    )



# ── 3.결과2 ─────────────────────────
def render_res2() -> None:
    """결과2 탭 본문.

    함수로 감싼 이유: 결과 폴더가 아직 없을 때 중간에 빠져나가야 하는데,
    탭 블록 안에서 st.stop() 을 부르면 뒤따르는 탭들까지 그리다 만다.
    """
    st.subheader(f"언어별 정확도 — 검색 [{unit}]")
    st.caption(
        "질문과 문서는 같은 언어"
    )

    # 이 탭만 results_lang 을 읽는다. 다른 탭은 사이드바에서 고른 results 폴더를
    # 그대로 쓰므로, 두 판을 한 화면에서 견줄 수 있다.
    if not (LANG_DIR / "summary.csv").exists():
        st.info(f"`{LANG_DIR / 'summary.csv'}` 가 없습니다.")
        st.code(
            f"cd {DATA_DIR}\npython -m src.run_benchmark "
            f"--questions questions/questions_lang.json --out results_lang",
            language="bash",
        )
        return

    summary2_all, details2, _chunks2 = load_results(
        str(LANG_DIR), results_stamp(LANG_DIR)
    )
    qdf2_all = build_question_frame(
        details2, dict(zip(summary2_all["model"], summary2_all["검색"]))
    )
    # 검색 단위와 비교 모델은 사이드바 선택을 그대로 따른다 — 두 판을 같은 조건에서
    # 나란히 놓기 위해서다. 언어판에 없는 모델은 조용히 빠진다.
    qdf2 = qdf2_all[(qdf2_all["검색"] == unit) & (qdf2_all["model"].isin(picked))]
    if qdf2.empty:
        st.info(f"`{LANG_DIR.name}` 에 검색 [{unit}] · 선택한 모델의 결과가 없습니다.")
        return

    order2 = [m for m in order if m in set(qdf2["model"])]
    hit_cols2 = sorted([c for c in qdf2.columns if c.startswith("Hit@")],
                       key=lambda c: int(c.split("@")[1]))
    hi2 = max(int(c.split("@")[1]) for c in hit_cols2)

    # Hit 은 "맞혔나", MRR·nDCG 는 "얼마나 위에 올렸나"를 본다.
    # 셋 다 0~1 범위라 같은 히트맵 색 눈금을 그대로 쓸 수 있다.
    lang_metrics2 = hit_cols2 + [
        c for c in (f"MRR@{hi2}", f"nDCG@{hi2}") if c in qdf2.columns
    ]
    lang_default2 = (lang_metrics2.index(f"nDCG@{hi2}")
                     if f"nDCG@{hi2}" in lang_metrics2 else 0)
    klang2 = st.radio("기준", lang_metrics2, horizontal=True, index=lang_default2,
                      key="lang_k2")
    is_hit2 = klang2.startswith("Hit@")   # 이때만 "맞힌 문항 수"가 말이 된다

    lang_models2 = sorted(order2, key=lang_model_key)
    lang_order2 = [l for l in LANG_LABEL if l in set(qdf2["lang"])]
    label_order2 = [LANG_LABEL[l] for l in lang_order2]

    g2 = qdf2.groupby(["model", "lang"])[klang2].agg(["mean", "size"]).reset_index()
    g2.columns = ["model", "lang", "값", "문항수"]
    # Hit@k 는 0/1 이라 평균 x 문항수 = 맞힌 개수지만, MRR·nDCG 는 부분점수라
    # 같은 계산이 "맞힌 문항 수"를 뜻하지 않는다. 그래서 Hit 일 때만 만든다.
    if is_hit2:
        g2["맞힌수"] = (g2["값"] * g2["문항수"]).round().astype(int)
    g2["언어"] = g2["lang"].map(LANG_LABEL)

    # 램프가 테마에 따라 뒤집히므로 셀 글자색 조건도 함께 뒤집는다
    hi_ink = p["on_weak"] if dark else p["on_strong"]
    lo_ink = p["on_strong"] if dark else p["on_weak"]

    # 히트맵 맨 끝 평균 칸 — 표와 같은 매크로 평균(언어별 값의 합 / 언어 수).
    mean_rows2 = (
        g2.groupby("model")
        .agg(**{"값": ("값", "mean"), "문항수": ("문항수", "sum"),
                **({"맞힌수": ("맞힌수", "sum")} if is_hit2 else {})})
        .reset_index()
        .assign(lang="__mean__", 언어="평균")
    )
    gh2 = pd.concat([g2, mean_rows2], ignore_index=True)
    heat_order2 = label_order2 + ["평균"]

    cells2 = (
        alt.Chart(gh2)
        .mark_rect(stroke=p["surface"], strokeWidth=2)   # 셀 사이 2px 표면 간격
        .encode(
            x=alt.X("언어:N", sort=heat_order2, title=None,
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y("model:N", sort=lang_models2, title=None),
            color=alt.Color(
                "값:Q",
                scale=alt.Scale(range=p["seq"], domain=[0, 1]),
                legend=alt.Legend(title=klang2, format=".0%", gradientLength=150),
            ),
            tooltip=[alt.Tooltip("model:N", title="모델"),
                     alt.Tooltip("언어:N"),
                     alt.Tooltip("값:Q", format=".3f", title=klang2)]
                    + ([alt.Tooltip("맞힌수:Q", title="맞힌 문항")] if is_hit2 else [])
                    + [alt.Tooltip("문항수:Q", title="전체 문항")],
        )
    )
    cell_labels2 = cells2.mark_text(fontSize=11).encode(
        text=alt.Text("값:Q", format=".2f"),
        color=alt.condition(alt.datum["값"] >= 0.5, alt.value(hi_ink), alt.value(lo_ink)),
    )
    st.altair_chart(
        themed((cells2 + cell_labels2).properties(height=44 * len(lang_models2) + 20), p),
        width="stretch",
    )

    st.subheader("표로 보기")
    pivot2 = g2.pivot(index="model", columns="언어", values="값")
    pivot2 = pivot2.reindex(index=[m for m in lang_models2 if m in pivot2.index],
                            columns=[c for c in label_order2 if c in pivot2.columns])
    # 맨 끝 평균 열 — 화면에 보이는 언어 칸들의 단순 평균(언어당 가중치 동일).
    # 문항 수가 언어마다 달라서 전체 문항 평균(개요 탭 값)과는 일치하지 않는다.
    pivot2["평균"] = pivot2.mean(axis=1)
    counts2 = g2.drop_duplicates("lang").set_index("언어")["문항수"]
    st.dataframe(
        pivot2.style.format("{:.3f}"),
        width="stretch",
    )
    st.caption(
        "언어별 문항 수 — "
        + " · ".join(f"{c} {int(counts2[c])}개"
                     for c in pivot2.columns if c in counts2.index)
    )

    st.subheader("언어별 1위")
    winners2 = (
        g2.loc[g2.groupby("lang")["값"].idxmax()]
        .assign(순서=lambda x: x["lang"].map({l: i for i, l in enumerate(lang_order2)}))
        .sort_values("순서")
    )
    st.dataframe(
        winners2[["언어", "model", "값"] + (["맞힌수"] if is_hit2 else []) + ["문항수"]]
        .rename(columns={"model": "1위 모델", "값": klang2})
        .set_index("언어")
        .style.format({klang2: "{:.3f}"}),
        width="stretch",
    )

    st.subheader("모델 하나의 언어별 성적")
    st.caption("한 모델이 어떤 언어에서 강하고 어디서 무너지는지 한 줄로 봅니다.")
    mpick2 = st.selectbox("모델", lang_models2, key="lang_model2")
    one2 = g2[g2["model"] == mpick2].sort_values("값", ascending=False)
    obars2 = (
        alt.Chart(one2)
        .mark_bar(cornerRadiusEnd=4, size=20, color=p["series1"])
        .encode(
            y=alt.Y("언어:N", sort=one2["언어"].tolist(), title=None),
            x=alt.X("값:Q", title=klang2, scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".0%")),
            tooltip=[alt.Tooltip("언어:N"),
                     alt.Tooltip("값:Q", format=".3f", title=klang2)]
                    + ([alt.Tooltip("맞힌수:Q", title="맞힌 문항")] if is_hit2 else [])
                    + [alt.Tooltip("문항수:Q", title="전체 문항")],
        )
    )
    olabels2 = obars2.mark_text(align="left", dx=6, fontSize=11).encode(
        text=alt.Text("값:Q", format=".3f"), color=alt.value(p["muted"])
    )
    st.altair_chart(
        themed((obars2 + olabels2).properties(height=36 * len(one2) + 20), p),
        width="stretch",
    )

with tab_res2:
    render_res2()



# ── 4. 데이터 ────────────────────────────────────────────
with tab_data:
    st.subheader("데이터")

    # 지금 단위의 후보 목록이 chunks.json 에 없을 수도 있다(결과가 옛 판본이면
    # 단위 이름이 어긋난다). 그때는 파일에 실제로 들어 있는 변형을 고르게 해서,
    # 아래 통계와 표가 전부 같은 목록 하나를 보게 한다.
    variants = [v for v in chunks_all if chunks_all.get(v)]
    if chunks:
        data_unit, data_chunks = unit, chunks
    elif variants:
        data_unit = st.selectbox(
            "후보 목록", sorted(variants, key=index_sort_key), key="data_unit"
        )
        data_chunks = chunks_all[data_unit]
    else:
        data_unit, data_chunks = unit, []

    if not data_chunks:
        st.info(f"`chunks.json` 에 [{unit}] 후보 목록이 없습니다.")
    else:
        cdf = pd.json_normalize(data_chunks)
        cdf["글자수"] = cdf["text"].str.len()
        lang_col = "meta.lang" if "meta.lang" in cdf.columns else None
        qmeta = qdf.drop_duplicates("qid")
        lang_rank = {LANG_LABEL[l]: i for i, l in enumerate(LANG_LABEL)}

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("문서", f"{cdf['source'].nunique():,}")
        m2.metric(f"후보 [{data_unit}]", f"{len(cdf):,}")
        m3.metric("질문", f"{n_questions:,}")
        m4.metric("글자", f"{int(cdf['글자수'].sum()):,}")

        st.subheader("언어별")
        rows = []
        for l in LANG_LABEL:
            sub = cdf[cdf[lang_col] == l] if lang_col else cdf.iloc[0:0]
            nq = int((qmeta["lang"] == l).sum())
            if sub.empty and not nq:
                continue
            rows.append({
                "언어": LANG_LABEL[l],
                "문서": int(sub["source"].nunique()),
                "후보": len(sub),
                "질문": nq,
                "평균 글자": int(sub["글자수"].mean()) if len(sub) else 0,
            })
        st.dataframe(pd.DataFrame(rows).set_index("언어"), width="stretch")

        st.subheader("문서별")
        doc = cdf.groupby("source").agg(후보=("id", "size"), 글자수=("글자수", "sum"))
        if lang_col:
            doc.insert(0, "언어", cdf.groupby("source")[lang_col].first().map(LANG_LABEL))
        if "meta.doc_type" in cdf.columns:
            doc.insert(1, "종류", cdf.groupby("source")["meta.doc_type"].first())
        doc.index.name = "문서"
        doc = doc.sort_index()
        if lang_col:
            # 언어 묶음 안에서는 문서명 순서를 유지해야 하므로 stable 정렬이다
            doc = (doc.assign(_o=doc["언어"].map(lang_rank))
                      .sort_values("_o", kind="stable").drop(columns="_o"))
        st.dataframe(doc, width="stretch")

        st.subheader("후보 길이 분포")
        hist = (
            alt.Chart(cdf[["글자수"]])
            .mark_bar(color=p["series1"])
            .encode(
                x=alt.X("글자수:Q", bin=alt.Bin(maxbins=40)),
                y=alt.Y("count():Q", title=None),
                tooltip=[alt.Tooltip("count():Q", title="후보 수")],
            )
        )
        st.altair_chart(themed(hist.properties(height=220), p), width="stretch")

        st.subheader("질문")
        qtab = qmeta[["qid", "lang", "question", "n_gold"]].copy()
        qtab["언어"] = qtab["lang"].map(LANG_LABEL)
        qtab = (qtab.assign(_o=qtab["언어"].map(lang_rank))
                    .sort_values(["_o", "qid"]).drop(columns=["_o", "lang"]))
        st.dataframe(
            qtab.rename(columns={"question": "질문", "n_gold": "정답 후보"})
                .set_index("qid"),
            width="stretch",
            height=420,
        )

        st.subheader("후보 탐색")
        f0, f1, f2, f3 = st.columns(4)
        if lang_col:
            langs = [l for l in LANG_LABEL if l in set(cdf[lang_col].dropna())]
            lang_pick = f0.selectbox(
                "언어", ["전체"] + langs, key="data_lang",
                format_func=lambda l: l if l == "전체" else f"{LANG_LABEL[l]} ({l})",
            )
        else:
            lang_pick = "전체"
        pool = cdf if lang_pick == "전체" else cdf[cdf[lang_col] == lang_pick]
        src = f1.selectbox("문서", ["전체"] + sorted(pool["source"].unique()),
                           key="data_src")
        if "meta.doc_type" in cdf.columns:
            dtypes = sorted(x for x in cdf["meta.doc_type"].dropna().unique())
            dtype = f2.selectbox("종류", ["전체"] + dtypes, key="data_type")
        else:
            dtype = "전체"
        if "meta.섹션" in cdf.columns:
            secs = sorted(x for x in cdf["meta.섹션"].dropna().unique())
            sec = f3.selectbox("섹션", ["전체"] + secs, key="data_sec")
        else:
            sec = "전체"

        view = pool
        if src != "전체":
            view = view[view["source"] == src]
        if dtype != "전체":
            view = view[view["meta.doc_type"] == dtype]
        if sec != "전체":
            view = view[view["meta.섹션"] == sec]

        cols = [c for c in ["id", "source", "locator", "meta.섹션", "글자수", "text"]
                if c in view.columns]
        st.dataframe(
            view[cols].rename(columns={"meta.섹션": "섹션", "text": "본문",
                                       "source": "문서"}).set_index("id"),
            width="stretch",
            height=520,
        )





# ── 5. 상세 ──────────────────────────────────

qmap = qdf.drop_duplicates("qid").set_index("qid")["question"].to_dict()

with tab_detail:
    st.subheader(f"질문 × 모델 — 검색 [{unit}]")
    st.caption(
        "칸의 숫자는 **정답이 몇 등으로 올라왔는지**입니다. "
        "`1` 이면 1등으로 맞힌 것, `—` 는 상위 5개 안에 못 넣은 것입니다."
    )

    lang_filter = st.multiselect(
        "언어로 좁히기", [LANG_LABEL[l] for l in LANG_LABEL if l in set(qdf["lang"])],
        default=[], help="비우면 전체 언어를 봅니다.",
    )
    picked_langs = {l for l, lab in LANG_LABEL.items() if lab in lang_filter}
    qview = qdf[qdf["lang"].isin(picked_langs)] if picked_langs else qdf

    matrix = qview.pivot(index="qid", columns="model", values="gold_rank")
    matrix = matrix[[m for m in order if m in matrix.columns]]

    def tint(v):
        if pd.isna(v):
            return f"background-color: {MISS_TINT}"
        return f"background-color: {RANK_TINT.get(int(v), MISS_TINT)}"

    st.dataframe(
        matrix.style.map(tint).format(lambda v: "—" if pd.isna(v) else f"{int(v)}"),
        width="stretch",
        height=min(600, 38 * len(matrix) + 40),
    )

    st.divider()
    st.subheader("질문 하나 뜯어보기")

    qid = st.selectbox(
        "질문", list(qmap), format_func=lambda q: f"{q} — {qmap.get(q, '')}"
    )

    st.info(qmap[qid])
    # 검색 단위를 섞지 않도록 지금 보고 있는 단위의 모델만 편다
    detail_by_model = {
        d["model"]: d for d in details
        if d["qid"] == qid and d["model"] in set(order)
    }

    for m in order:
        d = detail_by_model.get(m)
        if not d:
            continue
        rank = next((i for i, t in enumerate(d["top5"], start=1) if t["is_gold"]), None)
        badge = f"{rank}등" if rank else "실패"
        with st.expander(f"**{m}** — 정답 {badge}", expanded=(m == order[0])):
            rows = []
            for i, t in enumerate(d["top5"], start=1):
                rows.append(
                    {
                        "순위": i,
                        "정답": "●" if t["is_gold"] else "",
                        "유사도": round(t["score"], 4),
                        "문서": t["source"],
                        "위치": t["locator"],
                        "섹션": t.get("섹션") or "",
                        # gold 청크는 정답 근거 문구 주변을, 나머지는 앞부분을 보여준다
                        "내용": t.get("match") or t["preview"],
                    }
                )
            st.dataframe(pd.DataFrame(rows).set_index("순위"),
                         width="stretch")


    k = st.radio("기준", hit_cols, horizontal=True, index=0)

    miss = qdf[qdf[k] == 0]
    per_model = (
        miss.groupby("model").size().reindex(order, fill_value=0)
        .rename("틀린 문제 수").reset_index()
    )

    st.subheader(f"모델별 {k} 오답 수 — 검색 [{unit}]")
    mbars = (
        alt.Chart(per_model)
        .mark_bar(cornerRadiusEnd=4, size=20, color=p["series1"])
        .encode(
            y=alt.Y("model:N", sort=order, title=None),
            x=alt.X("틀린 문제 수:Q", title=None,
                    scale=alt.Scale(domain=[0, n_questions])),
            tooltip=["model:N", "틀린 문제 수:Q"],
        )
    )
    mlabels = mbars.mark_text(align="left", dx=6, fontSize=11).encode(
        text=alt.Text("틀린 문제 수:Q"), color=alt.value(p["muted"])
    )
    st.altair_chart(
        themed((mbars + mlabels).properties(height=42 * len(order) + 20), p),
        width="stretch",
    )

    st.subheader(f"모든 모델이 {k} 에서 틀린 문제")
    common = (
        miss.groupby("qid").size().pipe(lambda s: s[s == len(order)]).index.tolist()
    )
    if not common:
        st.success("없습니다. 모든 문제를 최소 한 모델은 맞혔습니다.")
    else:
        st.caption(
            f"{len(common)}개. **모델 성능 문제가 아닐 가능성이 높습니다** — "
            "질문 문장이나 `must_include` 라벨을 먼저 점검하세요."
        )
        st.dataframe(
            qdf[qdf["qid"].isin(common)][["qid", "question", "n_gold"]]
            .drop_duplicates("qid").set_index("qid"),
            width="stretch",
        )

    st.subheader(f"언어별 {k} 오답 수")
    st.caption(
        "오답이 어느 언어에서 나오는지. 괄호 안은 그 언어의 전체 문항 수입니다 — "
        "문항 수가 언어마다 다르므로 개수만 보고 판단하면 안 됩니다 "
        "(비율은 **언어별** 탭에서 봅니다)."
    )
    totals = qdf.groupby("lang")["qid"].nunique()
    lang_cols = [l for l in LANG_LABEL if l in totals.index]
    mtab = (
        miss.groupby(["model", "lang"]).size().rename("틀린 수").reset_index()
        .pivot(index="model", columns="lang", values="틀린 수")
        .reindex(index=order, columns=lang_cols)
        .fillna(0).astype(int)
    )
    mtab.columns = [f"{LANG_LABEL[l]} ({int(totals[l])})" for l in lang_cols]
    st.dataframe(
        mtab.style.background_gradient(cmap="Reds", axis=None).format("{:d}"),
        width="stretch",
    )

    st.subheader("문제별 난이도")
    st.caption(f"{k} 기준으로 몇 개 모델이 틀렸는지. 위쪽이 어려운 문제입니다.")
    hardness = (
        miss.groupby("qid").size().rename("틀린 모델 수")
        .reset_index().sort_values("틀린 모델 수", ascending=False)
    )
    if hardness.empty:
        st.caption("오답이 없습니다.")
    else:
        hardness["언어"] = hardness["qid"].str.split("-").str[0].map(LANG_LABEL)
        hardness["question"] = hardness["qid"].map(qmap)
        st.dataframe(hardness.set_index("qid"), width="stretch")
