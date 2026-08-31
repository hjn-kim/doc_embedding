# 다국어 임베딩 모델 검색 성능 비교

7개 언어 **51개 법률·수사 문서**를 대상으로 6개 임베딩 모델의 **검색(Retrieval) 정확도**를
비교합니다. 질문은 150문항이고, **한국어 질의**와 **원문 언어 질의** 두 세트로 각각 돌려
cross-lingual 검색 능력까지 함께 잽니다.

| 모델 (`key`) | HF ID | 차원 | 컨텍스트 | prefix 규칙 |
|---|---|---|---|---|
| `kure-v1` | `nlpai-lab/KURE-v1` | 1024 | 8194 | 없음 |
| `bge-m3` | `BAAI/bge-m3` | 1024 | 8192 | 없음 (dense + sparse 하이브리드도 함께 측정) |
| `e5-large-instruct` | `intfloat/multilingual-e5-large-instruct` | 1024 | 512 | `Instruct:...\nQuery:` |
| `e5-small-ko` | `dragonkue/multilingual-e5-small-ko` | 384 | 512 | `query: ` / `passage: ` |
| `harrier-0.6b` | `microsoft/harrier-oss-v1-0.6b` | 1024 | 32768 | `Instruct:...\nQuery:` |
| `harrier-270m` | `microsoft/harrier-oss-v1-270m` | 640 | 32768 | `Instruct:...\nQuery:` |

모델 목록·차원·prefix 는 전부 `config.yaml` 의 `models` 에서 옵니다. 위 표는 그 사본입니다.

## 코퍼스

`data/<언어>/*.pdf` 51개. 채점 대상 텍스트는 PDF 가 아니라 **사람이 검수한 `data/gt/<언어>/*.txt`**
입니다 (`chunking.source: gt`). OCR·추출 오류가 임베딩 점수에 섞이지 않게 하려는 것입니다.

| 언어 | 문서 | 510토큰 청크 | 문서당 청크(중앙값) | 질문 |
|---|---|---|---|---|
| `ko` | 10 | 71 | 7 | 30 |
| `en` | 10 | 30 | 3 | 20 |
| `ch` | 7 | 34 | 4 | 20 |
| `pil` | 7 | 26 | 3 | 20 |
| `ru` | 7 | 25 | 3 | 20 |
| `uz` | 5 | 34 | 6 | 20 |
| `vn` | 5 | 44 | 9 | 20 |
| **합계** | **51** | **264** | 5 | **150** |

## 평가 방식

임베딩 모델은 답변을 생성하지 않고 **관련 청크를 찾아오는** 역할만 합니다.
따라서 "성능"은 생성 품질이 아니라 검색 정확도로 측정합니다.

```
GT 텍스트 → 공통 청킹(모든 모델 동일) → 모델별 임베딩 → 질문과 코사인 유사도 → 랭킹 → 지표
```

### 청킹 단위는 글자가 아니라 토큰

문자 밀도가 언어마다 다릅니다. bge-m3 토크나이저 기준 중국어는 1.42자/토큰, 러시아어는
3.96자/토큰이라 **같은 500자라도 중국어 청크가 러시아어 청크보다 2.8배 큽니다.** 그러면
"ch 점수가 높다"가 모델 실력인지 청크가 커서인지 구분이 안 됩니다.

그래서 `chunking.unit: token` 으로 자르고, 길이를 재는 **기준 토크나이저 하나(`BAAI/bge-m3`)를
모든 모델에 공통 적용**합니다. 모델별 토크나이저로 자르면 모델마다 청크 경계가 달라져
"모든 모델이 똑같은 청크를 쓴다"는 전제가 깨집니다.

`chunk_size` 는 overlap 을 포함한 **최종** 청크 상한입니다. 510 인 이유는 e5 계열 상한이
512 인데 XLM-R 이 특수토큰 2개를 붙이기 때문입니다. overlap 은 100 토큰입니다.

### 색인 단위 두 가지 (`index.variants`)

| 변형 | 색인 단위 | 후보 수 | 대상 |
|---|---|---|---|
| `512` | 510토큰 청크 | **264** | 전 모델 (6개, bge-m3 는 dense/hybrid 2행) |
| `full` | 문서 1개 = 벡터 1개 | **51** | 컨텍스트 상한 1024 이상인 모델만 (4개) |

**지표는 그 행의 색인 단위에서 그대로 잽니다.** 후보 수가 달라 무작위 기준선부터 다르므로
(1/264 vs 1/51) **512 행끼리, full 행끼리만 비교하세요.** 요약표도 색인별로 묶여 나옵니다.
`bge-m3 [512]` 와 `bge-m3 [full]` 의 숫자를 가로로 빼서 비교하면 안 됩니다.

`full` 에서 상한 512 모델(`e5-large-instruct`, `e5-small-ko`)을 빼는 이유: 문서 51개 중 49개가
510토큰을 넘고 중앙값이 청크 5개 분량이라, 512 로 자르면 코퍼스 대부분이 잘려나갑니다.
그러면 임베딩 품질이 아니라 컨텍스트 길이를 재게 됩니다. `min_context_for_full`(1024) 미만인
모델은 자동으로 제외됩니다.

### 질문 세트 두 가지

같은 150문항을 **질문 문장의 언어만 바꿔** 두 벌 준비했습니다. `id`·`source`·`gold_chunks` 는
완전히 같으므로 두 결과를 나란히 놓고 빼면 그 차이가 곧 cross-lingual 손실입니다.

| 파일 | 질문 언어 | 재는 것 | 출력 |
|---|---|---|---|
| `questions/questions.json` | **한국어** (고유명사만 원문 표기) | 한국어로 물어 외국어 문서를 찾는 능력 | `results/` |
| `questions/questions_lang.json` | **문서와 같은 언어** | 언어 장벽이 없을 때의 상한선 | `results_lang/` |

```
vn-01  questions.json      Hội thẩm nhân dân Bùi Thị Kim Thủy·Dương Thị Được 와 함께
                           Võ Minh P 를 심리한 Thẩm phán – Chủ tọa phiên tòa 를 찾아주세요
vn-01  questions_lang.json Hãy tìm Thẩm phán – Chủ tọa phiên tòa đã xét xử Võ Minh P
                           cùng với các Hội thẩm nhân dân Bùi Thị Kim Thủy, Dương Thị Được
```

한국어 문서 30문항은 두 파일이 동일합니다 (바꿀 언어가 없으므로).

### 정답 라벨

현재 150문항은 **`gold_chunks` 에 정답 청크 id 를 직접 박아** 두었습니다. `note` 에 그 근거를
적어 둡니다.

```json
{
  "id": "ko-01",
  "question": "형사2부장 이성희가 이끈 정부합동 의약품 리베이트 수사단이 적발·기소·행정처분 의뢰한 인원을 찾아주세요",
  "source": "ko/140801_서부지검_보도자료.pdf",
  "gold_chunks": [64, 65],
  "note": "근거: 227명 적발, 46명 기소[1명 구속], 222명 행정처분 의뢰"
}
```

`gold.py` 는 문구 기반 라벨링(`must_include` / `any_include` / `must_exclude`)도 그대로
지원합니다. 둘을 같이 적으면 **`gold_chunks` 가 이깁니다.**

| 방식 | 장점 | 단점 |
|---|---|---|
| `gold_chunks` (현재) | 문구 오타로 조용히 어긋날 일이 없다 | **청킹 설정을 바꾸면 id 가 밀려 전부 다시 잡아야 한다** |
| `must_include` | 청킹을 바꿔도 라벨이 따라온다 | 원문과 한 글자만 달라도 영영 못 맞힌다 |

> `chunk_size` / `chunk_overlap` / `chunking.source` 를 건드릴 계획이라면 `must_include` 로
> 옮기는 편이 낫습니다. 지금 설정 그대로 쓸 거라면 `gold_chunks` 가 안전합니다.

### 측정 지표

| 지표 | 의미 |
|---|---|
| **Hit@1 / Hit@3** | 상위 k개 안에 정답이 하나라도 있었는가. RAG 체감 성능에 가장 직결 |
| **정답@1 / 정답@3** | 같은 값을 비율이 아니라 실제 맞힌 문제 개수로 센 것 (150 중 몇 개) |
| **MRR@3** | 첫 정답의 순위 역수 (1위=1.0, 2위=0.5, 3위=0.33) |
| **nDCG@3** | 정답을 얼마나 위쪽에 몰아놨는가 (최종 정렬 기준) |
| VRAM_MB / index_MB | 인코딩 중 최대 GPU 메모리 / 벡터 인덱스 용량 |
| chunks_per_s / query_ms | 인덱싱·검색 속도 (워밍업 후 측정) |

Precision·Recall·F1 은 쓰지 않습니다. 질문당 정답 청크가 1~2개라 `Recall@k` 는 `Hit@k` 와
사실상 같고, `Precision@k` 는 상한이 1/k 라 k 를 키우면 오히려 내려가서 @1 과 @3 을
나란히 놓고 읽을 수 없기 때문입니다.

## 실행 순서

### 1. 준비 (로컬)

```bash
data/<언어>/문서.pdf        # 비교 대상 원본
data/gt/<언어>/문서.txt     # 검수한 정답 텍스트 (채점·청킹은 이쪽을 쓴다)
```

GT 초안이 없으면 PDF 텍스트 레이어에서 뽑아 쓸 수 있습니다. **뽑은 뒤 반드시 눈으로
고치세요** — 표·2단 편집은 읽는 순서와 다르게 나오고 머리말·쪽번호가 섞여 들어옵니다.

```bash
python -m src.make_gt          # data/gt/ 에 초안 저장
python -m src.paginate_gt      # 쪽 경계 표시가 필요할 때
```

### 2. RunPod 세팅

RunPod PyTorch 템플릿(CUDA 12.x, GPU 16GB 이상 권장 — BGE-M3 fp16 기준 약 5GB)에서:

```bash
git clone <your-repo-url> && cd <repo>

# 모델 캐시를 네트워크 볼륨에 두면 파드 재시작해도 재다운로드가 없다
export HF_HOME=/workspace/hf_cache

pip install -r requirements.txt
python -m src.download_models     # 모델 6개 미리 받기
```

> torch 버전 충돌이 나면 `requirements.txt` 의 `torch` 줄을 주석 처리하세요
> (RunPod 템플릿에 이미 설치돼 있습니다).

### 3. 청크 확인 후 질문 작성 ← **여기가 제일 중요합니다**

```bash
python -m src.inspect_chunks --limit 50
python -m src.inspect_chunks --grep 계약   # 특정 키워드 주변 확인
```

`results/chunks_preview.txt` 가 생깁니다 (커밋하지 않는 임시 산출물). 그걸 보면서
`questions/questions.json` 을 채웁니다.

질문 작성 규칙:

- **질문은 대상 문서를 유일하게 지목하는 앵커를 반드시 포함해야 합니다.** 사건번호
  (`2015고단7004`, `Bản án 132/2020/HS-ST`), 인명(`曾庆芬`), 기관(`Прокуратура Камызякского района`),
  고유 금액 중 하나입니다. `scope: all` 이라 51개 문서가 한 인덱스에 들어가므로,
  "피고인에게 선고된 형량은?" 같은 질문은 판결문 어느 것에나 해당해서 정답이 없습니다.
- **"이 사건", "이 문서" 같은 지시어를 쓰지 마세요.** 무엇을 가리키는지 알 수 없습니다.
- **고유명사는 원문 언어 표기를 쓰세요.** `쩡칭펀`이 아니라 `曾庆芬`. `questions.json` 은
  질문 문장 자체는 한국어로 두되 앵커만 원문 표기로 남깁니다. 그래야 한국어 질의로
  외국어 문서를 찾는 cross-lingual 검색 능력을 재게 됩니다.
- **원문 문장을 그대로 베끼지 마세요.** 베끼면 어휘만 겹쳐서 모든 모델이 다 맞히고
  변별력이 사라집니다. 실제 사용자가 쓸 법한 구어체·동의어로 바꿔 쓰세요.
- `source` 는 `data/` 기준 상대경로 그대로입니다 (`ko/140801_서부지검_보도자료.pdf`).
  `gold.py` 가 정확히 일치 비교라 오타는 에러가 아니라 조용히 어긋납니다.
- **문항 수는 문서 길이에 비례**시키세요. 청크가 3개뿐인 문서에 10문항을 만들면 같은 청크가
  정답인 질문이 여러 개 생겨서, 문서 이해도가 아니라 한 문단을 반복해 물은 것이 됩니다.
- 전체 순위에는 150개 이상이 필요하고, 언어별 순위까지 보려면 **언어당 20문항이 하한**입니다.
  현재 배분(ko 30 / 나머지 20씩)은 언어별 비교가 가능한 최소선이라, 0.05 이내 차이는
  언어별로 갈라 읽지 마세요.
- `questions_lang.json` 은 같은 문항을 문서 언어로 옮긴 것입니다. **`id`·`source`·`gold_chunks`
  를 반드시 그대로 유지**하세요. 하나라도 어긋나면 두 결과를 뺄 수 없습니다.

### 3-1. 질문 검증 ← GPU 올리기 전에 반드시

```bash
python -m src.check_questions
```

로컬 CPU 로 몇 초면 끝납니다. 잡아내는 것:

| 검사 | 왜 필요한가 |
|---|---|
| gold 없음 | 정답 청크를 하나도 못 잡으면 그 문항은 영영 못 맞힙니다 |
| gold 과다 | 조건이 헐렁해 문서 절반이 정답이면 아무 모델이나 맞힙니다 |
| 앵커 누수 | 문구가 다른 문서에도 있으면 정답이 여러 문서로 번집니다 |
| 지시어 | "이 사건" 처럼 대상을 특정 못 하는 질문 |
| 앵커 없음 | 질문 안에 대상 문서를 변별하는 단어가 하나도 없는 경우 |
| 청크 중복 | 같은 문서 안 두 질문이 같은 청크만 가리키는 경우 |
| 배분 이탈 | 문서별 문항 수가 권장치에서 벗어난 경우 |

실행 시 `[경고] 정답 청크를 못 찾은 질문` 이 뜨면 `gold_chunks` id 가 현재 청킹과
어긋난 것입니다 (청킹 설정을 바꿨다면 id 를 다시 잡아야 합니다).

### 4. 벤치마크 실행

**두 세트를 모두 돌려야** 대시보드의 결과1/결과2 탭이 채워집니다.

```bash
# 결과1 — 한국어 질의
python -m src.run_benchmark

# 결과2 — 원문 언어 질의
python -m src.run_benchmark --questions questions/questions_lang.json --out results_lang

# 특정 모델만
python -m src.run_benchmark --models kure-v1 bge-m3

# 특정 모델만 다시 돌리되 나머지 모델의 이전 결과는 살려두기
python -m src.run_benchmark --models harrier-0.6b --resume
```

전체 옵션은 `--config` `--models` `--questions` `--out` `--resume` 다섯 개입니다.

출력 폴더는 매 실행마다 통째로 덮어써집니다. `--models` 로 일부만 재실행하면
나머지 모델의 행이 사라지므로, 이어붙이려면 `--resume` 을 함께 주세요.
질문 집합이나 청크 수가 이전 실행과 다르면 `--resume` 은 거부합니다 —
서로 다른 조건의 점수가 한 표에 섞이면 비교가 깨지기 때문입니다.

한 런이 실패해도 나머지는 계속 돌지만, **실패한 런은 요약표에 행이 아예 생기지
않습니다.** 실행 마지막의 `── 실패한 런` 목록을 꼭 확인하세요. 특히 CUDA
device-side assert 는 한 번 터지면 프로세스의 CUDA 컨텍스트를 오염시켜 뒤의 모든
런이 같은 에러로 죽습니다 — 이때는 **첫 번째 실패만이 진짜 원인**입니다.

### 5. 출력

`results/` 와 `results_lang/` 에 각각 다음이 생깁니다.

| 파일 | 내용 |
|---|---|
| `summary.csv` / `summary.md` | 모델 × 색인 단위 지표 요약표 (12행) |
| `details.json` | 질문별 상위 검색 결과 — **왜 틀렸는지 확인용** |
| `chunks.json` | 사용된 청크 전체 (`512`, `full` 두 키) |
| `misses.md` | 틀린 문항만 추린 목록 |
| `chunks_preview.txt` | `inspect_chunks` 산출물. 커밋하지 않음 |

## 6. 결과 보기 (대시보드)

`2-4_document_embedding.py` 는 사내 통합 Streamlit 앱의 한 페이지입니다. 단독 실행용이
아니라 **`app/pages/` 에 두고 데이터는 `app/document_lab/embedding/` 에 모으는** 배치를
전제합니다.

```
app/
  pages/2-4_document_embedding.py
  shared_utils.py                 # apply_common_styles 를 여기서 가져온다
  document_lab/embedding/
    config.yaml
    results/                      # 결과1 (한국어 질의)
    results_lang/                 # 결과2 (원문 언어 질의)
```

경로는 `DATA_DIR = <파일위치>/../document_lab/embedding` 로 고정돼 있습니다 (환경변수
오버라이드 없음). 사이드바의 "results 폴더" 입력칸으로 결과1 폴더만 바꿔 볼 수 있고,
결과2 탭은 항상 `DATA_DIR/results_lang` 을 읽습니다.

| 탭 | 보는 것 |
|---|---|
| 개요 | 모델별 Hit/nDCG 순위, 정확도 대비 비용 |
| 결과1 | 한국어 질의 결과 상세 |
| 결과2 | 원문 언어 질의 결과, 결과1 과의 차이 |
| 데이터셋 | 문서·청크·질문 분포 |
| 상세 | 질문별 검색 결과와 오답 |

## 튜닝 포인트 (`config.yaml`)

| 항목 | 설명 |
|---|---|
| `chunking.source` | `gt` = 검수 텍스트로 청킹(현재). `data` 로 바꾸면 PDF 추출 오류가 점수에 섞입니다 |
| `chunking.chunk_size` | 토큰 기준. 192 / 384 / 510 으로 바꿔가며 돌려보세요. 최적값은 문서 성격에 따라 다르고, **모델 순위 자체가 바뀌기도 합니다.** 단 `gold_chunks` id 가 전부 밀립니다 |
| `chunking.unit` / `tokenizer` | `token` 권장. `char` 로 되돌리면 언어 간 비교가 깨집니다 |
| `index.variants` | `[512, full]`. 청크 검색만 볼 거면 `[512]` 로 줄이면 런 수가 절반이 됩니다 |
| `index.min_context_for_full` | 1024. 이 미만인 모델은 `full` 에서 자동 제외됩니다 |
| `retrieval.scope` | `all` = 51개 문서를 한 인덱스에서 검색(어려움, 실전에 가까움) / `own_doc` = 문서 내부만 |
| `runtime.batch_size` | CUDA OOM 이면 16 → 8 로 |
| `runtime.max_seq_length` | 1024. BGE-M3 장문 성능을 보려면 4096~8192 로 (VRAM 많이 씀) |
| `models[].hybrid` | BGE-M3 의 sparse 가중치 (현재 dense 1.0 / sparse 0.3). dense 단독 대비 얼마나 오르는지 비교됩니다 |

## 현재 결과 요약

`nDCG@3` 기준, 색인 단위별 상위권만 옮긴 것입니다. 전체는 `summary.md` 를 보세요.

**[512] 264개 청크 중에서 찾기**

| 모델 | nDCG@3 (한국어 질의) | nDCG@3 (원문 질의) | 차이 |
|---|---|---|---|
| kure-v1 | **0.920** | 0.914 | −0.006 |
| bge-m3 (hybrid) | 0.901 | **0.919** | +0.018 |
| bge-m3 (dense) | 0.875 | 0.876 | +0.001 |
| e5-small-ko | 0.822 | 0.890 | +0.068 |
| harrier-270m | 0.649 | 0.828 | +0.179 |
| harrier-0.6b | 0.598 | 0.817 | +0.219 |
| e5-large-instruct | 0.279 | 0.807 | **+0.528** |

**[full] 51개 문서 중에서 찾기**

| 모델 | nDCG@3 (한국어 질의) | nDCG@3 (원문 질의) |
|---|---|---|
| bge-m3 (hybrid) | **0.947** | 0.961 |
| kure-v1 | 0.936 | 0.945 |
| bge-m3 (dense) | 0.926 | 0.933 |
| harrier-270m | 0.834 | 0.929 |
| harrier-0.6b | 0.796 | **0.974** |

읽는 법:

- **한국어로 물을 거라면 KURE-v1 또는 BGE-M3 hybrid 입니다.** 두 모델만 두 세트에서
  점수가 거의 같습니다. 즉 질의 언어가 바뀌어도 성능이 흔들리지 않습니다.
- **`e5-large-instruct` 의 0.279 → 0.807 이 이 벤치마크의 핵심 발견입니다.** 이 모델은
  문서를 못 읽는 게 아니라 **한국어 질의를 외국어 문서에 연결하지 못합니다.**
  전체 평균만 보면 "성능이 나쁜 모델"로 오해하게 됩니다. harrier 두 모델도 정도는
  덜하지만 같은 성격(+0.18~0.22)입니다.
- **harrier-0.6b 는 `full` 원문 질의에서 1위(0.974)인데 `512` 한국어 질의에서는 꼴찌권(0.598)
  입니다.** 32K 컨텍스트를 통째로 먹일 때만 강합니다. 색인 단위를 가로질러 비교하면
  안 되는 이유가 이 행에 다 들어 있습니다.
- **속도 대비 정확도:** `e5-small-ko` 는 384차원에 인덱스가 0.39MB(1/3), 인코딩이
  542 chunks/s(3.5배)인데 원문 질의 nDCG 는 0.890 으로 1위와 0.03 차이입니다.
  서비스 조건에 따라 이쪽이 정답일 수 있습니다.

## 결과 해석 시 주의

- 150문항은 전체 순위를 가리기엔 충분하지만 **언어별로 쪼개면 언어당 20문항**입니다.
  언어별 표에서 0.05 이내 차이는 노이즈로 보세요.
- 차원이 큰 모델이 항상 이기지 않습니다. `index_MB` 와 `chunks_per_s` 를 함께 보고
  **정확도 대비 비용**으로 판단하세요.
- **`512` 행과 `full` 행은 서로 다른 크기의 건초더미에서 바늘을 찾은 결과입니다.** full 쪽
  숫자가 높게 나오는 건 후보가 51개뿐이어서이지 모델이 더 좋아서가 아닙니다.
- 실제 RAG 품질은 리랭커(BGE-reranker 등)를 얹으면 또 달라집니다. 이 벤치마크는
  1차 검색기(retriever) 성능만 측정합니다.

## 구조

```
config.yaml                       설정 (모델 목록, 청킹, 색인, 런타임)
requirements.txt
2-4_document_embedding.py         결과 대시보드 (통합 앱의 app/pages/ 에 배치)

questions/
  questions.json                  150문항 · 한국어 질의   → results/
  questions_lang.json             150문항 · 원문 언어 질의 → results_lang/
data/
  <언어>/*.pdf                    원본 문서 51개 (ch en ko pil ru uz vn)
  gt/<언어>/*.txt                 검수한 정답 텍스트 (청킹·채점은 이쪽을 쓴다)
results/ results_lang/            summary.csv|md, details.json, chunks.json, misses.md

src/
  loaders.py             docx/pdf 텍스트 추출
  metadata.py            문서종류·기관·작성일·제목·섹션 추출 (규칙 기반, 기록 전용)
  chunker.py             공통 청킹
  gold.py                질문 로드 + 정답 청크 라벨링 (gold_chunks / 문구 매칭)
  models.py              모델별 prefix 규칙 + 인코더 래퍼
  metrics.py             Hit/MRR/nDCG
  run_benchmark.py       메인 실행
  check_questions.py     질문 검증 (GPU 올리기 전 필수)
  inspect_chunks.py      청크 확인 헬퍼
  download_models.py     HF 사전 다운로드
  make_gt.py             PDF 텍스트 레이어로 GT 초안 만들기 (1회성)
  paginate_gt.py         GT 에 쪽 경계 표시 (1회성)
```
