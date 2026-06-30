# RAG Starter — 프로젝트 전체 개요

> 미국 항공 규정(14 CFR)에 대한 **Agentic RAG 챗봇**.
> 고정 크기 청크 유사도 검색 대신, 모델이 규정의 **구조(목차 → 섹션 → 상호참조)**를 직접 탐색하며 답하고 인용을 답니다.
> 브랜치: `feat/react-toc-retrieval`

---

## 1. 한눈에 보기

- **무엇**: "자가용 조종사 최소 비행시간은?" 같은 항공 규정 질문에, 실제로 읽은 섹션 번호를 인용해 답하는 챗봇
- **어떻게**: ReAct 에이전트가 3개 도구(`navigate_toc` / `read_section` / `follow_refs`)로 규정을 사람처럼 탐색
- **비용 최적화**: **Adaptive RAG**(질문 복잡도에 따른 라우팅) + **프롬프트 캐싱**으로 토큰 절감
- **UX**: 실시간 스트리밍 출력, 현재 작업 단계 표시, 토큰·도구 사용량 텔레메트리

### 기술 스택
| 영역 | 사용 |
|---|---|
| 백엔드 | Python 3.11, Flask, `anthropic` SDK |
| LLM | Claude Sonnet 4.6 (`claude-sonnet-4-6`) — 메인 / Claude Haiku 4.5 (`claude-haiku-4-5`) — 라우터 |
| 임베딩 | `sentence-transformers` (multilingual MiniLM, 384-dim, 로컬·무료) |
| 프런트엔드 | React + Vite, `react-markdown` + `remark-gfm` |
| 통신 | SSE (Server-Sent Events) 스트리밍 |

---

## 2. 디렉터리 구조

```
rag-starter/
├── documents/              # 코퍼스: 14 CFR (항공 규정) PDF 들 (part 61, 71, 73, 67, 91, vol1)
├── indexer.py              # 문서 → 청크/섹션 분할 → 임베딩 → index.pkl / sections.pkl
├── agent.py                # ★ ReAct 에이전트 + Adaptive RAG 라우팅 + 캐싱 + 스트리밍
├── backend/
│   ├── app.py              # Flask: POST /api/chat → SSE로 에이전트 이벤트 중계
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.jsx         # ★ 채팅 UI: 스트리밍, 상태 배너, 텔레메트리, 인용
│       └── index.css
├── .env                    # ANTHROPIC_API_KEY (git 미추적)
└── .env.example
```

> `index.pkl`, `sections.pkl`은 `python indexer.py`로 생성되는 산출물이라 git에서 무시됩니다.

---

## 3. 아키텍처 / 데이터 흐름

```
[ documents/*.pdf ]
        │  python indexer.py  (1회)
        ▼
[ sections.pkl ]   ← 섹션 단위(§ 61.109 등) 임베딩. 에이전트 검색의 기반
        │
        ▼
 사용자 질문 ──HTTP POST /api/chat──▶ backend/app.py
                                          │  run_agent(question, client)
                                          ▼
                                      agent.py  (ReAct 루프)
                                          │   SSE 이벤트 스트림 (status/tier/usage/tool/delta/done...)
                                          ▼
                                   frontend/App.jsx  ──▶ 화면 렌더링
```

### 인덱싱 (`indexer.py`)
규정 문서는 계층 구조(`§ 61.109 Aeronautical experience.`)가 강해서, **고정 청크가 아니라 섹션 단위**로 인덱싱합니다.
- `split_sections()`: `§` 헤더 정규식으로 문서를 (섹션번호, 제목, 본문)으로 분할
- `find_section_refs()`: 본문이 인용하는 다른 섹션 번호 추출 → `follow_refs`의 기반
- 각 섹션의 "제목 + 첫 500자"를 임베딩 → `navigate_toc`가 목차 검색에 사용

### 에이전트 (`agent.py`)
모델이 다음 3개 도구로 루프(reason → act → observe)를 돕니다:
| 도구 | 역할 |
|---|---|
| `navigate_toc(query)` | 목차에서 후보 섹션을 유사도 랭킹 (전문 X, 미리보기만) |
| `read_section(id)` | 한 섹션 전문을 읽음 + 인용한 상호참조 목록 반환 |
| `follow_refs(id)` | 그 섹션이 참조하는 모든 섹션을 한 번에 읽음 (가장 무거움) |

답변 시 읽은 섹션에 `[1]`, `[2]` 형태로 인용 번호를 매기고, 프런트가 `Sources`로 표시합니다.

---

## 4. 이번에 추가/개선한 기능 (작업 내역)

### 4.1 Adaptive RAG 라우팅 — 토큰 절감의 핵심
모든 질문이 무조건 풀 ReAct 루프(최대 6회 반복)를 도는 비효율을 해결. **값싼 Haiku 모델로 질문 복잡도를 먼저 분류**하고, 티어에 맞는 예산만 씁니다.

| 티어 | 의미 | 예산 | 도구 |
|---|---|---|---|
| **A** | 비검색 (인사, 범위 밖 질문) | 모델 호출 1회, 검색 없음 | 없음 |
| **B** | 단일 섹션 사실 질문 | 최대 2회 반복 | `follow_refs` **제외** |
| **C** | 복합/멀티홉 질문 | 최대 6회 반복 | 전체 도구 |

- 구현: `classify()` (라우터), `TIER_BUDGET` / `TIER_LABEL` (예산표)
- 라우터가 애매하면 안전하게 **C로 폴백**
- 효과: 단순 질문이 트래픽 절반이면 전체 토큰 **30~60% 절감** 기대

### 4.2 프롬프트 캐싱
ReAct 루프는 매 반복마다 **전체 히스토리(이미 읽은 섹션 전문 포함)를 재전송**해서 input 토큰이 누적됩니다.
- `SYSTEM`/`TOOLS` 고정 프리픽스 + 직전 tool_result에 `cache_control` 부여
- `_cache_last()`가 반복마다 캐시 브레이크포인트를 최신 메시지로 **이동** → 큰 섹션 텍스트가 캐시 읽기(~0.1×)로 전환
- UI에 `⚡ cached N tok`로 절감량 가시화

### 4.3 토큰·도구 텔레메트리 UI
- **input/output/합계 토큰** + **캐시 읽기량**을 답변마다 표시 (실제 `usage` 값, 정확)
- **티어 배지**(A=초록/B=노랑/C=빨강)로 라우팅 결과 표시
- **Tool calls 목록**: 도구 이름 · 인자 · 실행 시간(ms) · 결과 토큰(추정, `~`)을 영속 표시

### 4.4 실시간 스트리밍 + 현재 작업 표시
- 가짜 스트리밍(`_chunked`)을 실제 `client.messages.stream()`으로 교체 → 토큰 생성 즉시 출력
- 단계별 상태 배너(펄스 애니메이션): `🧭 Routing…` → `🤔 Thinking… (step N)` → `🔎/📖/🔗 …` → `✍️ Writing…`
- ReAct 턴이 도구 호출로 끝나면 그 턴의 스트리밍 텍스트(preamble)를 `reset_answer`로 비워, 최종 답변만 깨끗하게 남김

### 4.5 마크다운 표 렌더링 버그 수정
- 증상: 답변의 표가 한 줄로 깨짐
- 원인: `react-markdown`은 기본 CommonMark만 지원 → GFM 표 미지원
- 수정: `remark-gfm` 설치 + `<ReactMarkdown remarkPlugins={[remarkGfm]}>` + 표 CSS 추가

---

## 5. SSE 이벤트 프로토콜 (backend → frontend)

`backend/app.py`는 `agent.run()`이 yield하는 dict를 그대로 `data: {json}\n\n` 형태로 중계합니다. 프런트(`App.jsx`)는 아래 타입을 처리합니다.

| `type` | 페이로드 | 프런트 동작 |
|---|---|---|
| `status` | `text` | "현재 작업" 배너 표시 (delta 오면 사라짐) |
| `tier` | `tier`(A/B/C), `label` | 티어 배지 표시 |
| `usage` | `total_input/output`, `total_cache_read/write`, `step_*` | 토큰 패널 갱신 |
| `tool` | `name`, `input`, `duration_ms`, `result_tokens` 등 | Tool calls 목록에 누적 |
| `reset_answer` | — | 스트리밍된 preamble 텍스트 초기화 |
| `delta` | `text` | 답변 텍스트에 실시간 append |
| `done` | `citations`, `usage` | Sources 표시 + 최종 토큰 확정 |

---

## 6. 실행 방법

### 사전 준비 (1회)
```bash
cd rag-starter
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

cp .env.example .env       # 이미 있으면 생략
# .env 에 본인의 ANTHROPIC_API_KEY 설정

python indexer.py          # documents/ → index.pkl, sections.pkl
                           # 최초 1회 임베딩 모델 ~470MB 다운로드
```

### 서버 실행
```bash
# 터미널 1 — 백엔드
cd backend && python app.py          # http://localhost:5000

# 터미널 2 — 프런트엔드
cd frontend && npm install && npm run dev   # http://localhost:5173
```

브라우저에서 <http://localhost:5173> 접속 후 질문.

**예시 질문 (티어별 체감용)**
- A: "안녕?" / "넌 뭐 하는 봇이야?"
- B: "자가용 조종사 자격의 최소 나이는?"
- C: "VFR 연료 예비 요구량은? 비행기와 회전익기를 비교해줘" (상호참조 추적 필요)

---

## 7. 주요 설정값 / 튜닝 포인트 (`agent.py` 상단)

| 상수 | 기본값 | 설명 |
|---|---|---|
| `MODEL` | `claude-sonnet-4-6` | 메인 답변 모델 |
| `ROUTER_MODEL` | `claude-haiku-4-5` | 복잡도 라우터 (값쌈) |
| `MAX_ITERS` | `6` | 티어 C 최대 반복 |
| `MAX_SECTION_CHARS` | `18000` | 한 섹션 전문 길이 상한 |
| `TIER_BUDGET` | — | 티어별 반복 횟수·도구 허용 |
| `ROUTER_SYSTEM` | — | 라우터 분류 기준 프롬프트 (분류가 안 맞으면 여기 조정) |
| `SYSTEM` | — | 메인 에이전트 시스템 프롬프트 (전략·인용 규칙) |

---

## 8. ⚠️ 보안 주의 (중요)

- **`ANTHROPIC_API_KEY`는 절대 커밋·공유 금지.** `.env`는 `.gitignore`에 등록되어 있고 현재 git 미추적 상태.
- 키가 화면/로그/대화에 노출되면 **유출로 간주하고 콘솔에서 즉시 재발급(rotate)** 할 것.
- 새 환경 세팅 시 `.env.example`의 플레이스홀더만 복사해 본인 키를 채울 것.

---

## 9. 알려진 한계 / 다음 작업 후보

- **티어 분류 정확도**: 실제 질문들로 `ROUTER_SYSTEM` 기준을 튜닝 필요. 오분류 시 C 폴백이라 품질은 안전하나 비용↑.
- **캐시 최소 프리픽스**: Sonnet 4.6은 2048토큰부터 캐시됨. `SYSTEM+TOOLS` 단독은 그 미만일 수 있으나, 섹션 텍스트가 붙은 대화 프리픽스는 충분히 커서 캐시됨(절감은 거기서 발생).
- **`result_tokens`는 추정치**(글자수÷4). 정확값이 필요하면 `count_tokens`로 교체 가능(도구마다 추가 호출 비용).
- **README.md는 초기 템플릿**(Apollo 코퍼스 기준)이라 실제 상태와 다름 — 본 문서를 기준으로 볼 것.
- `indexer.py` chunk_text 도크스트링에 오타(`조`) 한 글자 존재 — 동작엔 무관.

---

_문서 작성 기준: 브랜치 `feat/react-toc-retrieval` 현재 상태 (2026-06-30)_
