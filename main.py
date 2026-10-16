"""
AI Media Finder - 어렴풋한 기억으로 책/영화/드라마/애니메이션을 찾아주는 FastAPI 백엔드

실행 방법:
    pip install -r requirements.txt
    PowerShell: $env:GEMINI_API_KEY="AIza..."   (https://aistudio.google.com/apikey 에서 무료 발급)
    python -m uvicorn main:app --reload

선택: TMDB_API_KEY 를 설정하면 '최신작 검색' 때 TMDB의 최근 상영·방영 목록을 참고 자료로 함께 보낸다.
"""

import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field, ValidationError

import tmdb

# gemini-2.5/2.0/1.5-flash는 신규 사용자에게 제공되지 않아(404) 현재 사용 가능한 모델을 사용
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# 기본 모델이 과부하(503 등)일 때 순서대로 재시도할 모델 (쉼표 구분 환경변수로 변경 가능)
GEMINI_FALLBACK_MODELS = [
    m.strip()
    # 무료 등급은 모델별 일일 한도(20회)가 따로라서 여러 모델을 순서대로 시도한다
    for m in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-flash-lite-latest",
    ).split(",")
    if m.strip()
]
# 일시적 오류로 보고 다음 모델로 넘어갈 HTTP 상태 코드
# 429: 모델별 무료 할당량 초과, 500: 내부 오류, 503: 과부하(UNAVAILABLE), 504: 시간 초과
RETRYABLE_STATUS_CODES = {429, 500, 503, 504}

# Google 검색 그라운딩: 모델 학습 이후의 최신 작품·현재 OTT 정보를 검색해서 답하게 한다.
# 응답이 느려지거나 한도가 걱정되면 SEARCH_GROUNDING=0 으로 끈다.
SEARCH_GROUNDING = os.getenv("SEARCH_GROUNDING", "1").strip().lower() not in {"0", "false", "no", "off"}
# 무료 등급은 검색 한도가 거의 없어 매번 429가 난다 → 한 번 걸리면 한동안 검색을 시도하지 않아 요청을 아낀다
GROUNDING_COOLDOWN_SECONDS = 60 * 60
_grounding_blocked_until = 0.0

# 서버 전체의 하루 검색 횟수 제한 (모든 사용자 합산, 한국 시간 자정에 초기화).
# 공개 배포 시 다른 사람이 API 키 한도를 소진하지 않도록 막는다.
# 메모리에만 저장하므로 서버가 재시작되거나 잠들면(Render 무료 플랜) 0으로 돌아간다.
DAILY_SEARCH_LIMIT = int(os.getenv("DAILY_SEARCH_LIMIT", "100"))
KST = timezone(timedelta(hours=9))  # 한국은 서머타임이 없어 고정 오프셋으로 충분하다
_usage_day: Optional[date] = None
_usage_count = 0


def _reset_if_new_day() -> None:
    global _usage_day, _usage_count
    today = datetime.now(KST).date()
    if _usage_day != today:
        _usage_day, _usage_count = today, 0


def remaining_daily_quota() -> int:
    _reset_if_new_day()
    return max(DAILY_SEARCH_LIMIT - _usage_count, 0)


def consume_daily_quota() -> bool:
    """오늘 사용량을 1 늘리고, 한도를 넘었으면 False를 반환한다."""
    global _usage_count
    _reset_if_new_day()
    if _usage_count >= DAILY_SEARCH_LIMIT:
        return False
    _usage_count += 1
    return True

logger = logging.getLogger("uvicorn.error")

# 실행 위치(cwd)와 상관없이 main.py 옆의 index.html을 찾는다
INDEX_HTML = Path(__file__).resolve().parent / "index.html"

MediaType = Literal["book", "movie", "drama", "animation"]

app = FastAPI(
    title="AI Media Finder",
    description="어렴풋한 설명으로 책, 영화, 드라마, 애니메이션을 찾아주는 API",
    version="1.0.0",
)

# CORS 설정: ALLOWED_ORIGINS 환경변수(쉼표 구분)로 허용 도메인을 지정할 수 있다.
# 지정하지 않으면 개발 편의를 위해 모든 출처를 허용한다.
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    # 와일드카드(*) 출처와 credentials는 함께 쓸 수 없으므로 명시적 출처일 때만 허용
    allow_credentials="*" not in allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

_client: Optional[genai.Client] = None


def get_client() -> genai.Client:
    """API 키가 없어도 서버가 기동되도록 첫 요청 시점에 클라이언트를 만든다."""
    global _client
    if _client is None:
        _client = genai.Client()  # GEMINI_API_KEY 또는 GOOGLE_API_KEY 환경변수를 사용
    return _client


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="찾고 싶은 작품에 대한 어렴풋한 설명",
        examples=["주인공이 매일 같은 날을 반복하는 영화인데 마지막에 사랑을 깨닫는 내용"],
    )
    media_types: List[MediaType] = Field(
        default_factory=list,
        max_length=4,
        description="이 종류의 작품만 찾는다. 비어 있으면 모든 종류",
    )
    latest: bool = Field(False, description="Google 검색으로 최신작·현재 상영작까지 찾는다 (느리고 한도를 더 씀)")


class Candidate(BaseModel):
    title: str
    media_type: MediaType
    release_year: Optional[int] = None
    reason: str = Field(..., description="이 후보가 설명과 맞을 수 있는 이유")


class SearchResponse(BaseModel):
    title: str
    media_type: MediaType
    release_year: Optional[int] = None
    ott_platforms: List[str] = Field(default_factory=list)
    one_liner_summary: str
    match_reason: str
    candidates: List[Candidate] = Field(default_factory=list)


SYSTEM_PROMPT = """너는 책, 영화, 드라마, 애니메이션에 정통한 미디어 큐레이터다.
사용자가 어렴풋하게 기억하는 줄거리, 장면, 인물, 분위기, 시기 등의 단서를 분석해서
가장 가능성이 높은 작품 하나와 다른 후보 작품들을 찾아라.

반드시 아래 형식의 JSON 객체 하나만 응답하라. 다른 텍스트는 절대 포함하지 마라.
{
  "title": "가장 유력한 작품의 제목 (한국어 제목이 있으면 한국어 제목, 괄호 안에 원제)",
  "media_type": "book | movie | drama | animation 중 하나",
  "release_year": 출시/출간 연도 (정수, 모르면 null),
  "ott_platforms": ["한국에서 시청 가능한 것으로 알려진 OTT 플랫폼 (예: Netflix, Disney+, TVING, Wavve, Coupang Play, Watcha, Apple TV+, Prime Video). 책이거나 모르면 빈 배열"],
  "one_liner_summary": "작품을 스포일러 없이 재치 있고 위트 넘치는 한 문장으로 요약 (예: 영화 홍보 문구처럼)",
  "match_reason": "사용자의 설명 중 어떤 단서가 이 작품과 일치하는지 구체적으로 설명",
  "candidates": [
    {
      "title": "다른 후보 작품 제목",
      "media_type": "book | movie | drama | animation 중 하나",
      "release_year": 정수 또는 null,
      "reason": "이 후보도 해당될 수 있는 이유"
    }
  ]
}

규칙:
- candidates에는 최유력 작품을 제외한 후보를 2~4개 넣어라.
- 실제로 존재하는 작품만 제시하고, 확실하지 않은 정보는 지어내지 말고 null 또는 빈 배열로 둬라.
- OTT 정보는 시기에 따라 바뀔 수 있으므로 확실한 것만 넣어라.
- 모든 설명 문장은 한국어로 작성하라."""

# 최신작 검색을 켰을 때만 붙이는 규칙 (검색 결과가 입력 토큰으로 들어가고 검색 한도도 쓰므로 필요할 때만)
GROUNDING_RULE = """- 답하기 전에 반드시 검색 도구로 검색하라. 학습 이후에 나온 최신작·현재 상영작일 수 있으니
  기억만으로 판단하지 말고, 사용자 설명의 핵심 단서로 검색해 후보를 확인하고 OTT 정보도 검색으로 확인하라."""


def build_configs(latest: bool) -> List[types.GenerateContentConfig]:
    """시도할 요청 설정 목록 (앞에서부터 시도).

    최신작 검색(latest)을 켜면:
    1) 검색 + JSON 스키마 강제  2) 검색만 (모델이 조합을 지원하지 않을 때, 프롬프트로 JSON 유도)
    3) 검색 없이 JSON 스키마 강제 (모델이 검색 도구 자체를 지원하지 않을 때)
    끄면 3)만 사용한다.
    """
    use_search = latest and SEARCH_GROUNDING
    prompt = f"{SYSTEM_PROMPT}\n{GROUNDING_RULE}" if use_search else SYSTEM_PROMPT
    # 오늘 날짜를 알려줘야 '최근', '올해' 같은 표현을 제대로 해석한다
    system = f"{prompt}\n\n오늘 날짜: {datetime.now(KST).date().isoformat()}"
    schema = dict(response_mime_type="application/json", response_schema=SearchResponse)
    search = dict(tools=[types.Tool(google_search=types.GoogleSearch())])
    variants = [{**search, **schema}, search, schema] if use_search else [schema]
    return [types.GenerateContentConfig(system_instruction=system, temperature=0.3, **v) for v in variants]


def log_grounding(model: str, config: types.GenerateContentConfig, response: types.GenerateContentResponse) -> None:
    """모델이 검색 도구를 받고도 실제로 검색했는지는 모델이 정하므로, 실제 검색어를 로그로 남긴다."""
    meta = response.candidates[0].grounding_metadata if response.candidates else None
    queries = (meta.web_search_queries if meta else None) or []
    logger.info(
        "Gemini 모델 %s 응답 (%s), 검색어: %s", model, describe_config(config), queries or "검색 안 함"
    )


def describe_config(config: types.GenerateContentConfig) -> str:
    """로그용: 이 요청에 검색 도구를 줬는지와 JSON 스키마를 강제했는지."""
    tool = "검색 도구 제공" if config.tools else "검색 도구 없음"
    return f"{tool}, {'JSON 강제' if config.response_schema else 'JSON 미강제'}"


def parse_response(text: str) -> SearchResponse:
    """스키마를 강제하지 못한 응답은 ```json 코드 블록이나 앞뒤 문장이 섞여 올 수 있어 JSON 부분만 꺼낸다."""
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    return SearchResponse.model_validate_json(text)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """프런트엔드(index.html)를 같은 서버에서 제공해 /api/search를 상대 경로로 호출할 수 있게 한다."""
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "AI Media Finder"}


class QuotaResponse(BaseModel):
    remaining: int
    limit: int
    latest_search: bool  # 서버에서 최신작 검색(Google 검색 또는 TMDB 목록)을 쓸 수 있는지


@app.get("/api/quota", response_model=QuotaResponse)
async def get_quota() -> QuotaResponse:
    """오늘 남은 검색 횟수 (서버 전체 합산)."""
    return QuotaResponse(
        remaining=remaining_daily_quota(), limit=DAILY_SEARCH_LIMIT, latest_search=SEARCH_GROUNDING or tmdb.enabled(),
    )


MEDIA_TYPE_NAMES = {"book": "책", "movie": "영화", "drama": "드라마", "animation": "애니메이션"}


def build_contents(request: SearchRequest, recent: List[tmdb.RecentItem]) -> str:
    """사용자 설명 뒤에 종류 제한 조건과 최근 작품 목록(최신작 검색 시)을 붙인다."""
    parts = [request.query]
    types_ = list(dict.fromkeys(request.media_types))  # 중복 제거, 순서 유지
    if types_ and len(types_) < len(MEDIA_TYPE_NAMES):
        names = ", ".join(f"{MEDIA_TYPE_NAMES[t]}({t})" for t in types_)
        parts.append(
            f"[검색 조건] 최유력 작품과 모든 후보는 반드시 다음 종류 중에서만 골라라: {names}. "
            "media_type도 이 값들 중 하나여야 한다."
        )
    if recent:
        parts.append(
            "[참고: 최근 상영·방영 작품 목록 (TMDB 기준, 제목 | 종류 | 연도 | 줄거리)]\n"
            f"{tmdb.format_items(recent)}\n"
            "이 목록은 학습 이후에 나온 최신작을 보충하기 위한 참고 자료다. 사용자 설명과 맞는 작품이 "
            "목록에 있으면 우선 고려하되, 맞는 작품이 없으면 무시하고 원래 알고 있는 작품에서 찾아라."
        )
    return "\n\n".join(parts)


async def recent_items_for(request: SearchRequest) -> List[tmdb.RecentItem]:
    """최신작 검색일 때만, 필터에 맞는 종류의 최근 작품 목록 (TMDB에는 책이 없다)."""
    if not request.latest:
        return []
    items = await tmdb.get_recent_items()
    if request.media_types:
        items = [it for it in items if it.media_type in request.media_types]
    return items


class SearchResult(SearchResponse):
    """API 응답. SearchResponse는 Gemini에 넘기는 스키마이기도 해서, 서버가 채우는 값은 여기에 둔다."""

    latest_fallback: bool = False  # 최신작 검색을 요청했지만 Google 검색도 TMDB 목록도 없이 답함


@app.post("/api/search", response_model=SearchResult)
async def search_media(request: SearchRequest) -> SearchResult:
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

    if not consume_daily_quota():
        raise HTTPException(
            status_code=429,
            detail=f"오늘 준비된 티켓({DAILY_SEARCH_LIMIT}장)이 모두 매진됐어요. 내일 다시 찾아 주세요!",
        )

    recent = await recent_items_for(request)
    contents = build_contents(request, recent)
    models = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL]
    last_error: Optional[errors.APIError] = None

    global _grounding_blocked_until
    use_search = request.latest and time.monotonic() >= _grounding_blocked_until
    if recent:
        logger.info("TMDB 최근 작품 %d개를 참고 자료로 보냅니다.", len(recent))

    for model in models:
        configs = build_configs(use_search)
        i = 0
        while i < len(configs):
            config = configs[i]
            try:
                response = await get_client().aio.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except errors.APIError as e:
                # 400: 이 모델이 검색 도구(또는 도구+JSON 스키마 조합)를 지원하지 않음 → 다음 방식으로
                if e.code == 400 and i < len(configs) - 1:
                    logger.warning("Gemini 모델 %s가 설정 %d을(를) 거부(%s), 다음 방식으로 재시도합니다.", model, i, e)
                    i += 1
                    continue
                # 검색 그라운딩은 한도가 따로라 먼저 소진된다 → 이번 요청은 남은 모델까지 검색 없이 진행
                if e.code == 429 and config.tools:
                    logger.warning("Gemini 모델 %s 검색 한도 초과로 보고 검색 없이 재시도합니다: %s", model, e.message)
                    use_search = False
                    _grounding_blocked_until = time.monotonic() + GROUNDING_COOLDOWN_SECONDS
                    configs, i = build_configs(False), 0
                    continue
                if e.code in RETRYABLE_STATUS_CODES:
                    # 어떤 한도(분당/일일, 검색 그라운딩 등)에 걸렸는지 알 수 있게 Google의 메시지도 남긴다
                    logger.warning(
                        "Gemini 모델 %s 일시 오류(%s, %s), 다음 모델로 재시도합니다: %s",
                        model, e.code, describe_config(config), e.message,
                    )
                    last_error = e
                    break
                raise HTTPException(status_code=502, detail=f"Gemini API 호출 실패: {e}") from e

            try:
                log_grounding(model, config, response)
                result = parse_response(response.text or "")
            except ValidationError as e:
                raise HTTPException(status_code=502, detail="AI 응답을 해석하지 못했습니다. 다시 시도해 주세요.") from e
            # 최신작 검색을 요청했지만 검색 없이 답한 경우 화면에 알린다
            return SearchResult(
                **result.model_dump(), latest_fallback=request.latest and not config.tools and not recent
            )

    # 모든 모델이 일시 오류로 실패한 경우
    if last_error is not None and last_error.code == 429:
        raise HTTPException(
            status_code=429, detail="Gemini 무료 사용량 한도를 초과했습니다. 잠시 후 다시 시도해 주세요."
        ) from last_error
    raise HTTPException(
        status_code=503, detail="Gemini 서버가 일시적으로 혼잡합니다. 잠시 후 다시 시도해 주세요."
    ) from last_error

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
