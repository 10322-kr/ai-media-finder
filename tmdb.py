"""
TMDB(The Movie Database)에서 최근 상영·방영 작품 목록을 받아 Gemini에 참고 자료로 넘긴다.

Gemini 무료 등급은 Google 검색 그라운딩을 거의 쓸 수 없어서, 모델 학습 이후에 나온 최신작은
무료 TMDB API로 목록을 받아 보충한다. TMDB_API_KEY 환경변수가 없으면 아무것도 하지 않는다.
키 발급: https://www.themoviedb.org/settings/api (API 키 또는 '읽기 액세스 토큰' 둘 다 가능)
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")
# v3 API 키는 URL 쿼리에 들어가므로, 요청 URL을 남기는 httpx 로그와 예외 메시지를 그대로 기록하지 않는다
logging.getLogger("httpx").setLevel(logging.WARNING)

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()
API_BASE = "https://api.themoviedb.org/3"
CACHE_SECONDS = 6 * 60 * 60  # 상영·방영 목록은 자주 바뀌지 않으므로 6시간 동안 재사용
REQUEST_TIMEOUT = 8.0
OVERVIEW_CHARS = 160  # 줄거리를 잘라 Gemini 입력 토큰을 아낀다
ANIMATION_GENRE_ID = 16  # TMDB 장르 '애니메이션' (영화·TV 공통)

# (경로, TMDB 종류, 추가 파라미터)
ENDPOINTS: List[Tuple[str, str, Dict[str, str]]] = [
    ("/movie/now_playing", "movie", {"region": "KR"}),  # 한국 극장 현재 상영작
    ("/trending/movie/week", "movie", {}),
    ("/trending/tv/week", "tv", {}),
    ("/tv/on_the_air", "tv", {}),  # 최근 방영 중인 시리즈
]


class RecentItem(BaseModel):
    title: str
    original_title: Optional[str] = None
    media_type: str  # movie | drama | animation (main.MediaType과 같은 값)
    year: Optional[int] = None
    overview: str = ""


_cache: List[RecentItem] = []
_cache_time = 0.0
_lock = asyncio.Lock()


def enabled() -> bool:
    return bool(TMDB_API_KEY)


def _auth() -> Tuple[Dict[str, str], Dict[str, str]]:
    """(헤더, 쿼리 파라미터). 읽기 액세스 토큰(JWT, eyJ로 시작)은 Bearer, v3 API 키는 api_key로 보낸다."""
    if TMDB_API_KEY.startswith("eyJ"):
        return {"Authorization": f"Bearer {TMDB_API_KEY}"}, {}
    return {}, {"api_key": TMDB_API_KEY}


def _describe_error(e: BaseException) -> str:
    """로그용 오류 설명 (URL에 들어 있는 API 키가 새지 않도록 상태 코드·종류만)."""
    if isinstance(e, httpx.HTTPStatusError):
        return f"HTTP {e.response.status_code}"
    return type(e).__name__


def _to_item(raw: dict, kind: str) -> Optional[RecentItem]:
    title = raw.get("title") or raw.get("name")
    if not title:
        return None
    original = raw.get("original_title") or raw.get("original_name")
    date_str = raw.get("release_date") or raw.get("first_air_date") or ""
    if ANIMATION_GENRE_ID in (raw.get("genre_ids") or []):
        media_type = "animation"
    else:
        media_type = "movie" if kind == "movie" else "drama"
    overview = (raw.get("overview") or "").strip().replace("\n", " ")
    if len(overview) > OVERVIEW_CHARS:
        overview = overview[:OVERVIEW_CHARS] + "…"
    return RecentItem(
        title=title,
        original_title=original if original and original != title else None,
        media_type=media_type,
        year=int(date_str[:4]) if date_str[:4].isdigit() else None,
        overview=overview,
    )


async def _fetch_all() -> List[RecentItem]:
    headers, auth_params = _auth()
    async with httpx.AsyncClient(base_url=API_BASE, headers=headers, timeout=REQUEST_TIMEOUT) as client:

        async def fetch(path: str, params: Dict[str, str]) -> List[dict]:
            res = await client.get(path, params={"language": "ko-KR", **auth_params, **params})
            res.raise_for_status()
            return res.json().get("results", [])

        results = await asyncio.gather(
            *(fetch(path, params) for path, _, params in ENDPOINTS), return_exceptions=True
        )

    items: List[RecentItem] = []
    seen = set()
    for (path, kind, _), result in zip(ENDPOINTS, results):
        if isinstance(result, BaseException):
            logger.warning("TMDB %s 조회 실패: %s", path, _describe_error(result))
            continue
        for raw in result:
            key = (kind, raw.get("id"))
            if key in seen:
                continue
            seen.add(key)
            item = _to_item(raw, kind)
            if item:
                items.append(item)
    return items


async def get_recent_items() -> List[RecentItem]:
    """최근 작품 목록 (캐시). 키가 없거나 조회에 실패하면 이전 캐시 또는 빈 목록."""
    global _cache, _cache_time
    if not enabled():
        return []
    async with _lock:  # 동시에 여러 요청이 와도 TMDB는 한 번만 조회
        if _cache and time.monotonic() - _cache_time < CACHE_SECONDS:
            return _cache
        try:
            items = await _fetch_all()
        except httpx.HTTPError as e:
            logger.warning("TMDB 조회 실패, 이전 목록을 사용합니다: %s", _describe_error(e))
            return _cache
        if items:
            _cache, _cache_time = items, time.monotonic()
            logger.info("TMDB 최근 작품 %d개를 받아왔습니다.", len(items))
        return _cache


def format_items(items: List[RecentItem]) -> str:
    """Gemini 프롬프트용 한 줄 요약 목록."""
    names = {"movie": "영화", "drama": "드라마", "animation": "애니메이션"}
    lines = []
    for it in items:
        title = f"{it.title} ({it.original_title})" if it.original_title else it.title
        parts = [title, names.get(it.media_type, it.media_type), str(it.year or "연도 미상")]
        if it.overview:
            parts.append(it.overview)
        lines.append("- " + " | ".join(parts))
    return "\n".join(lines)
