"""tools.py - Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET  # arXiv answers with Atom XML

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

ATOM = "{http://www.w3.org/2005/Atom}"
RETRY_STATUSES = {429, 500, 502, 503, 504}
SUMMARY_CHARS = 600
FETCH_CHARS = 12_000
ARXIV_MIN_INTERVAL = 3.0  # arXiv API etiquette: at least 3 s between two calls
USER_AGENT = "deep-research-lab/1.0 (educational)"

_arxiv_lock = threading.Lock()  # researchers run in parallel: serialise arXiv calls to keep the interval
_arxiv_last_call = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0, sleep=time.sleep):
    """Call fn(); when it raises RetryableError (or httpx.TransportError), wait and call it again.

    Delay: the server's Retry-After when given, else exponential backoff base * 2**attempt plus random jitter;
    always capped at `cap`. The last failure is re-raised without sleeping. Any other exception propagates at once.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except (RetryableError, httpx.TransportError) as exc:
            if attempt == attempts - 1:
                raise
            retry_after = getattr(exc, "retry_after", None)
            if retry_after is not None:
                delay = min(float(retry_after), cap)
            else:
                backoff = base * 2 ** attempt
                delay = min(backoff + random.uniform(0, backoff), cap)
            sleep(delay)


def _retry_after(response):
    """Seconds from a numeric Retry-After header, else None."""
    try:
        return max(0.0, float(response.headers.get("Retry-After", "")))
    except ValueError:
        return None


def _get(url, params):
    """One GET; retryable statuses become RetryableError, other HTTP errors raise httpx.HTTPStatusError."""
    response = httpx.get(url, params=params, timeout=30, follow_redirects=True, headers={"User-Agent": USER_AGENT})
    if response.status_code in RETRY_STATUSES:
        raise RetryableError(f"HTTP {response.status_code} from {url}", _retry_after(response))
    response.raise_for_status()
    return response


def _clean(text):
    return " ".join(str(text or "").split())


def _short(text, limit=SUMMARY_CHARS):
    text = _clean(text)
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def _error(exc, secret=None):
    message = f"ERROR: {type(exc).__name__}: {exc}"
    return message.replace(secret, "***") if secret else message


# ---- TODO 2: arXiv ----
def _arxiv_get(params):
    global _arxiv_last_call
    with _arxiv_lock:
        wait = ARXIV_MIN_INTERVAL - (time.monotonic() - _arxiv_last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            return _get(ARXIV_URL, params)
        finally:
            _arxiv_last_call = time.monotonic()


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords (a few plain words work best, e.g. "world model survey"), newest first.
    Returns a JSON list of {id, url, published, title, summary}; url is https://arxiv.org/abs/<id>.
    Use it for primary research papers and recent preprints. Returns "NO RESULTS" or "ERROR: ..." otherwise."""
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", query or "")
    terms = [t for t in terms if t.upper() not in {"AND", "OR", "ANDNOT", "ALL"}][:8]
    if not terms:
        return "NO RESULTS"
    params = {"search_query": " AND ".join(f"all:{t}" for t in terms), "sortBy": "submittedDate",
              "sortOrder": "descending", "max_results": max(1, min(int(max_results), 30))}
    try:
        response = with_retry(lambda: _arxiv_get(params), attempts=6, base=3.0, cap=60.0)
        records = []
        for entry in ET.fromstring(response.content).findall(f"{ATOM}entry"):
            raw_id = entry.findtext(f"{ATOM}id", "").rsplit("/abs/", 1)[-1]
            paper_id = re.sub(r"v\d+$", "", raw_id.strip())
            if not paper_id:
                continue
            records.append({"id": paper_id, "url": f"https://arxiv.org/abs/{paper_id}",
                            "published": entry.findtext(f"{ATOM}published", "")[:10],
                            "title": _clean(entry.findtext(f"{ATOM}title")),
                            "summary": _short(entry.findtext(f"{ATOM}summary"))})
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # a tool never raises
        return _error(exc)


# ---- TODO 3: Hugging Face ----
def _hf_record(item):
    paper = item.get("paper") or {}
    paper_id = paper.get("id")
    if not paper_id:
        return None
    return {"id": paper_id, "url": f"https://huggingface.co/papers/{paper_id}",
            "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
            "title": _clean(paper.get("title") or item.get("title")),
            "summary": _short(paper.get("ai_summary") or paper.get("summary") or item.get("summary")),
            "upvotes": paper.get("upvotes", 0), "github": paper.get("githubRepo"), "stars": paper.get("githubStars")}


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    params = {"limit": max(1, min(int(limit), 100))}
    if date:
        params["date"] = date
    try:
        items = with_retry(lambda: _get(HF_DAILY_URL, params)).json()
        records = [r for r in map(_hf_record, items) if r]
        if keyword:
            needle = keyword.lower()
            records = [r for r in records if needle in f"{r['title']} {r['summary']}".lower()]
        records.sort(key=lambda r: r["upvotes"] or 0, reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic (short keyword query). Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}; url is https://huggingface.co/papers/<id>.
    Good for finding well-known, community-upvoted papers and their code."""
    if not _clean(query):
        return "NO RESULTS"
    params = {"q": _clean(query), "limit": max(1, min(int(limit), 50))}
    try:
        items = with_retry(lambda: _get(HF_SEARCH_URL, params)).json()
        records = [r for r in map(_hf_record, items) if r][:params["limit"]]
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _exa_call(name, arguments):
    """Call one Exa MCP tool over plain HTTP (JSON-RPC tools/call). Returns its text; retries on rate limits."""
    key = (os.getenv("EXA_API_KEY") or "").strip()
    url = f"{EXA_URL}?exaApiKey={key}" if key else EXA_URL
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}

    def call():
        response = httpx.post(url, json=payload, timeout=60,
                              headers={"Accept": "application/json, text/event-stream", "User-Agent": USER_AGENT})
        if response.status_code in RETRY_STATUSES:
            raise RetryableError(f"HTTP {response.status_code} from Exa", _retry_after(response))
        response.raise_for_status()
        message = None
        for line in response.text.splitlines():
            if line.startswith("data:"):
                message = json.loads(line[5:])
        if message is None:  # plain JSON answer instead of server-sent events
            message = response.json()
        if "error" in message:
            raise RuntimeError(f"Exa JSON-RPC error: {message['error']}")
        result = message.get("result") or {}
        # the free tier answers HTTP 200 with a text notice and this flag instead of a 429
        if (result.get("_meta") or {}).get("ai.exa/rateLimited"):
            raise RetryableError("Exa rate limited")
        text = "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text").strip()
        if result.get("isError"):
            raise RuntimeError(f"Exa tool error: {text[:300]}")
        return text

    try:
        return with_retry(call, attempts=6, base=5.0, cap=60.0) or "NO RESULTS"
    except Exception as exc:
        return _error(exc, secret=key or None)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs.
    Use it for blogs, project pages, surveys and news that are not on arXiv/Hugging Face; `objective` says what you
    want to learn from the results."""
    if not _clean(query):
        return "NO RESULTS"
    objective = _clean(objective) or f"Find authoritative, informative pages about: {_clean(query)}"
    return _exa_call("web_search_exa", {"query": _clean(query), "objective": objective,
                                        "numResults": max(1, min(int(num_results), 10))})


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated.
    Use it to verify a claim or to read the details of a page whose URL you already have."""
    if not str(url or "").startswith(("http://", "https://")):
        return "ERROR: url must start with http:// or https://"
    text = _exa_call("web_fetch_exa", {"urls": [url]})
    return text if len(text) <= FETCH_CHARS else text[:FETCH_CHARS] + "\n...[truncated]"


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
