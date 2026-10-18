"""Shared helpers: config loading, caching, HTTP with retry.

Every network call writes its response to disk under data/raw/<source>/<date>/
so that the same date range can be re-read offline ("replay mode").
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests
import yaml
from dotenv import load_dotenv

# Load .env once at import time
load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG: dict[str, Any] | None = None


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    """Read config.yaml and return as a dict. Cached after first call."""
    global _CONFIG
    if _CONFIG is None:
        with open(path, encoding="utf-8") as fh:
            _CONFIG = yaml.safe_load(fh)
    return _CONFIG


def get_secret(name: str) -> str:
    """Read a secret from the environment (loaded from .env)."""
    return os.getenv(name, "")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def cache_path(source: str, date_str: str, filename: str) -> Path:
    """Return the path where a cached response should live.

    Layout: data/raw/<source>/<YYYY-MM-DD>/<filename>.json
    """
    cfg = load_config()
    base = Path(cfg["ingest"]["cache_dir"])
    directory = base / source / date_str
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def read_cache(source: str, date_str: str, filename: str) -> dict | list | None:
    """Load a cached JSON file. Returns None if not found."""
    path = cache_path(source, date_str, filename)
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def write_cache(
    source: str, date_str: str, filename: str, data: dict | list
) -> Path:
    """Write data to cache as JSON. Returns the path written."""
    path = cache_path(source, date_str, filename)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def fetch_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    retries: int = 2,
    backoff_base: float = 2.0,
    timeout: int = 15,
) -> dict | list:
    """GET a URL and return parsed JSON. Retries on transient errors.

    Retries on: connection errors, timeouts, 429 (rate limit), 5xx,
    and JSON decode errors (API returning HTML error pages).
    Raises after all retries fail.
    """
    req_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
    }
    if headers:
        req_headers.update(headers)

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                url, params=params, headers=req_headers, timeout=timeout
            )
            # Retry on 429 (rate limit) and 5xx (server errors)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == retries:
                    resp.raise_for_status()
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = min(float(retry_after), 4.0)
                else:
                    wait = min(backoff_base * attempt, 4.0)
                print(f"  [retry {attempt}/{retries}] HTTP {resp.status_code} - waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if attempt == retries:
                raise
            wait = min(backoff_base * attempt, 3.0)
            print(f"  [retry {attempt}/{retries}] {exc!r} - waiting {wait:.0f}s")
            time.sleep(wait)
        except (ValueError, requests.exceptions.JSONDecodeError) as exc:
            # API returned non-JSON (e.g. HTML error page after rate limit)
            last_exc = exc
            if attempt == retries:
                raise
            wait = min(backoff_base * attempt, 3.0)
            print(f"  [retry {attempt}/{retries}] Bad JSON response - waiting {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"fetch_json: all retries exhausted: {last_exc}")

