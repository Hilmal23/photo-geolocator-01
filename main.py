#!/usr/bin/env python3
"""photo-geolocator-01 entry: place 25 street-level photos on the map.

One vision-model call per photo through the trial's metered gateway
(Anthropic Messages for claude-sonnet-5-5, OpenAI Chat Completions for
gpt-6-sol as the fallback vendor), asking for brief reasoning and then a
{"lat", "lon"} JSON estimate. Every answer is posted to the env at the
end, in photo order. If no vendor key was pushed (every model call
answered 401) the entry still posts an answer for every photo, using a
fixed global prior, rather than leaving photos unanswered.

Standard library only. Deterministic given the model's answers: no wall
clock, no unseeded randomness; model calls run 4 at a time and their
answers are collected by photo index, so the posted answers never depend
on which call came back first.
"""

import base64
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ANTHROPIC_MODEL = "claude-sonnet-5-5"
OPENAI_MODEL = "gpt-6-sol"
MAX_TOKENS = 1024
CONCURRENCY = 4
RETRY_STATUSES = (408, 429, 500, 502, 503, 504, 529)
RETRIES = 3
HTTP_TIMEOUT_S = 120

# Fixed global prior, used only when no model call succeeds: a point in
# the middle of the world's inhabited landmass band (northern Africa),
# the least-bad single guess for a street photo from anywhere on Earth.
FALLBACK_LAT = 22.0
FALLBACK_LON = 10.0

PROMPT = (
    "You are an expert geolocation analyst. Study this street-level photo "
    "and work out where on Earth it was taken.\n\n"
    "Reason briefly, in a few short lines, about:\n"
    "- country and region, and the nearest city or town you can name\n"
    "- any visible landmarks, street names, or business names\n"
    "- the language and script on signs\n"
    "- which side of the road traffic drives on\n"
    "- vegetation, climate, and terrain\n"
    "- architecture, road markings, and license plates\n\n"
    "Then give your best estimate as one line of JSON on its own final line:\n"
    '{"lat": <degrees north>, "lon": <degrees east>}'
)

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def log(msg):
    sys.stderr.write("geo: %s\n" % msg)
    sys.stderr.flush()


def _exchange(method, url, headers, body):
    """One HTTP exchange -> (status, body bytes); None status = no answer."""
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with _OPENER.open(req, timeout=HTTP_TIMEOUT_S) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as err:
        try:
            return err.code, err.read()
        finally:
            err.close()
    except (urllib.error.URLError, OSError):
        return None, b""


def _is_lat(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and -90.0 <= v <= 90.0


def _is_lon(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and -180.0 <= v <= 180.0


def coordinates(text):
    """The latitude/longitude an answer gives: its first JSON object with
    lat/lon, else its first two numbers in range."""
    if not text:
        return None
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            o = json.loads(m.group(0))
        except ValueError:
            continue
        if not isinstance(o, dict):
            continue
        lat = o.get("lat", o.get("latitude"))
        lon = o.get("lon", o.get("lng", o.get("longitude")))
        if _is_lat(lat) and _is_lon(lon):
            return (float(lat), float(lon))
    nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text)]
    for i in range(len(nums) - 1):
        if _is_lat(nums[i]) and _is_lon(nums[i + 1]):
            return (nums[i], nums[i + 1])
    return None


class Gateway:
    """The trial's model gateway: Anthropic first, OpenAI on a 401."""

    def __init__(self):
        self.anthropic_base = (os.environ.get("ANTHROPIC_BASE_URL") or "").rstrip("/")
        self.openai_base = (os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY") or ""
        self.openai_key = os.environ.get("OPENAI_API_KEY") or ""
        self.vendor = None  # "anthropic" | "openai" | "none"
        self._lock = threading.Lock()

    def _anthropic_call(self, jpeg_b64):
        body = json.dumps(
            {
                "model": ANTHROPIC_MODEL,
                "max_tokens": MAX_TOKENS,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": jpeg_b64,
                                },
                            },
                            {"type": "text", "text": PROMPT},
                        ],
                    }
                ],
            },
            separators=(",", ":"),
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-api-key": self.anthropic_key,
            "anthropic-version": "2023-06-01",
        }
        return self._call("anthropic", self.anthropic_base + "/v1/messages", headers, body)

    def _openai_call(self, jpeg_b64):
        body = json.dumps(
            {
                "model": OPENAI_MODEL,
                "max_completion_tokens": MAX_TOKENS,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64," + jpeg_b64
                                },
                            },
                        ],
                    }
                ],
            },
            separators=(",", ":"),
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer " + self.openai_key,
        }
        return self._call("openai", self.openai_base + "/chat/completions", headers, body)

    def _call(self, vendor, url, headers, body):
        """One model call -> (text | None, refused: bool).

        refused is True only for a final gateway refusal (4xx other than
        the retried statuses): no point retrying, and for a 401 it means
        this vendor has no key pushed. Transient errors are retried.
        """
        for attempt in range(RETRIES + 1):
            status, raw = _exchange("POST", url, headers, body)
            if status is None:
                if attempt < RETRIES:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                return None, False
            if 200 <= status < 300:
                try:
                    msg = json.loads(raw.decode("utf-8"))
                except ValueError:
                    return None, False
                if vendor == "anthropic":
                    text = "\n".join(
                        b.get("text", "")
                        for b in msg.get("content", [])
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                else:
                    try:
                        text = msg["choices"][0]["message"]["content"] or ""
                    except (KeyError, IndexError, TypeError):
                        text = ""
                return text, False
            if status in RETRY_STATUSES and attempt < RETRIES:
                time.sleep(1.0 * (attempt + 1))
                continue
            log("%s call refused (HTTP %s): %s" % (vendor, status, raw[:200].decode("utf-8", "replace")))
            return None, True
        return None, False

    def ask(self, jpeg_b64):
        """Ask the working vendor for one photo -> answer text or None."""
        with self._lock:
            vendor = self.vendor
        if vendor == "none":
            return None
        if vendor is None:
            vendor = "anthropic"
        text, refused = self._try(vendor, jpeg_b64)
        if text is not None:
            with self._lock:
                self.vendor = vendor
            return text
        if refused:
            other = "openai" if vendor == "anthropic" else "anthropic"
            if self._has(other):
                text2, _ = self._try(other, jpeg_b64)
                with self._lock:
                    self.vendor = other if text2 is not None else "none"
                return text2
            with self._lock:
                self.vendor = "none"
            return None
        return None

    def _has(self, vendor):
        return bool(self.anthropic_base if vendor == "anthropic" else self.openai_base)

    def _try(self, vendor, jpeg_b64):
        if vendor == "anthropic":
            if not self.anthropic_base:
                return None, True
            return self._anthropic_call(jpeg_b64)
        if not self.openai_base:
            return None, True
        return self._openai_call(jpeg_b64)


def main():
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE")
    env_url = (os.environ.get("MAKEFUN_ENV_URL") or "").rstrip("/")
    if not prompt_file:
        log("MAKEFUN_PROMPT_FILE is not set")
        return 1
    if not env_url:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            brief = f.read()
    except OSError as err:
        log("cannot read the prompt file: %s" % err)
        return 1
    log("brief: %d chars" % len(brief))

    status, raw = _exchange("GET", env_url + "/photos", {"Accept": "application/json"}, None)
    if status != 200:
        log("GET /photos failed: HTTP %s" % status)
        return 1
    try:
        listed = json.loads(raw.decode("utf-8"))
        photos = [p for p in listed["photos"] if not p.get("answered")]
    except (ValueError, KeyError, TypeError):
        log("GET /photos answered a body that is not the photo list")
        return 1
    log("%d photos to place" % len(photos))

    jpegs = []
    for p in photos:
        status, raw = _exchange("GET", env_url + p["url"], {"Accept": "image/jpeg"}, None)
        if status != 200:
            log("GET %s failed: HTTP %s" % (p["url"], status))
            jpegs.append(None)
        else:
            jpegs.append(base64.b64encode(raw).decode("ascii"))
    log("fetched %d of %d photos" % (sum(1 for j in jpegs if j), len(jpegs)))

    gw = Gateway()
    guesses = [None] * len(photos)

    def place(i):
        jpeg_b64 = jpegs[i]
        if jpeg_b64 is None:
            return (FALLBACK_LAT, FALLBACK_LON)
        text = gw.ask(jpeg_b64)
        c = coordinates(text)
        if c is None:
            if text is not None:
                log("no coordinates in the answer for %s" % photos[i]["id"])
            return (FALLBACK_LAT, FALLBACK_LON)
        return c

    # The first photo decides the vendor (Anthropic, else OpenAI, else
    # none) before the rest run in parallel on that vendor.
    if photos:
        guesses[0] = place(0)
    if len(photos) > 1:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            for i, c in enumerate(pool.map(place, range(1, len(photos))), start=1):
                guesses[i] = c

    answers = [
        {"id": p["id"], "lat": round(guesses[i][0], 6), "lon": round(guesses[i][1], 6)}
        for i, p in enumerate(photos)
    ]
    body = json.dumps({"answers": answers}, separators=(",", ":")).encode("utf-8")
    for attempt in range(2):
        status, raw = _exchange(
            "POST", env_url + "/answers", {"Content-Type": "application/json"}, body
        )
        if status == 200:
            log("posted %d answers: %s" % (len(answers), raw.decode("utf-8", "replace")[:200]))
            return 0
        log("POST /answers failed (HTTP %s): %s" % (status, raw[:200].decode("utf-8", "replace")))
        time.sleep(1.0)
    return 1


if __name__ == "__main__":
    sys.exit(main())
