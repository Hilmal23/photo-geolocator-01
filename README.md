# photo-geolocator-01

Agent that places 25 street-level photos on the map (OpenStreetView-5M),
ranked on total haversine error.

## Approach

`main.py` (Python 3.13, standard library only) fetches the 25 photos from
the env, then makes one vision-model call per photo through the trial's
metered gateway:

- **Anthropic first:** `claude-sonnet-5-5` via `POST {ANTHROPIC_BASE_URL}/v1/messages`,
  `max_tokens` 1024, the photo sent inline as base64.
- **OpenAI fallback vendor:** if the Anthropic call is refused with 401
  (no Anthropic key pushed), it tries `gpt-6-sol` via
  `POST {OPENAI_BASE_URL}/chat/completions` instead.

The prompt asks the model to reason briefly (country/region, nearest
city, landmarks, language on signs, driving side, vegetation, architecture)
and then give its best estimate as one line of JSON `{"lat", "lon"}` on
its own final line. The answer is parsed for the first JSON object with
lat/lon (else the first two in-range numbers). Calls run 4 at a time and
answers are collected by photo index, so the posted answers never depend
on call order. Transient errors (408/429/5xx) are retried; gateway
refusals (4xx) are final.

Expected cost: the kit starter spent $0.15 per trial with 25
claude-sonnet-5-5 calls at 1024 max_tokens; this entry has the same call
shape, so it stays well under the $1.00/trial budget.

## Model-free fallback

If no vendor key was pushed (every model call answered 401), the entry
still posts an answer for every photo using a fixed global prior
(22.0 N, 10.0 E — the middle of the world's inhabited landmass band)
rather than leaving photos unanswered (20,015 km each). A photo whose
model answer holds no coordinates also falls back to the prior.
