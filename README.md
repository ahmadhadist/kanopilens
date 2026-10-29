# Kanopi Lens

A trusted grading and health standard for the rare plant trade, built with Meta SAM 3.1 and Muse Spark.

Upload a photo of an aroid and get:

- **Leaf segmentation**: SAM 3.1 isolates every leaf.
- **Objective variegation**: a per-leaf percentage computed from the pixels inside each mask. It is deterministic, so the same photo always gives the same number.
- **Health scan**: Muse Spark inspects each labelled leaf for pests, diseases, deficiencies and stress, and returns a confidence and a care recommendation for each finding.
- **Grading Card**: a shareable page that sellers attach to a listing and buyers abroad can trust.
- **Buyer care guide**: care instructions in the buyer's language, adapted to the climate and indoor conditions of their country.
- **Proof of Plant (video)**: the seller records 10–20 s walking around the plant. SAM 3.1 tracks every leaf across frames, so leaves are counted on the whole plant, not on the best angle. Each leaf's variegation is measured on the frame where it appears largest. The SHA-256 of the original recording is stored with the scan as evidence.
- **Arrival Check**: the buyer scans the plant on arrival, and Kanopi Lens compares it with the seller's scan. It checks leaf count, variegation, health, plant identity and shipping damage, then returns a verdict that settles disputes with evidence.

Kanopi Lens is API-first. Kanopi (a local and export plant marketplace) is its first client. Nurseries, exporters and other marketplaces can integrate the same API with a paid API key.

> Automated pre-screening only. It is not a substitute for expert inspection or an official phytosanitary certificate.

## How it works

```
photo ──► SAM 3.1 "leaf" ──► per-leaf masks ──► HSV pixel classification ──► variegation % + grade
                                   │
                                   └──► labelled overlay ──► Muse Spark (JSON schema) ──► health issues + score
scan ──► Muse Spark (JSON schema, language + country) ──► cached care guide
video ──► resample (4 fps, ≤15 s, ≤960 px) ──► Files API ──► SAM 3.1 video, streamed ──► leaf tracks
      └──► best frame per leaf ──► variegation per leaf + leaf gallery ──► Muse Spark health scan
seller scan + arrival photo/video ──► same pipeline ──► rule-based metric deltas
                                         └──► Muse Spark (both overlays) ──► same plant? shipping damage? ──► verdict
```

The Arrival Check verdict (`consistent`, `minor_changes` or `significant_changes`) comes from fixed rules. Losing leaves, a variegation shift (variegation does not change in transit, so a shift suggests a different plant), a health drop, new issues, and Muse's identity and damage findings each raise it. Every finding is listed in plain language.

- SAM output is parsed with Meta's `meta-sam-parser`. Masks are box-local rasters, so each one is scaled back into its box on the full frame.
- The health and care prompts use `text.format` JSON schemas with `strict: true`, so responses always parse.
- The health score is computed deterministically from the reported issues (severity × confidence), not taken from the model's own guess.

## API

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/v1/scans` | `X-API-Key` | Multipart `image` (a JPEG/PNG/WebP photo or an MP4/MOV video up to 30 MB) plus an optional `species_hint` (max 120 characters) |
| `GET` | `/v1/scans/{id}` | public | Scan result as JSON |
| `GET` | `/v1/scans/{id}/image.jpg`, `/overlay.jpg` | public | Processed photo or labelled overlay |
| `GET` | `/v1/scans/{id}/video` | public | Original recording of a video scan (matches `video.sha256`) |
| `GET` | `/v1/scans/{id}/card` | public | Shareable HTML Grading Card |
| `GET` | `/v1/scans/{id}/care-guide?lang=en&country=US` | public, rate-limited | Localized care guide, cached per scan, language and country |
| `POST` | `/v1/uploads` | `X-API-Key` | Direct upload for files above 30 MB (such as 4K phone video, up to 150 MB): returns a signed Cloud Storage URL. PUT the file there, then pass `upload_id` to a scan endpoint instead of `image` |
| `POST` | `/v1/scans/{id}/arrival` | `X-API-Key` | Arrival Check: upload the buyer's photo and compare it with seller scan `{id}` |
| `POST` | `/demo/scans`, `/demo/scans/{id}/arrival` | rate-limited per IP | Used by the web demo at `/` |

Scan IDs are random 128-bit hex values, so they work as unguessable share links. Interactive docs are served at `/docs`.

The web demo at `/` uploads a photo, shows the overlay, per-leaf variegation, the health findings and the care guide. A result can be reopened with `/?scan=<id>`.

## Run locally

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
cp .env.example .env
.venv/bin/uvicorn app.main:app --reload
# open http://localhost:8000
.venv/bin/pytest -q
```

Without an active Meta account, the API uses a rough colour-based segmenter (`SEGMENTATION_PROVIDER=local`) and mock health and care guides (`HEALTH_PROVIDER=mock`). To use the real models, set `META_API_KEY` and switch both settings to `meta`. Tests always force the local and mock providers.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `SEGMENTATION_PROVIDER` | `local` | `meta` uses SAM 3.1 |
| `HEALTH_PROVIDER` | `mock` | `meta` uses Muse Spark for the health scan and care guides |
| `META_API_KEY` | | Bearer key from dev.meta.ai |
| `META_SAM_MODEL` / `META_VISION_MODEL` | `sam-3.1` / `muse-spark-1.3` | |
| `META_TIMEOUT_S` / `META_MAX_RETRIES` | `120` / `2` | Retries 429 responses, 5xx responses and network errors with exponential backoff |
| `API_KEYS` | empty | Comma-separated client keys for `POST /v1/scans`. Empty disables auth |
| `DEMO_SCANS_PER_HOUR` | `20` | Per IP and per instance. `0` disables the web demo |
| `CARE_GUIDES_PER_HOUR` | `30` | New (uncached) guides per IP |
| `VARIEGATION_THRESHOLDS` | empty | JSON overrides for calibrated thresholds; keep these out of the repo |
| `STORAGE_BACKEND`, `GCS_BUCKET` | `local` | `gcs` in production |
| `REPOSITORY_BACKEND` | `memory` | `firestore` in production |
| `LOG_JSON` | `false` | `true` emits structured logs for Cloud Logging |

## Evaluation and calibration

`scripts/evaluate.py` measures accuracy against your own grading and tunes the variegation thresholds. Keep the photos outside this repo.

```bash
mkdir ~/kanopi-eval && cp eval/labels.example.csv ~/kanopi-eval/labels.csv   # one row per photo or video
# set SEGMENTATION_PROVIDER=meta in .env so the masks come from SAM 3.1
.venv/bin/python -m scripts.evaluate run ~/kanopi-eval        # segments once, caches masks, writes eval/summary.json
.venv/bin/python -m scripts.evaluate calibrate ~/kanopi-eval  # grid search on cached masks, no API calls
```

`run` reports the variegation error in percentage points (mean, median, max, and the share within 5 points), grade agreement, and leaf-count accuracy. `calibrate` reports the error before tuning, in-sample, and with k-fold cross-validation. Quote the cross-validated number. The calibrated thresholds go into the `VARIEGATION_THRESHOLDS` secret, not into git.

## Deploy to Cloud Run

One-time setup:

```bash
export PROJECT_ID=your-project REGION=us-central1
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  firestore.googleapis.com secretmanager.googleapis.com --project $PROJECT_ID
gcloud storage buckets create gs://$PROJECT_ID-kanopi-lens --location $REGION \
  --uniform-bucket-level-access --public-access-prevention --project $PROJECT_ID
gcloud firestore databases create --location $REGION --project $PROJECT_ID

# Browser uploads straight to the bucket (signed URLs), and abandoned uploads expire after a day
gcloud services enable iamcredentials.googleapis.com --project $PROJECT_ID
echo '[{"origin": ["<service URL>"], "method": ["PUT"], "responseHeader": ["Content-Type", "x-goog-content-length-range"], "maxAgeSeconds": 3600}]' > cors.json
gcloud storage buckets update gs://$PROJECT_ID-kanopi-lens --cors-file=cors.json
echo '{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 1, "matchesPrefix": ["uploads/"]}}]}' > lifecycle.json
gcloud storage buckets update gs://$PROJECT_ID-kanopi-lens --lifecycle-file=lifecycle.json

# Runtime service account with least privilege
gcloud iam service-accounts create kanopi-lens --project $PROJECT_ID
SA=kanopi-lens@$PROJECT_ID.iam.gserviceaccount.com
gcloud storage buckets add-iam-policy-binding gs://$PROJECT_ID-kanopi-lens --member serviceAccount:$SA --role roles/storage.objectAdmin
gcloud projects add-iam-policy-binding $PROJECT_ID --member serviceAccount:$SA --role roles/datastore.user
gcloud iam service-accounts add-iam-policy-binding $SA --member serviceAccount:$SA --role roles/iam.serviceAccountTokenCreator  # sign upload URLs

# Secrets (use a production Meta key, separate from development)
printf '%s' "$META_API_KEY" | gcloud secrets create meta-api-key --data-file=- --project $PROJECT_ID
printf '%s' "key-for-kanopi,key-for-partner" | gcloud secrets create kanopi-lens-api-keys --data-file=- --project $PROJECT_ID
printf '%s' '{"white_max_s": 55}' | gcloud secrets create kanopi-lens-thresholds --data-file=- --project $PROJECT_ID  # optional
for s in meta-api-key kanopi-lens-api-keys kanopi-lens-thresholds; do
  gcloud secrets add-iam-policy-binding $s --member serviceAccount:$SA --role roles/secretmanager.secretAccessor --project $PROJECT_ID
done
```

Deploy:

```bash
SERVICE_ACCOUNT=$SA USE_META=1 USE_THRESHOLDS=1 PROJECT_ID=$PROJECT_ID ./deploy.sh
# During judging, keep one warm instance:
MIN_INSTANCES=1 SERVICE_ACCOUNT=$SA USE_META=1 USE_THRESHOLDS=1 PROJECT_ID=$PROJECT_ID ./deploy.sh
```

The deploy runs with 4 vCPU (SAM masks are decoded in parallel worker processes), 4 GiB of memory, concurrency 8, a 900 s timeout (SAM video tracking takes 1–3 minutes) and `MAX_INSTANCES=3` as a cost cap. The container runs as a non-root user.

## Operations

- Structured logs: each scan logs `scan_id`, the leaf count, the variegation %, the health score, the providers and `latency_ms`. Meta API failures are logged with the upstream status, and clients get a generic 502 or 504.
- Health check: `GET /health`. Cloud Run reserves some paths ending in `z`, so this is not `/healthz`.
- Rate limits are held in memory per instance. Use Cloud Armor or a shared store if the API needs strict global limits.

## Project layout

```
app/
  main.py              FastAPI app, lifespan, error handling, web demo route
  static/index.html    Web demo (no build step)
  api/scans.py         HTTP endpoints
  services/
    pipeline.py        decode → segment → measure → overlay → health
    segmentation.py    SAM 3.1 segmenter + local fallback
    variegation.py     per-pixel variegation measurement and grading
    health.py          Muse health scan prompt, schema and scoring
    care.py            localized care guide prompt and schema
    arrival.py         Arrival Check rules and Muse visual comparison
    video.py           video resampling, best view per tracked leaf, leaf gallery
    grading.py         overlay rendering and Grading Card
    meta_client.py     Meta Model API client (Responses API, retries, meta-sam-parser)
    storage.py         local disk / Cloud Storage
    repository.py      memory / Firestore
tests/                 unit and API tests (no network)
```

## Roadmap

- [ ] Verify against the live API with real plant photos
- [ ] Calibrate variegation thresholds against expert-graded photos
- [x] "Proof of Plant" video scan (SAM 3.1 video tracking)
- [ ] Export pre-check checklist
- [ ] Per-client API keys with quotas, usage tracking and billing

## License

Copyright © 2026 Ahmad Hadi Setiawan. All rights reserved. The source code is published for hackathon judging only. No license is granted to use, copy, modify or distribute it.
