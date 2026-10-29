#!/usr/bin/env bash
# Deploy Kanopi Lens API to Cloud Run from source (Cloud Build builds the Dockerfile).
#
#   PROJECT_ID=my-project ./deploy.sh
#
# Optional: REGION, SERVICE, BUCKET, SERVICE_ACCOUNT,
#           MIN_INSTANCES (set 1 during judging), MAX_INSTANCES (cost cap),
#           USE_META=1 (switch providers to Meta and mount the Meta API key secret),
#           USE_THRESHOLDS=1 (mount calibrated variegation thresholds from Secret Manager).
# Client API keys are always mounted from the kanopi-lens-api-keys secret.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-kanopi-lens-api}"
BUCKET="${BUCKET:-${PROJECT_ID}-kanopi-lens}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-3}"

env_vars="STORAGE_BACKEND=gcs,GCS_BUCKET=${BUCKET},REPOSITORY_BACKEND=firestore,LOG_JSON=true,DEMO_SCANS_PER_HOUR=${DEMO_SCANS_PER_HOUR:-10}"
secrets=("API_KEYS=kanopi-lens-api-keys:latest")
extra_args=()

if [[ "${USE_META:-0}" == "1" ]]; then
  env_vars+=",SEGMENTATION_PROVIDER=meta,HEALTH_PROVIDER=meta,META_API_BASE_URL=${META_API_BASE_URL:-https://api.meta.ai/v1}"
  secrets+=("META_API_KEY=meta-api-key:latest")
fi
if [[ "${USE_THRESHOLDS:-0}" == "1" ]]; then
  secrets+=("VARIEGATION_THRESHOLDS=kanopi-lens-thresholds:latest")
fi
extra_args+=(--set-secrets "$(IFS=,; echo "${secrets[*]}")")
if [[ -n "${SERVICE_ACCOUNT:-}" ]]; then
  extra_args+=(--service-account "$SERVICE_ACCOUNT")
fi

gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --source . \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 4 \
  --concurrency 8 \
  --timeout 900 \
  --min-instances "$MIN_INSTANCES" \
  --max-instances "$MAX_INSTANCES" \
  --set-env-vars "$env_vars" \
  ${extra_args[@]+"${extra_args[@]}"}
