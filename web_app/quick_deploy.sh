#!/bin/bash
# 빠른 GCP 배포 스크립트

set -e

PROJECT_ID="vision-final-478501"
REGION="us-central1"
SERVICE_NAME="distilled-vision-agent"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Distilled Vision Agent - GCP 배포 시작"
echo "프로젝트: $PROJECT_ID"
echo "서비스: $SERVICE_NAME"
echo

# 1. 프로젝트 설정
echo "📋 프로젝트 설정..."
gcloud config set project $PROJECT_ID

# 2. Docker 이미지 빌드
echo "🏗️ Docker 이미지 빌드..."
docker build -t $IMAGE_NAME .

# 3. Container Registry에 푸시
echo "📤 이미지 푸시..."
docker push $IMAGE_NAME

# 4. Cloud Run에 배포
echo "☁️ Cloud Run 배포..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --concurrency 80 \
    --timeout 300 \
    --port 5000 \
    --set-env-vars ENVIRONMENT=production,GCS_BUCKET_NAME=distilled-vision-game-data

# 5. 서비스 URL 출력
echo
echo "✅ 배포 완료!"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')
echo "🌐 게임 URL: $SERVICE_URL"
echo
echo "🎮 브라우저에서 접속하여 게임을 플레이하세요!"
