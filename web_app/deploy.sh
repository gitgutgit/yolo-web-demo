#!/bin/bash
# GCP Cloud Run 배포 스크립트

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 프로젝트 설정
PROJECT_ID=${1:-"your-gcp-project-id"}
REGION=${2:-"us-central1"}
SERVICE_NAME="distilled-vision-agent"

echo -e "${BLUE}🚀 Distilled Vision Agent - GCP Cloud Run 배포${NC}"
echo "=================================================="
echo -e "프로젝트 ID: ${YELLOW}$PROJECT_ID${NC}"
echo -e "리전: ${YELLOW}$REGION${NC}"
echo -e "서비스명: ${YELLOW}$SERVICE_NAME${NC}"
echo

# GCP 프로젝트 설정 확인
echo -e "${BLUE}📋 GCP 프로젝트 설정 확인...${NC}"
gcloud config set project $PROJECT_ID

# 필요한 API 활성화
echo -e "${BLUE}🔧 필요한 GCP API 활성화...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Docker 이미지 빌드
echo -e "${BLUE}🏗️ Docker 이미지 빌드...${NC}"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"
docker build -t $IMAGE_NAME .

# Container Registry에 푸시
echo -e "${BLUE}📤 Container Registry에 이미지 푸시...${NC}"
docker push $IMAGE_NAME

# Cloud Run에 배포
echo -e "${BLUE}☁️ Cloud Run에 배포...${NC}"
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --concurrency 80 \
    --timeout 300 \
    --port 8080 \
    --set-env-vars ENVIRONMENT=production

# 배포 완료 정보
echo
echo -e "${GREEN}✅ 배포 완료!${NC}"
echo "=================================================="

# 서비스 URL 가져오기
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')

echo -e "${GREEN}🌐 서비스 URL: ${YELLOW}$SERVICE_URL${NC}"
echo -e "${GREEN}📊 모니터링: ${YELLOW}https://console.cloud.google.com/run/detail/$REGION/$SERVICE_NAME${NC}"
echo

# 브라우저에서 열기 (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${BLUE}🌍 브라우저에서 열기...${NC}"
    open $SERVICE_URL
fi

echo -e "${GREEN}🎉 배포가 성공적으로 완료되었습니다!${NC}"
echo
echo "사용법:"
echo "  - Human Mode: 직접 게임 플레이"
echo "  - AI Mode: AI 플레이 관찰"
echo "  - 실시간 성능 모니터링"
echo "  - 리더보드 기능"
