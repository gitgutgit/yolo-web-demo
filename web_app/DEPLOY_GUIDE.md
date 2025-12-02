# 🚀 GCP Cloud Run 배포 가이드

## ✅ 사전 준비 확인

- [x] 로컬에서 테스트 완료
- [x] 이미지 프레임 저장 확인
- [x] GCP 프로젝트 준비 (`vision-final-478501`)

---

## 📋 배포 단계

### **Step 1: GCP 프로젝트 설정**

```bash
# 프로젝트 설정
gcloud config set project vision-final-478501

# 현재 프로젝트 확인
gcloud config get-value project
```

**예상 출력**: `vision-final-478501`

---

### **Step 2: Cloud Storage 버킷 생성**

```bash
# 버킷 생성 (이미 존재하면 에러 무시)
gsutil mb -p vision-final-478501 -c STANDARD -l us-central1 gs://distilled-vision-game-data

# 버킷 확인
gsutil ls gs://distilled-vision-game-data/
```

**예상 출력**: 버킷이 비어있거나 기존 파일 표시

---

### **Step 3: 필요한 API 활성화**

```bash
# Cloud Build API
gcloud services enable cloudbuild.googleapis.com

# Cloud Run API
gcloud services enable run.googleapis.com

# Container Registry API
gcloud services enable containerregistry.googleapis.com

# Cloud Storage API
gcloud services enable storage.googleapis.com
```

**예상 출력**: `Operation "..." finished successfully.`

---

### **Step 4: Cloud Run 배포**

```bash
cd "/Users/aidesigner/Columbia Univ Course/deeplearningvision4995/final_project/web_app"

# 배포 스크립트 실행
chmod +x quick_deploy.sh
./quick_deploy.sh
```

**배포 과정** (5-10분 소요):

1. Docker 이미지 빌드
2. Container Registry에 푸시
3. Cloud Run에 배포
4. 서비스 URL 출력

---

### **Step 5: 서비스 계정 권한 설정**

배포가 완료되면 Cloud Storage 권한을 부여해야 합니다:

```bash
# Cloud Run 서비스 계정 확인
SERVICE_EMAIL=$(gcloud run services describe distilled-vision-agent \
  --region us-central1 \
  --format="value(spec.template.spec.serviceAccount)")

echo "Service Account: $SERVICE_EMAIL"

# Storage Admin 권한 부여
gcloud projects add-iam-policy-binding vision-final-478501 \
  --member="serviceAccount:${SERVICE_EMAIL}" \
  --role="roles/storage.admin"
```

**예상 출력**:

```
Updated IAM policy for project [vision-final-478501].
```

---

### **Step 6: 배포 확인**

```bash
# 서비스 URL 확인
gcloud run services describe distilled-vision-agent \
  --region us-central1 \
  --format="value(status.url)"
```

**예상 출력**:

```
https://distilled-vision-agent-XXXXX-uc.a.run.app
```

브라우저에서 URL을 열어 게임을 플레이해보세요!

---

## 🧪 배포 후 테스트

### 1️⃣ **웹사이트 접속**

```
https://distilled-vision-agent-XXXXX-uc.a.run.app
```

### 2️⃣ **게임 플레이**

- Human Mode로 게임 플레이
- 리더보드에서 "Show More" 버튼 테스트
- 콘솔 로그에서 프레임 캡처 확인

### 3️⃣ **Cloud Storage 확인**

```bash
# 리더보드 확인
gsutil cat gs://distilled-vision-game-data/leaderboard/leaderboard.json

# 게임 세션 확인
gsutil ls gs://distilled-vision-game-data/gameplay/sessions/

# 이미지 프레임 확인
gsutil ls gs://distilled-vision-game-data/gameplay/frames/
```

### 4️⃣ **서버 재시작 테스트**

```bash
# 서비스 재배포 (빠른 재시작)
./quick_deploy.sh
```

브라우저에서 새로고침 → **리더보드 데이터가 유지되는지 확인!** ✅

---

## 🔧 문제 해결

### Q1. 배포 실패: "Permission denied"

```bash
# Docker 권한 확인
docker ps

# gcloud 인증 재설정
gcloud auth login
gcloud auth configure-docker
```

### Q2. 웹사이트 접속 안 됨

```bash
# Cloud Run 로그 확인
gcloud run services logs read distilled-vision-agent --region us-central1 --limit 50
```

### Q3. Cloud Storage 접근 안 됨

```bash
# 서비스 계정 권한 재확인
gcloud projects get-iam-policy vision-final-478501 \
  --flatten="bindings[].members" \
  --filter="bindings.role:roles/storage.admin"
```

---

## 📊 모니터링

### Cloud Run 대시보드

```
https://console.cloud.google.com/run/detail/us-central1/distilled-vision-agent/metrics
```

### Cloud Storage 대시보드

```
https://console.cloud.google.com/storage/browser/distilled-vision-game-data
```

---

## 💰 비용 예상

### Cloud Run (무료 할당량 내)

- 첫 200만 요청/월: 무료
- 360,000 GB-초/월: 무료

### Cloud Storage

- 5 GB: 무료
- 초과분: ~$0.02/GB/월

**예상 월 비용**: **$0 ~ $2** (매우 저렴!)

---

## 🎉 배포 완료 후

이제 다음을 할 수 있습니다:

1. ✅ **어디서든 게임 플레이** (URL 공유)
2. ✅ **리더보드 영구 저장** (서버 재시작해도 유지)
3. ✅ **팀원들과 데이터 공유** (GCS에서 다운로드)
4. ✅ **이미지 데이터 수집** (CV 모델 훈련용)

---

## 📝 다음 단계

1. **팀원들에게 URL 공유**

   ```
   게임: https://distilled-vision-agent-XXXXX-uc.a.run.app
   ```

2. **데이터 다운로드 방법 공유**

   ```bash
   # Jay (YOLO 훈련)
   gsutil -m cp -r gs://distilled-vision-game-data/gameplay/frames ./training_data/

   # Chloe (RL 훈련)
   gsutil -m cp -r gs://distilled-vision-game-data/gameplay/sessions ./rl_training/
   ```

3. **모델 훈련 & 배포**
   - Jay: YOLO 객체 감지 모델
   - Chloe: RL 에이전트
   - Larry: 데이터 증강 & 품질 관리

---

## 🚨 중요 참고사항

### 환경 변수 (자동 설정됨)

- `ENVIRONMENT=production` → Cloud Storage 사용
- `GCS_BUCKET_NAME=distilled-vision-game-data`

### 보안

- Cloud Run은 HTTPS 자동 적용
- 서비스 계정으로 안전한 인증
- 버킷 접근 권한 제어

### 로그

```bash
# 실시간 로그 보기
gcloud run services logs tail distilled-vision-agent --region us-central1
```

---

**배포 준비 완료!** 터미널에서 위 명령어들을 순서대로 실행하세요! 🚀
