#!/usr/bin/env python3
"""
GCP Cloud Storage Manager
리더보드 및 게임 데이터 영구 저장
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Cloud Storage import (로컬에서 없으면 fallback)
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("⚠️ google-cloud-storage not installed. Using local storage fallback.")

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Cloud Storage 관리자
    
    환경:
    - 로컬 개발: 파일 시스템 사용
    - GCP 배포: Cloud Storage 사용
    """
    
    def __init__(self, 
                 bucket_name: str = None,
                 local_data_dir: str = "./data",
                 use_gcs: bool = None):
        """
        Args:
            bucket_name: GCS 버킷 이름
            local_data_dir: 로컬 저장 경로 (fallback용)
            use_gcs: Cloud Storage 사용 여부 (None이면 자동 감지)
        """
        self.bucket_name = bucket_name or os.getenv('GCS_BUCKET_NAME', 'distilled-vision-game-data')
        self.local_data_dir = Path(local_data_dir)
        
        # Cloud Storage 사용 여부 결정
        if use_gcs is None:
            # 환경 변수로 판단 (프로덕션에서는 ENVIRONMENT=production)
            self.use_gcs = (os.getenv('ENVIRONMENT') == 'production' and GCS_AVAILABLE)
        else:
            self.use_gcs = use_gcs and GCS_AVAILABLE
        
        # GCS 클라이언트 초기화
        self.client = None
        self.bucket = None
        
        if self.use_gcs:
            try:
                self.client = storage.Client()
                self.bucket = self.client.bucket(self.bucket_name)
                logger.info(f"✅ Cloud Storage 연결 성공: gs://{self.bucket_name}")
                print(f"☁️ Cloud Storage 사용: gs://{self.bucket_name}")
            except Exception as e:
                logger.warning(f"⚠️ Cloud Storage 연결 실패: {e}")
                print(f"⚠️ Cloud Storage 연결 실패. 로컬 스토리지 사용: {e}")
                self.use_gcs = False
        
        # 로컬 디렉토리 생성 (fallback)
        if not self.use_gcs:
            self.local_data_dir.mkdir(parents=True, exist_ok=True)
            print(f"💾 로컬 스토리지 사용: {self.local_data_dir.absolute()}")
    
    # ========== 리더보드 관리 ==========
    
    def load_leaderboard(self) -> Dict:
        """
        리더보드 로드
        
        Returns:
            {'scores': [...]} 형식의 딕셔너리
        """
        if self.use_gcs:
            return self._load_leaderboard_gcs()
        else:
            return self._load_leaderboard_local()
    
    def save_leaderboard(self, leaderboard: Dict) -> bool:
        """
        리더보드 저장
        
        Args:
            leaderboard: {'scores': [...]} 형식
            
        Returns:
            성공 여부
        """
        if self.use_gcs:
            return self._save_leaderboard_gcs(leaderboard)
        else:
            return self._save_leaderboard_local(leaderboard)
    
    def add_score(self, player_name: str, score: int, survival_time: float, 
                  mode: str, session_id: str) -> Dict:
        """
        리더보드에 점수 추가 (기존 로직 유지)
        
        Returns:
            업데이트된 리더보드
        """
        leaderboard = self.load_leaderboard()
        
        leaderboard['scores'].append({
            'player': player_name,
            'score': score,
            'time': round(survival_time, 2),
            'mode': mode,
            'date': datetime.now().isoformat(),
            'session_id': session_id
        })
        
        # 점수순 정렬 (내림차순)
        leaderboard['scores'].sort(key=lambda x: x['score'], reverse=True)
        
        # 상위 100개만 유지
        leaderboard['scores'] = leaderboard['scores'][:100]
        
        self.save_leaderboard(leaderboard)
        return leaderboard
    
    # ========== GCS 구현 ==========
    
    def _load_leaderboard_gcs(self) -> Dict:
        """Cloud Storage에서 리더보드 로드"""
        try:
            blob = self.bucket.blob('leaderboard/leaderboard.json')
            
            if not blob.exists():
                logger.info("리더보드 파일 없음. 새로 생성.")
                return {'scores': []}
            
            data = blob.download_as_text()
            return json.loads(data)
        
        except Exception as e:
            logger.error(f"GCS 리더보드 로드 실패: {e}")
            return {'scores': []}
    
    def _save_leaderboard_gcs(self, leaderboard: Dict) -> bool:
        """Cloud Storage에 리더보드 저장"""
        try:
            blob = self.bucket.blob('leaderboard/leaderboard.json')
            blob.upload_from_string(
                json.dumps(leaderboard, indent=2, ensure_ascii=False),
                content_type='application/json'
            )
            logger.info("✅ GCS 리더보드 저장 완료")
            return True
        
        except Exception as e:
            logger.error(f"❌ GCS 리더보드 저장 실패: {e}")
            return False
    
    # ========== 로컬 구현 (Fallback) ==========
    
    def _load_leaderboard_local(self) -> Dict:
        """로컬 파일에서 리더보드 로드"""
        leaderboard_file = self.local_data_dir / 'leaderboard.json'
        
        if leaderboard_file.exists():
            with open(leaderboard_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return {'scores': []}
    
    def _save_leaderboard_local(self, leaderboard: Dict) -> bool:
        """로컬 파일에 리더보드 저장"""
        try:
            leaderboard_file = self.local_data_dir / 'leaderboard.json'
            with open(leaderboard_file, 'w', encoding='utf-8') as f:
                json.dump(leaderboard, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 로컬 리더보드 저장: {leaderboard_file}")
            return True
        
        except Exception as e:
            logger.error(f"❌ 로컬 리더보드 저장 실패: {e}")
            return False
    
    # ========== 게임 세션 저장 ==========
    
    def save_gameplay_session(self, session_data: Dict, session_id: str) -> str:
        """
        게임 세션 메타데이터 저장
        
        Args:
            session_data: 세션 정보 딕셔너리
            session_id: 세션 ID
            
        Returns:
            저장 경로
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp}_{session_id[:8]}.json"
        
        if self.use_gcs:
            return self._save_session_gcs(session_data, filename)
        else:
            return self._save_session_local(session_data, filename)
    
    def _save_session_gcs(self, session_data: Dict, filename: str) -> str:
        """Cloud Storage에 세션 저장"""
        try:
            # 날짜별 폴더 구조
            date_folder = datetime.now().strftime("%Y-%m-%d")
            blob_path = f"gameplay/sessions/{date_folder}/{filename}"
            
            blob = self.bucket.blob(blob_path)
            blob.upload_from_string(
                json.dumps(session_data, indent=2, ensure_ascii=False),
                content_type='application/json'
            )
            
            logger.info(f"✅ GCS 세션 저장: {blob_path}")
            return f"gs://{self.bucket_name}/{blob_path}"
        
        except Exception as e:
            logger.error(f"❌ GCS 세션 저장 실패: {e}")
            return ""
    
    def _save_session_local(self, session_data: Dict, filename: str) -> str:
        """로컬 파일에 세션 저장"""
        try:
            # 날짜별 폴더 구조
            date_folder = datetime.now().strftime("%Y-%m-%d")
            session_dir = self.local_data_dir / 'gameplay' / 'sessions' / date_folder
            session_dir.mkdir(parents=True, exist_ok=True)
            
            session_file = session_dir / filename
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 로컬 세션 저장: {session_file}")
            return str(session_file)
        
        except Exception as e:
            logger.error(f"❌ 로컬 세션 저장 실패: {e}")
            return ""
    
    # ========== 이미지 저장 (Phase 2 확장) ==========
    
    def save_frame_image(self, image_data: bytes, session_id: str, frame_number: int) -> str:
        """
        게임 프레임 이미지 저장 (PNG)
        
        Args:
            image_data: PNG 이미지 바이트 데이터
            session_id: 세션 ID
            frame_number: 프레임 번호
            
        Returns:
            저장 경로
        """
        if self.use_gcs:
            return self._save_frame_gcs(image_data, session_id, frame_number)
        else:
            return self._save_frame_local(image_data, session_id, frame_number)
    
    def _save_frame_gcs(self, image_data: bytes, session_id: str, frame_number: int) -> str:
        """Cloud Storage에 프레임 이미지 저장"""
        try:
            date_folder = datetime.now().strftime("%Y-%m-%d")
            blob_path = f"gameplay/frames/{date_folder}/{session_id[:8]}/frame_{frame_number:05d}.png"
            
            blob = self.bucket.blob(blob_path)
            blob.upload_from_string(image_data, content_type='image/png')
            
            logger.info(f"✅ GCS 프레임 저장: {blob_path}")
            return f"gs://{self.bucket_name}/{blob_path}"
        
        except Exception as e:
            logger.error(f"❌ GCS 프레임 저장 실패: {e}")
            return ""
    
    def _save_frame_local(self, image_data: bytes, session_id: str, frame_number: int) -> str:
        """로컬 파일에 프레임 이미지 저장"""
        try:
            date_folder = datetime.now().strftime("%Y-%m-%d")
            frames_dir = self.local_data_dir / 'gameplay' / 'frames' / date_folder / session_id[:8]
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            frame_file = frames_dir / f"frame_{frame_number:05d}.png"
            with open(frame_file, 'wb') as f:
                f.write(image_data)
            
            logger.info(f"💾 로컬 프레임 저장: {frame_file}")
            return str(frame_file)
        
        except Exception as e:
            logger.error(f"❌ 로컬 프레임 저장 실패: {e}")
            return ""
    
    # ========== 통계 ==========
    
    def get_stats(self) -> Dict:
        """통계 정보 반환"""
        leaderboard = self.load_leaderboard()
        scores = leaderboard['scores']
        
        if not scores:
            return {
                'total_games': 0,
                'avg_score': 0,
                'highest_score': 0,
                'total_playtime': 0,
                'human_games': 0,
                'ai_games': 0
            }
        
        return {
            'total_games': len(scores),
            'avg_score': round(sum(s['score'] for s in scores) / len(scores), 2),
            'highest_score': scores[0]['score'] if scores else 0,
            'total_playtime': round(sum(s['time'] for s in scores), 2),
            'human_games': len([s for s in scores if s['mode'] == 'human']),
            'ai_games': len([s for s in scores if s['mode'] == 'ai'])
        }


# ========== 싱글톤 인스턴스 (앱 전역 사용) ==========

_storage_manager_instance = None


def get_storage_manager() -> StorageManager:
    """
    StorageManager 싱글톤 인스턴스 반환
    
    앱 시작 시 한 번만 초기화됨
    """
    global _storage_manager_instance
    
    if _storage_manager_instance is None:
        _storage_manager_instance = StorageManager()
    
    return _storage_manager_instance


# ========== 테스트용 ==========

if __name__ == '__main__':
    # 테스트
    print("🧪 Storage Manager 테스트\n")
    
    sm = StorageManager(use_gcs=False)  # 로컬 테스트
    
    # 리더보드 로드
    print("1️⃣ 리더보드 로드:")
    leaderboard = sm.load_leaderboard()
    print(f"   현재 점수 개수: {len(leaderboard['scores'])}")
    
    # 점수 추가 테스트
    print("\n2️⃣ 테스트 점수 추가:")
    sm.add_score("TestPlayer", 100, 30.5, "human", "test-session-123")
    
    # 재로드
    print("\n3️⃣ 재로드 확인:")
    leaderboard = sm.load_leaderboard()
    print(f"   업데이트된 점수 개수: {len(leaderboard['scores'])}")
    
    # 통계
    print("\n4️⃣ 통계:")
    stats = sm.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ 테스트 완료!")

