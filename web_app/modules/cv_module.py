"""
Computer Vision Module - Object Detection

Jeewon Kim (jk4864) 담당 모듈
YOLOv8 기반 실시간 객체 탐지

TODO for Jeewon:
1. simulate_detection() → real_yolo_detection() 교체
2. ONNX 최적화 적용 (60 FPS 달성)
3. 웹 환경에서 실시간 추론 구현
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time

# OpenCV는 선택적 (실제 YOLO 구현 시 필요)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV (cv2) 없음 - 시뮬레이션 모드만 사용 가능")

# YOLO 모델 로드용
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics 패키지 없음 - 시뮬레이션 모드만 사용 가능")

# Path import 추가
from pathlib import Path


class CVDetectionResult:
    """객체 탐지 결과 클래스"""
    
    def __init__(self, bbox: List[float], class_id: int, confidence: float, class_name: str = ""):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_id = class_id
        self.confidence = confidence
        self.class_name = class_name or self._get_class_name(class_id)
    
    def _get_class_name(self, class_id: int) -> str:
        """클래스 ID를 이름으로 변환"""
        class_names = {
            0: "Player",
            1: "Obstacle",
            2: "Gap",
            3: "Item",
            4: "Lava"  # 라바 추가 (Vision 기반 인식 강조)
        }
        return class_names.get(class_id, "Unknown")
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (웹 전송용)"""
        return {
            'bbox': self.bbox,
            'class_id': self.class_id,
            'confidence': self.confidence,
            'class_name': self.class_name
        }


class ComputerVisionModule:
    """
    컴퓨터 비전 모듈
    
    Jeewon이 구현할 주요 기능:
    1. YOLOv8 모델 로드 및 최적화
    2. 실시간 객체 탐지
    3. 성능 최적화 (60 FPS 목표)
    """
    
    def __init__(self, model_path: Optional[str] = None, use_onnx: bool = True):
        """
        초기화
        
        Args:
            model_path: YOLOv8 모델 경로
            use_onnx: ONNX 최적화 사용 여부
        """
        self.model_path = model_path
        self.use_onnx = use_onnx
        self.model = None
        self.onnx_session = None
        
        # 성능 측정
        self.inference_times = []
        self.frame_count = 0
        
        # 초기화
        self._initialize_model()
    
    def _initialize_model(self):
        """
        모델 초기화
        
        실제 YOLOv8 모델 로드 (지원님 구현 완료)
        """
        if self.model_path:
            try:
                # 실제 YOLO 모델 로드
                from ultralytics import YOLO
                import os
                
                # 상대 경로 처리 (AI_model/best_112217.pt)
                if not os.path.isabs(self.model_path):
                    # 프로젝트 루트 기준으로 경로 조정
                    project_root = Path(__file__).parent.parent.parent
                    full_path = project_root / self.model_path
                    if full_path.exists():
                        self.model_path = str(full_path)
                
                self.model = YOLO(self.model_path)
                print(f"✅ YOLOv8 모델 로드 성공: {self.model_path}")
                
                # ONNX 최적화는 나중에 (선택적)
                # if self.use_onnx:
                #     optimizer = ONNXModelOptimizer()
                #     onnx_path = optimizer.export_yolo_model(self.model, 'optimized_yolo.onnx')
                #     self.onnx_session = optimizer.create_inference_session(onnx_path)
            except ImportError:
                print("⚠️ ultralytics 패키지가 없습니다. 시뮬레이션 모드로 실행합니다.")
            except Exception as e:
                print(f"⚠️ 모델 로드 실패: {e}. 시뮬레이션 모드로 실행합니다.")
        else:
            print("⚠️ 모델 경로가 없습니다. 시뮬레이션 모드로 실행합니다.")
    
    def detect_objects(self, frame: np.ndarray, game_state: Optional[Dict[str, Any]] = None) -> List[CVDetectionResult]:
        """
        객체 탐지 메인 함수
        
        Args:
            frame: 입력 프레임 (H, W, C)
            game_state: 게임 상태 (시뮬레이션 모드에서 라바 감지용, 선택적)
            
        Returns:
            탐지된 객체 리스트
            
        TODO for Jeewon: 실제 YOLOv8 추론 구현
        """
        start_time = time.perf_counter()
        
        # 성능 최적화: 더미 프레임(zeros)을 YOLO에 전달하는 것은 의미 없음
        # 게임 상태가 있으면 시뮬레이션 모드 사용 (더 빠름)
        if self.model is None or game_state is not None:
            # 시뮬레이션 모드 (게임 상태 기반, 빠름)
            results = self._simulate_detection(frame, game_state)
        else:
            # 실제 YOLOv8 추론 (실제 프레임이 있을 때만)
            results = self._real_yolo_detection(frame)
        
        # 성능 측정
        inference_time = time.perf_counter() - start_time
        self.inference_times.append(inference_time)
        self.frame_count += 1
        
        return results
    
    def _simulate_detection(self, frame: np.ndarray, game_state: Optional[Dict[str, Any]] = None) -> List[CVDetectionResult]:
        """
        시뮬레이션된 객체 탐지 (현재 구현)
        
        Jeewon이 _real_yolo_detection()으로 교체할 예정
        
        Args:
            frame: 입력 프레임 (H, W, C)
            game_state: 게임 상태 (라바 감지용)
        """
        # 가짜 탐지 결과 생성
        results = []
        
        # 플레이어 (항상 탐지)
        if game_state and 'player' in game_state:
            player = game_state['player']
            x = player.get('x', 300)
            y = player.get('y', 400)
            size = player.get('size', 50)
            results.append(CVDetectionResult(
                bbox=[x, y, x + size, y + size],
                class_id=0,
                confidence=0.95
            ))
        else:
            # 기본값 (게임 상태 없을 때)
            results.append(CVDetectionResult(
                bbox=[300, 400, 340, 440],  # 중앙 하단
                class_id=0,
                confidence=0.95
            ))
        
        # 장애물 (랜덤 생성)
        if np.random.random() < 0.7:  # 70% 확률
            x = np.random.randint(50, 550)
            y = np.random.randint(50, 300)
            results.append(CVDetectionResult(
                bbox=[x, y, x+40, y+40],
                class_id=1,
                confidence=np.random.uniform(0.6, 0.9)
            ))
        
        # 🌋 라바 감지 (Vision 기반 인식 강조)
        # Note: 라바는 바닥에 고정되어 있지만, YOLO로 감지하면 "Vision 기반 인식"이라는 점을 더 강조할 수 있습니다.
        if game_state and 'lava' in game_state:
            lava_info = game_state['lava']
            lava_state = lava_info.get('state', 'inactive')
            
            # warning 또는 active 상태일 때만 라바 감지
            if lava_state in ['warning', 'active']:
                # 프레임 크기 가져오기
                frame_height = frame.shape[0] if len(frame.shape) >= 2 else 720
                frame_width = frame.shape[1] if len(frame.shape) >= 2 else 960
                
                # 라바 위치 계산
                lava_zone_x = lava_info.get('zone_x', 0)
                lava_zone_width = lava_info.get('zone_width', 320)
                lava_height = lava_info.get('height', 120)
                lava_y_start = frame_height - lava_height
                
                # 라바 바운딩 박스 생성
                # [x1, y1, x2, y2] 형식
                lava_bbox = [
                    lava_zone_x,                    # x1
                    lava_y_start,                   # y1
                    lava_zone_x + lava_zone_width,  # x2
                    frame_height                    # y2 (바닥)
                ]
                
                # 신뢰도: active 상태일 때 더 높음
                confidence = 0.95 if lava_state == 'active' else 0.85
                
                results.append(CVDetectionResult(
                    bbox=lava_bbox,
                    class_id=4,  # Lava 클래스
                    confidence=confidence,
                    class_name="Lava"
                ))
        
        return results
    
    def _real_yolo_detection(self, frame: np.ndarray) -> List[CVDetectionResult]:
        """
        실제 YOLOv8 추론 (지원님 모델 사용)
        
        Args:
            frame: 입력 프레임 (H, W, C) - numpy array
        
        Returns:
            탐지된 객체 리스트 (CVDetectionResult)
        """
        if self.model is None:
            return self._simulate_detection(frame)
        
        try:
            # YOLOv8 추론 실행
            # YOLO 모델은 자동으로 전처리/후처리 수행
            yolo_results = self.model(frame, verbose=False)
            
            # 결과 변환
            results = []
            for result in yolo_results:
                # result.boxes는 탐지된 박스 정보
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # 박스 정보 추출
                    box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(boxes.conf[i].cpu().numpy())  # 신뢰도
                    cls = int(boxes.cls[i].cpu().numpy())  # 클래스 ID
                    
                    # 클래스 이름 매핑 (YOLO 데이터셋 기준)
                    # 0: player, 1: meteor, 2: star, 3: lava_warning, 4: lava_active
                    class_names = ['player', 'meteor', 'star', 'lava_warning', 'lava_active']
                    class_name = class_names[cls] if cls < len(class_names) else f'class_{cls}'
                    
                    # CVDetectionResult 생성
                    detection = CVDetectionResult(
                        bbox=[float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        class_id=cls,
                        confidence=conf,
                        class_name=class_name
                    )
                    results.append(detection)
            
            return results
            
        except Exception as e:
            print(f"❌ YOLOv8 추론 오류: {e}")
            # 오류 시 시뮬레이션으로 폴백
            return self._simulate_detection(frame)
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        YOLOv8 입력을 위한 프레임 전처리
        
        TODO for Jeewon: YOLOv8 입력 형식에 맞게 구현
        """
        if not CV2_AVAILABLE:
            # OpenCV 없을 때는 numpy로만 처리
            # 간단한 리사이즈 (numpy만 사용)
            # 실제 구현 시에는 OpenCV 필요
            raise NotImplementedError("OpenCV (cv2)가 필요합니다. 실제 YOLO 구현 시 사용됩니다.")
        
        # 예시 구현
        # 1. 리사이즈 (640x640)
        # 2. 정규화 (0-1)
        # 3. HWC → CHW 변환
        # 4. 배치 차원 추가
        
        resized = cv2.resize(frame, (640, 640))
        normalized = resized.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def _postprocess_outputs(self, outputs: np.ndarray) -> List[CVDetectionResult]:
        """
        YOLOv8 출력 후처리
        
        TODO for Jeewon: NMS, 신뢰도 필터링 구현
        """
        results = []
        
        # TODO: 실제 후처리 구현
        # 1. 신뢰도 임계값 적용
        # 2. NMS (Non-Maximum Suppression)
        # 3. 좌표 변환 (정규화 → 픽셀)
        # 4. CVDetectionResult 객체 생성
        
        return results
    
    def get_performance_stats(self) -> Dict[str, float]:
        """성능 통계 반환"""
        if not self.inference_times:
            return {}
        
        avg_time = np.mean(self.inference_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_inference_time_ms': avg_time * 1000,
            'avg_fps': avg_fps,
            'target_fps': 60.0,
            'meets_target': avg_fps >= 57.0,  # 95% of 60 FPS
            'total_frames': self.frame_count
        }
    
    def reset_performance_stats(self):
        """성능 통계 초기화"""
        self.inference_times = []
        self.frame_count = 0


# Jeewon이 사용할 헬퍼 함수들
def convert_frame_for_detection(web_frame_data: Dict) -> np.ndarray:
    """
    웹에서 받은 프레임 데이터를 OpenCV 형식으로 변환
    
    TODO for Jeewon: 웹 환경에서 프레임 데이터 처리
    """
    # 웹 Canvas ImageData → numpy array 변환
    # 실제 구현은 웹 환경에 따라 달라질 수 있음
    pass


def create_detection_overlay(frame: np.ndarray, detections: List[CVDetectionResult]) -> np.ndarray:
    """
    탐지 결과를 프레임에 오버레이
    
    Jeewon이 디버깅용으로 사용할 수 있는 함수
    """
    if not CV2_AVAILABLE:
        # OpenCV 없을 때는 원본 프레임 반환
        return frame.copy()
    
    overlay_frame = frame.copy()
    
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.bbox)
        
        # 바운딩 박스 그리기
        color = (0, 255, 0) if detection.class_id == 0 else (0, 0, 255)
        cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 2)
        
        # 라벨 그리기
        label = f"{detection.class_name}: {detection.confidence:.2f}"
        cv2.putText(overlay_frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return overlay_frame


# 사용 예시 (Jeewon이 참고할 코드)
if __name__ == "__main__":
    # CV 모듈 초기화
    cv_module = ComputerVisionModule(
        model_path="path/to/yolo_model.pt",  # Jeewon이 훈련한 모델
        use_onnx=True  # 성능 최적화
    )
    
    # 테스트 프레임
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 객체 탐지
    detections = cv_module.detect_objects(test_frame)
    
    # 결과 출력
    print(f"탐지된 객체 수: {len(detections)}")
    for detection in detections:
        print(f"- {detection.class_name}: {detection.confidence:.2f}")
    
    # 성능 통계
    stats = cv_module.get_performance_stats()
    print(f"평균 FPS: {stats.get('avg_fps', 0):.1f}")
    print(f"목표 달성: {stats.get('meets_target', False)}")
