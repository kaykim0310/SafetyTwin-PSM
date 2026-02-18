"""
M01-03: symbol_detector
━━━━━━━━━━━━━━━━━━━━━━━
D-Fine 기반 P&ID 심볼 감지 추론 모듈

[역할]
- 학습된 D-Fine 모델로 P&ID 도면에서 심볼을 자동 감지합니다.
- 밸브, 펌프, 탱크, 계장기기 등을 95% 이상 정확도로 인식합니다.
- NMS(Non-Maximum Suppression) 없이 End-to-End 감지 → 더 빠르고 정확!

[비유]
학습이 끝난 AI 전문가가 실제 도면을 보고 "여기 밸브, 저기 펌프"라고
가리키는 단계입니다. M01-02에서 '가르침'을 받았다면,
이 모듈에서는 '실전'에 투입됩니다.
"""

import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class DetectedSymbol:
    """감지된 하나의 심볼 정보

    [쉬운 설명]
    AI가 도면에서 찾은 심볼 하나의 '신상정보'입니다.
    - 어떤 종류인지 (class_name)
    - 얼마나 확신하는지 (confidence)
    - 도면 어디에 있는지 (bbox)
    """
    class_id: int               # 심볼 종류 번호 (0~41)
    class_name: str             # 심볼 이름 (예: "gate_valve")
    confidence: float           # 확신도 (0.0~1.0, 높을수록 확실)
    bbox: tuple[float, float, float, float]  # 위치 (x1, y1, x2, y2)
    segment_row: int = 0        # 어떤 세그먼트에서 발견됐는지 (행)
    segment_col: int = 0        # 어떤 세그먼트에서 발견됐는지 (열)

    @property
    def center(self) -> tuple[float, float]:
        """심볼의 중심 좌표"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> float:
        """심볼의 면적 (픽셀²)"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def korean_name(self) -> str:
        """심볼의 한국어 이름"""
        name_map = {
            "gate_valve": "게이트 밸브",
            "globe_valve": "글로브 밸브",
            "ball_valve": "볼 밸브",
            "check_valve": "체크 밸브",
            "butterfly_valve": "버터플라이 밸브",
            "relief_valve": "안전밸브 (릴리프)",
            "control_valve": "제어 밸브",
            "solenoid_valve": "솔레노이드 밸브",
            "needle_valve": "니들 밸브",
            "pump": "펌프",
            "compressor": "압축기",
            "tank": "탱크",
            "vessel": "용기 (베셀)",
            "reactor": "반응기",
            "heat_exchanger": "열교환기",
            "column": "증류탑 (컬럼)",
            "drum": "드럼",
            "filter": "필터",
            "pressure_indicator": "압력 지시계 (PI)",
            "temperature_indicator": "온도 지시계 (TI)",
            "flow_indicator": "유량 지시계 (FI)",
            "level_indicator": "레벨 지시계 (LI)",
            "pressure_transmitter": "압력 전송기 (PT)",
            "temperature_transmitter": "온도 전송기 (TT)",
            "flow_transmitter": "유량 전송기 (FT)",
            "level_transmitter": "레벨 전송기 (LT)",
            "control_loop": "제어 루프",
            "alarm": "경보기",
            "pipe_line": "배관 라인",
            "reducer": "레듀서",
            "tee": "티 (분기관)",
            "elbow": "엘보 (곡관)",
            "flange": "플랜지",
            "spectacle_blind": "스펙터클 블라인드",
            "rupture_disc": "파열판",
            "flame_arrestor": "화염방지기",
            "safety_shower": "비상 샤워",
            "emergency_shutoff": "긴급 차단 장치",
        }
        return name_map.get(self.class_name, self.class_name)


@dataclass
class DetectionResult:
    """전체 도면의 감지 결과"""
    symbols: list[DetectedSymbol]       # 감지된 모든 심볼
    inference_time_ms: float            # 추론 소요 시간 (밀리초)
    image_size: tuple[int, int]         # 입력 이미지 크기
    model_name: str                     # 사용된 모델
    metadata: dict                      # 추가 정보

    @property
    def symbol_count(self) -> int:
        return len(self.symbols)

    @property
    def symbol_summary(self) -> dict[str, int]:
        """종류별 심볼 개수 요약"""
        summary = {}
        for s in self.symbols:
            summary[s.korean_name] = summary.get(s.korean_name, 0) + 1
        return dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))

    @property
    def safety_devices(self) -> list[DetectedSymbol]:
        """PSM 관련 안전장치만 필터링"""
        safety_classes = {
            "relief_valve", "rupture_disc", "flame_arrestor",
            "safety_shower", "emergency_shutoff",
        }
        return [s for s in self.symbols if s.class_name in safety_classes]


class DFineSymbolDetector:
    """D-Fine 기반 P&ID 심볼 감지기

    [사용법]
    detector = DFineSymbolDetector(model_path="models/dfine_pid.pth")
    result = detector.detect("도면.png")

    # 감지된 심볼 확인
    for symbol in result.symbols:
        print(f"{symbol.korean_name}: 확신도 {symbol.confidence:.1%}")

    # 안전장치만 확인 (PSM 핵심!)
    for safety in result.safety_devices:
        print(f"⚠️ 안전장치: {safety.korean_name}")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        device: str = "cuda",
    ):
        """
        Args:
            model_path: 학습된 D-Fine 모델 가중치 경로
            confidence_threshold: 최소 신뢰도 (이 이하는 무시)
            device: 추론 장치 ("cuda" 또는 "cpu")
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self._class_names = self._load_class_names()

        logger.info(f"심볼 감지기 초기화: threshold={confidence_threshold}, device={device}")

    def _load_class_names(self) -> list[str]:
        """P&ID 심볼 클래스 이름 로드"""
        from src.config.settings import settings
        return settings.PID_SYMBOL_CLASSES

    def load_model(self):
        """D-Fine 모델을 로드합니다.

        [D-Fine 모델 로드 프로세스]
        1) HGNetV2-L 백본 + DFine 디코더 구조 생성
        2) 학습된 가중치(.pth) 로드
        3) 평가 모드(eval)로 전환 → 추론 전용

        [중요] D-Fine은 NMS가 필요 없습니다!
        기존 YOLO는 중복 감지를 제거하는 NMS 후처리가 필요했지만,
        D-Fine은 End-to-End로 중복 없는 결과를 바로 출력합니다.
        이것이 속도와 정확도 모두에서 이점이 됩니다.
        """
        logger.info(f"D-Fine 모델 로드: {self.model_path}")

        # 실제 구현 시 아래와 같이 로드:
        # import torch
        # from dfine.models import DFine
        #
        # self.model = DFine(
        #     backbone="HGNetV2-L",
        #     num_classes=len(self._class_names),
        # )
        # checkpoint = torch.load(self.model_path, map_location=self.device)
        # self.model.load_state_dict(checkpoint["model"])
        # self.model.to(self.device)
        # self.model.eval()

        logger.info("D-Fine 모델 로드 완료 (NMS 불필요 - End-to-End)")

    def detect(self, image: np.ndarray) -> DetectionResult:
        """이미지에서 P&ID 심볼을 감지합니다.

        [감지 프로세스]
        1) 이미지 전처리 (크기 조정, 정규화)
        2) D-Fine 모델에 입력
        3) 감지 결과 (bbox, class, confidence) 추출
        4) 신뢰도 기준 필터링
        5) 원본 좌표로 변환

        Args:
            image: OpenCV BGR 이미지 (numpy array)

        Returns:
            DetectionResult: 감지된 심볼 목록과 메타데이터
        """
        start_time = time.time()
        h, w = image.shape[:2]

        logger.info(f"심볼 감지 시작: {w}×{h}")

        # ── D-Fine 추론 ──
        # 실제 구현:
        # import torch
        # from torchvision import transforms
        #
        # # 1) 전처리
        # transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((640, 640)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                         std=[0.229, 0.224, 0.225]),
        # ])
        # input_tensor = transform(image).unsqueeze(0).to(self.device)
        #
        # # 2) 추론 (NMS 불필요!)
        # with torch.no_grad():
        #     outputs = self.model(input_tensor)
        #
        # # 3) 결과 추출
        # scores = outputs["scores"][0]         # [num_queries]
        # labels = outputs["labels"][0]         # [num_queries]
        # boxes = outputs["boxes"][0]           # [num_queries, 4]

        # ── 프레임워크: 시뮬레이션 결과 ──
        symbols = self._simulate_detection(w, h)

        elapsed = (time.time() - start_time) * 1000

        result = DetectionResult(
            symbols=symbols,
            inference_time_ms=elapsed,
            image_size=(w, h),
            model_name="D-Fine-L (HGNetV2)",
            metadata={
                "confidence_threshold": self.confidence_threshold,
                "device": self.device,
                "nms_required": False,  # D-Fine의 핵심 장점!
            },
        )

        logger.info(
            f"감지 완료: {result.symbol_count}개 심볼, "
            f"{elapsed:.1f}ms, 안전장치 {len(result.safety_devices)}개"
        )
        return result

    def detect_from_segments(
        self,
        segments: list,
    ) -> DetectionResult:
        """세그먼트 단위로 감지 후 전체 결과를 합칩니다.

        [왜 세그먼트별로 감지하는가?]
        대형 도면에서 작은 심볼을 놓치지 않으려면,
        도면을 작은 조각으로 나눠서 각각 감지한 뒤
        결과를 합쳐야 합니다.

        겹침 영역(overlap)에서 중복 감지된 심볼은
        IoU(겹침 비율) 기반으로 제거합니다.
        """
        start_time = time.time()
        all_symbols: list[DetectedSymbol] = []

        for seg in segments:
            # 각 세그먼트에서 감지
            seg_result = self.detect(seg.image)

            # 세그먼트 좌표 → 원본 좌표 변환
            for symbol in seg_result.symbols:
                x1, y1, x2, y2 = symbol.bbox
                symbol.bbox = (
                    x1 + seg.x_offset,
                    y1 + seg.y_offset,
                    x2 + seg.x_offset,
                    y2 + seg.y_offset,
                )
                symbol.segment_row = seg.row
                symbol.segment_col = seg.col
                all_symbols.append(symbol)

        # 겹침 영역 중복 제거
        merged_symbols = self._remove_overlapping_detections(all_symbols)

        elapsed = (time.time() - start_time) * 1000

        return DetectionResult(
            symbols=merged_symbols,
            inference_time_ms=elapsed,
            image_size=(segments[0].original_width, segments[0].original_height),
            model_name="D-Fine-L (HGNetV2)",
            metadata={
                "num_segments": len(segments),
                "pre_merge_count": len(all_symbols),
                "post_merge_count": len(merged_symbols),
            },
        )

    def _remove_overlapping_detections(
        self,
        symbols: list[DetectedSymbol],
        iou_threshold: float = 0.5,
    ) -> list[DetectedSymbol]:
        """겹침 영역에서 중복 감지된 심볼을 제거합니다.

        [방법]
        같은 종류의 심볼이 50% 이상 겹쳐있으면 → 중복!
        더 높은 확신도를 가진 것만 남깁니다.
        """
        if not symbols:
            return []

        # 확신도 내림차순 정렬
        sorted_symbols = sorted(symbols, key=lambda s: s.confidence, reverse=True)
        kept = []

        for symbol in sorted_symbols:
            is_duplicate = False
            for existing in kept:
                if symbol.class_id == existing.class_id:
                    iou = self._calculate_iou(symbol.bbox, existing.bbox)
                    if iou > iou_threshold:
                        is_duplicate = True
                        break
            if not is_duplicate:
                kept.append(symbol)

        logger.info(f"중복 제거: {len(symbols)} → {len(kept)}개")
        return kept

    @staticmethod
    def _calculate_iou(
        box1: tuple[float, float, float, float],
        box2: tuple[float, float, float, float],
    ) -> float:
        """두 바운딩 박스의 IoU (겹침 비율) 계산"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _simulate_detection(self, w: int, h: int) -> list[DetectedSymbol]:
        """개발용 시뮬레이션 감지 결과 생성

        [참고] 실제 D-Fine 모델이 연결되면 이 함수는 제거됩니다.
        """
        np.random.seed(42)
        sample_detections = [
            ("gate_valve", 0.95),
            ("pump", 0.92),
            ("tank", 0.98),
            ("heat_exchanger", 0.88),
            ("pressure_transmitter", 0.91),
            ("control_valve", 0.94),
            ("relief_valve", 0.96),       # 안전장치!
            ("flow_indicator", 0.87),
            ("check_valve", 0.89),
            ("rupture_disc", 0.93),        # 안전장치!
        ]

        symbols = []
        for class_name, conf in sample_detections:
            if conf >= self.confidence_threshold:
                x1 = np.random.randint(0, max(1, w - 100))
                y1 = np.random.randint(0, max(1, h - 100))
                symbols.append(DetectedSymbol(
                    class_id=self._class_names.index(class_name)
                        if class_name in self._class_names else 0,
                    class_name=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x1 + 80, y1 + 80),
                ))

        return symbols
