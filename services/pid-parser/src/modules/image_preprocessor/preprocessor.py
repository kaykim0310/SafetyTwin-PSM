"""
M01-01: image_preprocessor
━━━━━━━━━━━━━━━━━━━━━━━━━━
P&ID 도면 이미지 전처리 모듈

[역할]
- 도면 이미지를 D-Fine 모델이 인식하기 좋은 상태로 가공합니다.
- 노이즈 제거, 해상도 보정, 세그먼트 분할을 수행합니다.
- 대형 도면(A0, A1)을 640×640 세그먼트로 쪼개서 정밀 분석합니다.

[비유]
도면을 사진 찍어서 AI에게 넘기기 전에, 사진을 깨끗하게 정리하고
적당한 크기로 잘라주는 '사전 준비' 단계입니다.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class ImageSegment:
    """분할된 도면 조각 하나를 나타냅니다.

    [쉬운 설명]
    대형 도면을 격자(그리드)로 잘랐을 때, 각 칸이 ImageSegment입니다.
    row, col은 몇 번째 줄/칸인지, x, y는 원본에서의 위치입니다.
    """
    image: np.ndarray           # 잘린 이미지 데이터
    row: int                    # 세그먼트 행 번호
    col: int                    # 세그먼트 열 번호
    x_offset: int               # 원본 이미지에서의 X 좌표
    y_offset: int               # 원본 이미지에서의 Y 좌표
    original_width: int         # 원본 이미지 전체 너비
    original_height: int        # 원본 이미지 전체 높이


@dataclass
class PreprocessResult:
    """전처리 결과를 담는 컨테이너입니다."""
    original_image: np.ndarray          # 원본 이미지
    processed_image: np.ndarray         # 전처리된 이미지
    segments: list[ImageSegment]        # 분할된 조각들
    scale_factor: float                 # 크기 조정 비율
    metadata: dict                      # 메타정보


class PIDImagePreprocessor:
    """P&ID 도면 이미지 전처리기

    [사용법]
    preprocessor = PIDImagePreprocessor()
    result = preprocessor.process("도면파일.png")
    # result.segments → D-Fine 모델에 넣을 이미지 조각들
    """

    def __init__(
        self,
        max_size: int = 4096,
        segment_size: int = 640,
        overlap: int = 64,
    ):
        """
        Args:
            max_size: 이미지 최대 크기 (이보다 크면 축소)
            segment_size: D-Fine 입력 크기에 맞춘 세그먼트 크기
            overlap: 세그먼트 간 겹침 (경계에 있는 심볼 누락 방지)
        """
        self.max_size = max_size
        self.segment_size = segment_size
        self.overlap = overlap
        logger.info(
            f"전처리기 초기화: max={max_size}, segment={segment_size}, overlap={overlap}"
        )

    def process(self, image_path: str | Path) -> PreprocessResult:
        """도면 이미지를 전처리합니다.

        [처리 순서]
        1) 파일 읽기
        2) 크기 조정 (너무 크면 축소)
        3) 노이즈 제거 (깨끗하게)
        4) 대비 향상 (흐릿한 선을 선명하게)
        5) 이진화 (선과 배경 구분)
        6) 세그먼트 분할 (격자로 자르기)
        """
        image_path = Path(image_path)
        logger.info(f"도면 전처리 시작: {image_path.name}")

        # 1) 파일 읽기
        original = cv2.imread(str(image_path))
        if original is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

        h, w = original.shape[:2]
        logger.info(f"원본 크기: {w}×{h}")

        # 2) 크기 조정
        processed, scale = self._resize_if_needed(original)

        # 3) 노이즈 제거
        processed = self._denoise(processed)

        # 4) 대비 향상
        processed = self._enhance_contrast(processed)

        # 5) 세그먼트 분할
        segments = self._segment(processed, original_w=w, original_h=h)
        logger.info(f"세그먼트 분할 완료: {len(segments)}개 생성")

        return PreprocessResult(
            original_image=original,
            processed_image=processed,
            segments=segments,
            scale_factor=scale,
            metadata={
                "source_file": str(image_path),
                "original_size": (w, h),
                "processed_size": processed.shape[:2][::-1],
                "num_segments": len(segments),
                "segment_size": self.segment_size,
                "overlap": self.overlap,
            },
        )

    def _resize_if_needed(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """이미지가 너무 크면 축소합니다.

        [왜 필요한가?]
        A0 도면을 스캔하면 10000×7000px이 될 수 있습니다.
        GPU 메모리 한계가 있으므로 적절한 크기로 줄입니다.
        """
        h, w = image.shape[:2]
        max_dim = max(h, w)

        if max_dim <= self.max_size:
            return image, 1.0

        scale = self.max_size / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(f"크기 조정: {w}×{h} → {new_w}×{new_h} (비율: {scale:.2f})")
        return resized, scale

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """노이즈를 제거합니다.

        [왜 필요한가?]
        오래된 도면을 스캔하면 얼룩, 접힌 자국 등 노이즈가 있습니다.
        AI가 이런 것들을 심볼로 오인식하지 않도록 깨끗하게 정리합니다.
        """
        # 가우시안 블러로 미세 노이즈 제거
        denoised = cv2.GaussianBlur(image, (3, 3), 0)
        return denoised

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """대비를 향상시킵니다.

        [왜 필요한가?]
        바랜 도면이나 연한 잉크로 인쇄된 도면의 선을
        더 선명하게 만들어 AI가 잘 인식하도록 합니다.
        """
        # 그레이스케일 변환 후 CLAHE 적용
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # CLAHE: 지역 대비 향상 (도면의 세밀한 부분도 선명하게)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 다시 3채널로 변환 (D-Fine은 RGB 입력)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return enhanced_bgr

    def _segment(
        self,
        image: np.ndarray,
        original_w: int,
        original_h: int,
    ) -> list[ImageSegment]:
        """이미지를 격자로 분할합니다.

        [왜 필요한가?]
        D-Fine 모델은 640×640 크기 입력에서 최적 성능을 보입니다.
        대형 도면을 한번에 넣으면 작은 심볼을 놓칠 수 있으므로,
        640×640으로 잘라서 각각 분석합니다.

        세그먼트를 살짝 겹치게(overlap) 잘라서,
        경계선에 걸린 심볼도 놓치지 않습니다.
        """
        h, w = image.shape[:2]
        step = self.segment_size - self.overlap
        segments = []

        row_idx = 0
        y = 0
        while y < h:
            col_idx = 0
            x = 0
            while x < w:
                # 세그먼트 영역 계산
                x_end = min(x + self.segment_size, w)
                y_end = min(y + self.segment_size, h)

                # 세그먼트 추출
                segment_img = image[y:y_end, x:x_end]

                # 세그먼트가 지정 크기보다 작으면 패딩 추가
                if segment_img.shape[0] < self.segment_size or \
                   segment_img.shape[1] < self.segment_size:
                    padded = np.ones(
                        (self.segment_size, self.segment_size, 3),
                        dtype=np.uint8,
                    ) * 255  # 흰색 패딩
                    padded[:segment_img.shape[0], :segment_img.shape[1]] = segment_img
                    segment_img = padded

                segments.append(ImageSegment(
                    image=segment_img,
                    row=row_idx,
                    col=col_idx,
                    x_offset=x,
                    y_offset=y,
                    original_width=original_w,
                    original_height=original_h,
                ))

                x += step
                col_idx += 1

            y += step
            row_idx += 1

        return segments

    def binarize(self, image: np.ndarray) -> np.ndarray:
        """이미지를 흑백 이진화합니다. (OCR 전처리용)

        [용도]
        텍스트 추출(OCR)에서는 이진화된 이미지가 더 정확합니다.
        선은 검정, 배경은 흰색으로 깔끔하게 분리합니다.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 적응형 이진화 (도면의 밝기가 균일하지 않아도 잘 작동)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=10,
        )
        return binary
