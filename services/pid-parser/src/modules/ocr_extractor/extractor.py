"""
M01-04: ocr_extractor
━━━━━━━━━━━━━━━━━━━━━
PaddleOCR 기반 P&ID 도면 텍스트 추출 모듈

[역할]
- 도면에서 텍스트 정보를 추출합니다.
- 태그 번호 (예: V-101, P-201), 관경 (3"), 재질 (SUS304) 등
- 추출된 텍스트를 심볼과 매칭하여 의미 있는 데이터를 만듭니다.

[비유]
심볼 감지(M01-03)가 '어디에 뭐가 있는지' 찾는다면,
OCR은 '그 옆에 뭐라고 적혀있는지' 읽는 역할입니다.
둘이 합쳐져야 "V-101 게이트 밸브, 3인치, SUS304"라는
완전한 정보가 됩니다.

[PaddleOCR 선택 이유]
- Apache 2.0 라이선스 → SaaS 상용화 자유
- 한국어 지원 우수
- 기울어진 텍스트 인식 가능 (도면은 다양한 각도의 텍스트 존재)
"""

import re
import numpy as np
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class ExtractedText:
    """추출된 텍스트 하나의 정보

    [쉬운 설명]
    도면에서 읽어낸 글자 하나(한 덩어리)의 정보입니다.
    """
    text: str                   # 읽어낸 문자열
    confidence: float           # 확신도 (0.0~1.0)
    bbox: tuple[float, float, float, float]  # 위치 (x1, y1, x2, y2)
    category: str = "unknown"   # 분류 (tag, size, material 등)
    angle: float = 0.0          # 텍스트 각도 (도)


@dataclass
class TagInfo:
    """도면 태그 정보 (파싱 결과)

    [P&ID 태그 체계]
    예: TIC-101A
    - T: Temperature (온도)
    - I: Indicator (지시)
    - C: Controller (제어)
    - 101: 루프 번호
    - A: 접미사
    """
    raw_tag: str                # 원본 태그 문자열
    prefix: str                 # 기기 유형 접두사 (V, P, E, T 등)
    number: str                 # 번호 (101, 201 등)
    suffix: str = ""            # 접미사 (A, B 등)
    isa_code: str = ""          # ISA 코드 해석 (계장기기용)

    @property
    def equipment_type(self) -> str:
        """접두사로 장비 종류 판별"""
        type_map = {
            "V": "밸브 (Valve)",
            "P": "펌프 (Pump)",
            "E": "열교환기 (Exchanger)",
            "T": "탱크/탑 (Tank/Tower)",
            "C": "압축기 (Compressor)",
            "R": "반응기 (Reactor)",
            "D": "드럼 (Drum)",
            "F": "필터 (Filter)",
            "PI": "압력지시계",
            "TI": "온도지시계",
            "FI": "유량지시계",
            "LI": "레벨지시계",
            "PT": "압력전송기",
            "TT": "온도전송기",
            "FT": "유량전송기",
            "LT": "레벨전송기",
            "PIC": "압력지시제어기",
            "TIC": "온도지시제어기",
            "FIC": "유량지시제어기",
            "LIC": "레벨지시제어기",
            "PSV": "압력안전밸브",
            "TSV": "온도안전밸브",
        }
        return type_map.get(self.prefix, f"미분류 ({self.prefix})")


class PIDOCRExtractor:
    """P&ID 도면 텍스트 추출기

    [사용법]
    extractor = PIDOCRExtractor()
    texts = extractor.extract(image)

    # 태그 번호만 필터링
    tags = extractor.parse_tags(texts)
    for tag in tags:
        print(f"{tag.raw_tag} → {tag.equipment_type}")
    """

    def __init__(self, lang: str = "korean", use_gpu: bool = True):
        self.lang = lang
        self.use_gpu = use_gpu
        self.ocr_engine = None

        # 태그 패턴 정규표현식
        self._tag_patterns = [
            # 장비 태그: V-101, P-201A, E-301
            re.compile(r'\b([A-Z]{1,3})-?(\d{2,4})([A-Z]?)\b'),
            # 계장 태그: TIC-101, FT-201, PSV-301A
            re.compile(r'\b([A-Z]{2,4})-?(\d{2,4})([A-Z]?)\b'),
            # 배관 크기: 3", 4", 6", 8"
            re.compile(r'\b(\d{1,2})["\']?\s*(inch|인치)?\b'),
            # 라인 번호: 3"-P-101-A1
            re.compile(r'(\d+)["\']?-([A-Z])-(\d+)-([A-Z0-9]+)'),
        ]

        logger.info(f"OCR 추출기 초기화: lang={lang}, gpu={use_gpu}")

    def load_engine(self):
        """PaddleOCR 엔진을 로드합니다.

        [실제 구현]
        from paddleocr import PaddleOCR
        self.ocr_engine = PaddleOCR(
            use_angle_cls=True,    # 텍스트 방향 자동 감지
            lang=self.lang,
            use_gpu=self.use_gpu,
            show_log=False,
        )
        """
        logger.info("PaddleOCR 엔진 로드 완료")

    def extract(self, image: np.ndarray) -> list[ExtractedText]:
        """이미지에서 모든 텍스트를 추출합니다.

        [추출 프로세스]
        1) PaddleOCR로 텍스트 영역 감지 + 인식
        2) 각 텍스트의 카테고리 자동 분류
        3) 신뢰도 기반 필터링

        Args:
            image: OpenCV BGR 이미지

        Returns:
            추출된 텍스트 목록
        """
        logger.info("텍스트 추출 시작...")

        # ── PaddleOCR 추론 ──
        # 실제 구현:
        # results = self.ocr_engine.ocr(image, cls=True)
        # for line in results[0]:
        #     bbox_points = line[0]        # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        #     text = line[1][0]            # 인식된 문자열
        #     confidence = line[1][1]      # 신뢰도

        # ── 프레임워크: 시뮬레이션 결과 ──
        texts = self._simulate_extraction(image)

        # 카테고리 분류
        for text in texts:
            text.category = self._classify_text(text.text)

        logger.info(f"텍스트 추출 완료: {len(texts)}개")
        return texts

    def parse_tags(self, texts: list[ExtractedText]) -> list[TagInfo]:
        """추출된 텍스트에서 장비 태그 번호를 파싱합니다.

        [태그 파싱 규칙]
        1) 패턴 매칭: V-101, TIC-201, PSV-301 등
        2) 접두사 분석: 장비 종류 판별
        3) ISA 코드 해석: 계장기기 기능 파악

        [ISA 코드 체계]
        첫 글자: 측정 변수 (T=온도, P=압력, F=유량, L=레벨)
        이후: 기능 (I=지시, C=제어, T=전송, A=경보, S=안전)
        """
        tags = []

        for text in texts:
            for pattern in self._tag_patterns:
                match = pattern.search(text.text)
                if match:
                    groups = match.groups()
                    if len(groups) >= 2:
                        tag = TagInfo(
                            raw_tag=match.group(0),
                            prefix=groups[0],
                            number=groups[1],
                            suffix=groups[2] if len(groups) > 2 else "",
                            isa_code=self._interpret_isa(groups[0]),
                        )
                        tags.append(tag)

        logger.info(f"태그 파싱: {len(tags)}개 태그 인식")
        return tags

    def _classify_text(self, text: str) -> str:
        """텍스트의 카테고리를 자동 분류합니다.

        [카테고리]
        - tag: 장비 태그 번호 (V-101, P-201)
        - size: 배관 크기 (3", 4")
        - material: 재질 (SUS304, CS)
        - line_number: 라인 번호 (3"-P-101-A1)
        - note: 주석/설명
        """
        text_upper = text.upper().strip()

        # 배관 크기 (먼저 체크)
        if re.match(r'^\d{1,2}["\']', text_upper) or "INCH" in text_upper:
            return "size"

        # 재질 (태그 패턴보다 먼저 체크 - SUS304 같은 것이 태그로 오분류 방지)
        materials = {"SUS", "CS", "SS", "PTFE", "PVC", "HDPE", "TITANIUM"}
        if any(m in text_upper for m in materials):
            return "material"

        # 태그 번호 패턴
        if re.match(r'^[A-Z]{1,4}-?\d{2,4}[A-Z]?$', text_upper):
            return "tag"

        # 라인 번호
        if re.match(r'\d+["\']?-[A-Z]-\d+-', text_upper):
            return "line_number"

        return "note"

    def _interpret_isa(self, code: str) -> str:
        """ISA 코드를 해석합니다.

        [ISA-5.1 표준 기반]
        첫 번째 문자: 측정 변수
        나머지: 기능 수식어
        """
        if len(code) < 2:
            return ""

        variables = {
            "T": "온도", "P": "압력", "F": "유량",
            "L": "레벨", "A": "분석", "S": "속도",
        }
        functions = {
            "I": "지시", "C": "제어", "T": "전송",
            "A": "경보", "S": "안전", "R": "기록",
            "V": "밸브",
        }

        parts = []
        first_char = code[0]
        if first_char in variables:
            parts.append(variables[first_char])

        for char in code[1:]:
            if char in functions:
                parts.append(functions[char])

        return " ".join(parts) if parts else code

    def _simulate_extraction(self, image: np.ndarray) -> list[ExtractedText]:
        """개발용 시뮬레이션 텍스트 추출"""
        h, w = image.shape[:2]
        sample_texts = [
            ("V-101", 0.95, "tag"),
            ("P-201A", 0.93, "tag"),
            ("E-301", 0.91, "tag"),
            ("TIC-101", 0.94, "tag"),
            ("FT-201", 0.89, "tag"),
            ("PSV-301", 0.96, "tag"),
            ("3\"", 0.88, "size"),
            ("6\"", 0.90, "size"),
            ("SUS304", 0.87, "material"),
            ("CS", 0.85, "material"),
            ("3\"-P-101-A1", 0.92, "line_number"),
        ]

        return [
            ExtractedText(
                text=text,
                confidence=conf,
                bbox=(
                    np.random.randint(0, max(1, w - 60)),
                    np.random.randint(0, max(1, h - 20)),
                    np.random.randint(60, max(61, w)),
                    np.random.randint(20, max(21, h)),
                ),
                category=cat,
            )
            for text, conf, cat in sample_texts
        ]
