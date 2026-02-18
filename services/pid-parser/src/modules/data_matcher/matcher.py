"""
M01-05: data_matcher
━━━━━━━━━━━━━━━━━━━━
감지된 심볼과 추출된 텍스트를 매칭하여 구조화된 데이터 생성

[역할]
- D-Fine이 찾은 심볼(위치+종류)과 OCR이 읽은 텍스트(태그번호 등)를 연결합니다.
- "이 펌프의 태그는 P-201A, 3인치, SUS304" 같은 완전한 정보를 만듭니다.
- 장치 간 연결 관계(배관으로 연결된 장치)도 추론합니다.

[비유]
심볼 감지 = 사람 얼굴 인식
OCR = 이름표 읽기
데이터 매칭 = "이 얼굴 옆에 있는 이름표가 이 사람의 이름이다"를 연결

[매칭 원리]
심볼의 바운딩 박스 주변에 있는 텍스트 중
가장 가까운 것을 해당 심볼의 태그로 매칭합니다.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from src.modules.symbol_detector.detector import DetectedSymbol, DetectionResult
from src.modules.ocr_extractor.extractor import ExtractedText, TagInfo


@dataclass
class PIDEquipment:
    """매칭 완료된 P&ID 장비 정보

    [쉬운 설명]
    심볼 + 텍스트가 합쳐진 '완전한 장비 카드'입니다.
    PSM 위험성평가에 직접 활용되는 핵심 데이터입니다.
    """
    # ── 기본 정보 ──
    id: str                         # 고유 ID (auto-generated)
    symbol: DetectedSymbol          # 감지된 심볼 정보
    tag: Optional[TagInfo] = None   # 매칭된 태그 정보

    # ── 추가 속성 ──
    pipe_size: Optional[str] = None     # 배관 크기 (예: 3")
    material: Optional[str] = None      # 재질 (예: SUS304)
    line_number: Optional[str] = None   # 라인 번호

    # ── 연결 관계 ──
    connected_to: list[str] = field(default_factory=list)  # 연결된 장비 ID들
    upstream: list[str] = field(default_factory=list)       # 상류 장비
    downstream: list[str] = field(default_factory=list)     # 하류 장비

    # ── PSM 관련 ──
    is_safety_device: bool = False      # 안전장치 여부
    psm_criticality: str = "normal"     # PSM 중요도 (normal/high/critical)
    notes: list[str] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        """표시용 이름"""
        tag_str = self.tag.raw_tag if self.tag else "미확인"
        return f"{tag_str} ({self.symbol.korean_name})"

    def to_dict(self) -> dict:
        """API 응답용 딕셔너리 변환"""
        return {
            "id": self.id,
            "tag": self.tag.raw_tag if self.tag else None,
            "equipment_type": self.tag.equipment_type if self.tag else None,
            "symbol_class": self.symbol.class_name,
            "symbol_korean": self.symbol.korean_name,
            "confidence": self.symbol.confidence,
            "position": {
                "x": self.symbol.center[0],
                "y": self.symbol.center[1],
                "bbox": self.symbol.bbox,
            },
            "pipe_size": self.pipe_size,
            "material": self.material,
            "line_number": self.line_number,
            "connected_to": self.connected_to,
            "is_safety_device": self.is_safety_device,
            "psm_criticality": self.psm_criticality,
        }


@dataclass
class MatchingResult:
    """전체 매칭 결과"""
    equipments: list[PIDEquipment]              # 매칭된 장비 목록
    unmatched_symbols: list[DetectedSymbol]     # 태그 미매칭 심볼
    unmatched_texts: list[ExtractedText]        # 심볼 미매칭 텍스트
    statistics: dict = field(default_factory=dict)

    @property
    def match_rate(self) -> float:
        """매칭 성공률"""
        total = len(self.equipments) + len(self.unmatched_symbols)
        return len(self.equipments) / total if total > 0 else 0.0

    @property
    def safety_devices(self) -> list[PIDEquipment]:
        """안전장치 목록 (PSM 핵심!)"""
        return [e for e in self.equipments if e.is_safety_device]


class PIDDataMatcher:
    """P&ID 심볼-텍스트 매칭 엔진

    [사용법]
    matcher = PIDDataMatcher()
    result = matcher.match(
        symbols=detection_result.symbols,
        texts=extracted_texts,
        tags=parsed_tags,
    )

    for equip in result.equipments:
        print(f"{equip.display_name} - 중요도: {equip.psm_criticality}")
    """

    def __init__(self, max_distance: float = 150.0):
        """
        Args:
            max_distance: 심볼-텍스트 최대 매칭 거리 (픽셀)
                         이보다 멀리 있으면 관련 없다고 판단
        """
        self.max_distance = max_distance
        self._id_counter = 0

    def match(
        self,
        symbols: list[DetectedSymbol],
        texts: list[ExtractedText],
        tags: list[TagInfo],
    ) -> MatchingResult:
        """심볼과 텍스트를 매칭합니다.

        [매칭 알고리즘]
        1) 각 심볼의 중심점 계산
        2) 각 태그 텍스트의 중심점 계산
        3) 가장 가까운 심볼-태그 쌍을 매칭 (헝가리안 알고리즘)
        4) 매칭된 쌍 주변의 크기/재질 텍스트도 연결
        5) PSM 안전장치 여부 판별
        """
        logger.info(f"매칭 시작: 심볼 {len(symbols)}개, 텍스트 {len(texts)}개, 태그 {len(tags)}개")

        equipments: list[PIDEquipment] = []
        matched_symbol_ids: set[int] = set()
        matched_tag_ids: set[int] = set()

        # ── 1단계: 심볼-태그 매칭 (가장 가까운 태그) ──
        for sym_idx, symbol in enumerate(symbols):
            best_tag = None
            best_distance = float('inf')
            best_tag_idx = -1

            for tag_idx, tag in enumerate(tags):
                if tag_idx in matched_tag_ids:
                    continue

                # 태그에 해당하는 텍스트 위치 찾기
                tag_text = self._find_text_for_tag(tag, texts)
                if tag_text is None:
                    continue

                distance = self._distance(symbol.center, self._center(tag_text.bbox))
                if distance < best_distance and distance <= self.max_distance:
                    best_distance = distance
                    best_tag = tag
                    best_tag_idx = tag_idx

            # 매칭 성공
            self._id_counter += 1
            equip_id = f"EQ-{self._id_counter:04d}"

            equipment = PIDEquipment(
                id=equip_id,
                symbol=symbol,
                tag=best_tag,
                is_safety_device=self._is_safety_device(symbol),
                psm_criticality=self._assess_criticality(symbol, best_tag),
            )

            if best_tag:
                matched_tag_ids.add(best_tag_idx)

            # ── 2단계: 주변 크기/재질 텍스트 연결 ──
            nearby_texts = self._find_nearby_texts(symbol, texts)
            for text in nearby_texts:
                if text.category == "size":
                    equipment.pipe_size = text.text
                elif text.category == "material":
                    equipment.material = text.text
                elif text.category == "line_number":
                    equipment.line_number = text.text

            equipments.append(equipment)
            matched_symbol_ids.add(sym_idx)

        # ── 3단계: 미매칭 항목 수집 ──
        unmatched_symbols = [s for i, s in enumerate(symbols) if i not in matched_symbol_ids]
        unmatched_texts = [t for t in texts if t.category == "tag" and
                          not any(eq.tag and eq.tag.raw_tag == t.text for eq in equipments)]

        result = MatchingResult(
            equipments=equipments,
            unmatched_symbols=unmatched_symbols,
            unmatched_texts=unmatched_texts,
            statistics={
                "total_symbols": len(symbols),
                "total_tags": len(tags),
                "matched": len([e for e in equipments if e.tag]),
                "match_rate": f"{len([e for e in equipments if e.tag]) / max(len(symbols), 1) * 100:.1f}%",
                "safety_devices_found": len([e for e in equipments if e.is_safety_device]),
            },
        )

        logger.info(f"매칭 완료: {result.statistics}")
        return result

    def _find_text_for_tag(self, tag: TagInfo, texts: list[ExtractedText]) -> Optional[ExtractedText]:
        """태그에 해당하는 텍스트 객체를 찾습니다."""
        for text in texts:
            if tag.raw_tag in text.text:
                return text
        return None

    def _find_nearby_texts(
        self,
        symbol: DetectedSymbol,
        texts: list[ExtractedText],
        radius: float = 200.0,
    ) -> list[ExtractedText]:
        """심볼 주변의 텍스트를 찾습니다."""
        nearby = []
        for text in texts:
            dist = self._distance(symbol.center, self._center(text.bbox))
            if dist <= radius and text.category != "tag":
                nearby.append(text)
        return nearby

    def _is_safety_device(self, symbol: DetectedSymbol) -> bool:
        """PSM 안전장치 여부 판별

        [PSM 안전장치]
        - relief_valve: 압력 안전밸브 (과압 방지)
        - rupture_disc: 파열판 (비상 압력 해소)
        - flame_arrestor: 화염방지기
        - emergency_shutoff: 긴급 차단 장치
        - safety_shower: 비상 샤워

        이 장치들은 PSM 위험성평가에서 '독립방호계층(IPL)'으로
        매우 중요하게 취급됩니다.
        """
        safety_classes = {
            "relief_valve", "rupture_disc", "flame_arrestor",
            "safety_shower", "emergency_shutoff",
        }
        return symbol.class_name in safety_classes

    def _assess_criticality(
        self,
        symbol: DetectedSymbol,
        tag: Optional[TagInfo],
    ) -> str:
        """PSM 중요도를 평가합니다.

        [중요도 등급]
        - critical: 안전장치, 비상 장비
        - high: 반응기, 압력용기, 고압 장비
        - normal: 일반 장비
        """
        if self._is_safety_device(symbol):
            return "critical"

        high_risk = {"reactor", "vessel", "compressor", "column"}
        if symbol.class_name in high_risk:
            return "high"

        # PSV(압력안전밸브) 태그인 경우
        if tag and tag.prefix in ("PSV", "TSV", "ESD"):
            return "critical"

        return "normal"

    @staticmethod
    def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """두 점 사이의 거리"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    @staticmethod
    def _center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        """바운딩 박스의 중심점"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
