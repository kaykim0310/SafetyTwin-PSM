"""
PSM-SafetyTwin SVC-01: pid-parser 테스트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
pytest tests/ 로 실행

[테스트 범위]
- M01-01: 이미지 전처리
- M01-02: D-Fine 학습 설정
- M01-03: 심볼 감지
- M01-04: OCR 텍스트 추출
- M01-05: 심볼-텍스트 매칭
"""

import sys
import os
import numpy as np

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestImagePreprocessor:
    """M01-01: 이미지 전처리 테스트"""

    def test_preprocessor_init(self):
        """전처리기 초기화"""
        from src.modules.image_preprocessor.preprocessor import PIDImagePreprocessor
        pp = PIDImagePreprocessor(max_size=4096, segment_size=640, overlap=64)
        assert pp.max_size == 4096
        assert pp.segment_size == 640
        assert pp.overlap == 64

    def test_resize_large_image(self):
        """대형 이미지 축소"""
        from src.modules.image_preprocessor.preprocessor import PIDImagePreprocessor
        pp = PIDImagePreprocessor(max_size=2000)
        # 5000×3000 이미지 생성
        large_img = np.ones((3000, 5000, 3), dtype=np.uint8) * 255
        resized, scale = pp._resize_if_needed(large_img)
        assert resized.shape[1] <= 2000  # 너비가 2000 이하
        assert scale < 1.0

    def test_resize_small_image(self):
        """작은 이미지는 축소 안 함"""
        from src.modules.image_preprocessor.preprocessor import PIDImagePreprocessor
        pp = PIDImagePreprocessor(max_size=4096)
        small_img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
        resized, scale = pp._resize_if_needed(small_img)
        assert scale == 1.0

    def test_segmentation(self):
        """이미지 세그먼트 분할"""
        from src.modules.image_preprocessor.preprocessor import PIDImagePreprocessor
        pp = PIDImagePreprocessor(segment_size=640, overlap=64)
        img = np.ones((1280, 1920, 3), dtype=np.uint8) * 255
        segments = pp._segment(img, original_w=1920, original_h=1280)
        assert len(segments) > 1
        # 각 세그먼트가 640×640인지 확인
        for seg in segments:
            assert seg.image.shape[0] == 640
            assert seg.image.shape[1] == 640

    def test_binarize(self):
        """이진화 테스트"""
        from src.modules.image_preprocessor.preprocessor import PIDImagePreprocessor
        pp = PIDImagePreprocessor()
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        binary = pp.binarize(img)
        assert len(binary.shape) == 2  # 흑백 (2차원)
        assert set(np.unique(binary)).issubset({0, 255})  # 0 또는 255만


class TestDFineTrainer:
    """M01-02: D-Fine 학습 설정 테스트"""

    def test_training_config(self):
        """학습 설정 기본값"""
        from src.modules.dfine_trainer.trainer import TrainingConfig
        config = TrainingConfig()
        assert config.num_classes == 42
        assert config.model_name == "dfine_hgnetv2_l"
        assert config.epochs == 50

    def test_training_plan(self):
        """학습 계획 생성"""
        from src.modules.dfine_trainer.trainer import DFineTrainer, TrainingConfig
        config = TrainingConfig(epochs=10, warmup_epochs=2)
        trainer = DFineTrainer(config)
        plan = trainer._generate_training_plan()
        assert len(plan) == 10
        assert plan[0]["phase"] == "warmup"
        assert plan[5]["phase"] == "training"

    def test_yaml_generation(self):
        """YAML 설정 파일 생성"""
        from src.modules.dfine_trainer.trainer import (
            TrainingConfig, generate_dfine_training_yaml,
        )
        config = TrainingConfig(num_classes=42, epochs=50)
        yaml = generate_dfine_training_yaml(config)
        assert "D-Fine" in yaml or "dfine" in yaml.lower()
        assert "num_classes: 42" in yaml
        assert "epochs: 50" in yaml


class TestSymbolDetector:
    """M01-03: D-Fine 심볼 감지 테스트"""

    def test_detector_init(self):
        """감지기 초기화"""
        from src.modules.symbol_detector.detector import DFineSymbolDetector
        det = DFineSymbolDetector(confidence_threshold=0.5, device="cpu")
        assert det.confidence_threshold == 0.5
        assert det.device == "cpu"

    def test_simulate_detection(self):
        """시뮬레이션 감지"""
        from src.modules.symbol_detector.detector import DFineSymbolDetector
        det = DFineSymbolDetector(confidence_threshold=0.5, device="cpu")
        img = np.ones((640, 640, 3), dtype=np.uint8) * 255
        result = det.detect(img)
        assert result.symbol_count > 0
        assert result.model_name == "D-Fine-L (HGNetV2)"
        assert result.metadata["nms_required"] is False  # D-Fine 핵심!

    def test_safety_devices_filter(self):
        """PSM 안전장치 필터"""
        from src.modules.symbol_detector.detector import DFineSymbolDetector
        det = DFineSymbolDetector(confidence_threshold=0.3, device="cpu")
        img = np.ones((640, 640, 3), dtype=np.uint8) * 255
        result = det.detect(img)
        safety = result.safety_devices
        for s in safety:
            assert s.class_name in {
                "relief_valve", "rupture_disc", "flame_arrestor",
                "safety_shower", "emergency_shutoff",
            }

    def test_korean_names(self):
        """한국어 이름 매핑"""
        from src.modules.symbol_detector.detector import DetectedSymbol
        sym = DetectedSymbol(
            class_id=0, class_name="gate_valve",
            confidence=0.95, bbox=(0, 0, 100, 100),
        )
        assert sym.korean_name == "게이트 밸브"

    def test_iou_calculation(self):
        """IoU 계산"""
        from src.modules.symbol_detector.detector import DFineSymbolDetector
        # 완전 겹침 → IoU = 1.0
        iou = DFineSymbolDetector._calculate_iou(
            (0, 0, 100, 100), (0, 0, 100, 100)
        )
        assert iou == 1.0
        # 겹침 없음 → IoU = 0.0
        iou = DFineSymbolDetector._calculate_iou(
            (0, 0, 50, 50), (100, 100, 200, 200)
        )
        assert iou == 0.0


class TestOCRExtractor:
    """M01-04: OCR 텍스트 추출 테스트"""

    def test_extractor_init(self):
        """추출기 초기화"""
        from src.modules.ocr_extractor.extractor import PIDOCRExtractor
        ext = PIDOCRExtractor(lang="korean")
        assert ext.lang == "korean"

    def test_text_classification(self):
        """텍스트 카테고리 분류"""
        from src.modules.ocr_extractor.extractor import PIDOCRExtractor
        ext = PIDOCRExtractor()
        assert ext._classify_text("V-101") == "tag"
        assert ext._classify_text("3\"") == "size"
        assert ext._classify_text("SUS304") == "material"

    def test_isa_interpretation(self):
        """ISA 코드 해석"""
        from src.modules.ocr_extractor.extractor import PIDOCRExtractor
        ext = PIDOCRExtractor()
        assert "온도" in ext._interpret_isa("TIC")
        assert "압력" in ext._interpret_isa("PI")

    def test_tag_parsing(self):
        """태그 파싱"""
        from src.modules.ocr_extractor.extractor import PIDOCRExtractor, ExtractedText
        ext = PIDOCRExtractor()
        texts = [
            ExtractedText(text="V-101", confidence=0.95, bbox=(0, 0, 50, 20)),
            ExtractedText(text="TIC-201A", confidence=0.93, bbox=(100, 0, 180, 20)),
        ]
        tags = ext.parse_tags(texts)
        assert len(tags) >= 1


class TestDataMatcher:
    """M01-05: 데이터 매칭 테스트"""

    def test_matcher_init(self):
        """매칭기 초기화"""
        from src.modules.data_matcher.matcher import PIDDataMatcher
        matcher = PIDDataMatcher(max_distance=150)
        assert matcher.max_distance == 150

    def test_safety_assessment(self):
        """PSM 중요도 평가"""
        from src.modules.data_matcher.matcher import PIDDataMatcher
        from src.modules.symbol_detector.detector import DetectedSymbol
        matcher = PIDDataMatcher()

        # 안전장치 → critical
        relief = DetectedSymbol(
            class_id=5, class_name="relief_valve",
            confidence=0.95, bbox=(0, 0, 100, 100),
        )
        assert matcher._is_safety_device(relief) is True
        assert matcher._assess_criticality(relief, None) == "critical"

        # 반응기 → high
        reactor = DetectedSymbol(
            class_id=10, class_name="reactor",
            confidence=0.90, bbox=(0, 0, 100, 100),
        )
        assert matcher._assess_criticality(reactor, None) == "high"

        # 일반 밸브 → normal
        gate = DetectedSymbol(
            class_id=0, class_name="gate_valve",
            confidence=0.88, bbox=(0, 0, 100, 100),
        )
        assert matcher._assess_criticality(gate, None) == "normal"

    def test_distance_calculation(self):
        """거리 계산"""
        from src.modules.data_matcher.matcher import PIDDataMatcher
        dist = PIDDataMatcher._distance((0, 0), (3, 4))
        assert dist == 5.0  # 3-4-5 삼각형


# ============================================================
# 통합 테스트
# ============================================================
class TestEndToEndPipeline:
    """전체 파이프라인 통합 테스트"""

    def test_full_pipeline(self):
        """전처리 → 감지 → OCR → 매칭 전체 흐름"""
        from src.modules.image_preprocessor.preprocessor import PIDImagePreprocessor
        from src.modules.symbol_detector.detector import DFineSymbolDetector
        from src.modules.ocr_extractor.extractor import PIDOCRExtractor
        from src.modules.data_matcher.matcher import PIDDataMatcher

        # 테스트 이미지 생성 (흰 배경)
        test_image = np.ones((1280, 1920, 3), dtype=np.uint8) * 255

        # 1) 전처리
        pp = PIDImagePreprocessor()
        # 직접 처리 (파일 없이)
        processed = pp._enhance_contrast(test_image)
        assert processed is not None

        # 2) 심볼 감지
        det = DFineSymbolDetector(confidence_threshold=0.5, device="cpu")
        result = det.detect(test_image)
        assert result.symbol_count > 0

        # 3) OCR
        ext = PIDOCRExtractor()
        texts = ext.extract(test_image)
        tags = ext.parse_tags(texts)

        # 4) 매칭
        matcher = PIDDataMatcher()
        match_result = matcher.match(
            symbols=result.symbols,
            texts=texts,
            tags=tags,
        )
        assert len(match_result.equipments) > 0
        assert "safety_devices_found" in match_result.statistics

        print(f"\n✅ 통합 테스트 성공!")
        print(f"   심볼: {result.symbol_count}개")
        print(f"   안전장치: {len(result.safety_devices)}개")
        print(f"   태그: {len(tags)}개")
        print(f"   장비: {len(match_result.equipments)}개")
        print(f"   매칭률: {match_result.statistics['match_rate']}")


if __name__ == "__main__":
    """직접 실행 시 전체 테스트"""
    print("=" * 60)
    print("PSM-SafetyTwin P&ID Parser 테스트")
    print("=" * 60)

    test = TestEndToEndPipeline()
    test.test_full_pipeline()
