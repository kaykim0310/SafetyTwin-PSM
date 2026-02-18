"""
PSM-SafetyTwin SVC-01: pid-parser
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FastAPI 메인 애플리케이션

[API 엔드포인트 구성]
POST /api/v1/pid/upload          → 도면 파일 업로드
POST /api/v1/pid/analyze         → 도면 분석 (전체 파이프라인)
GET  /api/v1/pid/results/{id}    → 분석 결과 조회
GET  /api/v1/pid/symbols/{id}    → 심볼 목록 조회
GET  /api/v1/pid/safety/{id}     → 안전장치 목록 (PSM 핵심)
POST /api/v1/pid/export/{id}     → 분석 결과 내보내기
GET  /api/v1/health              → 서비스 상태 확인

[서비스 포트] 8001
"""

import uuid
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

# ── 내부 모듈 임포트 ──
from src.modules.image_preprocessor.preprocessor import PIDImagePreprocessor
from src.modules.symbol_detector.detector import DFineSymbolDetector, DetectionResult
from src.modules.ocr_extractor.extractor import PIDOCRExtractor
from src.modules.data_matcher.matcher import PIDDataMatcher, MatchingResult


# ============================================================
# FastAPI 앱 생성
# ============================================================
app = FastAPI(
    title="PSM-SafetyTwin P&ID Parser API",
    description=(
        "D-Fine 기반 P&ID 도면 자동 인식 및 디지털화 서비스\n\n"
        "- P&ID 도면에서 심볼(밸브, 펌프, 탱크 등) 자동 감지\n"
        "- 태그 번호, 배관 크기, 재질 등 텍스트 자동 추출\n"
        "- PSM 안전장치 자동 식별 및 중요도 분류\n"
        "- 장비 연결 관계 그래프 자동 생성"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 설정 (프론트엔드 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영 시 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 전역 객체 (서비스 시작 시 초기화)
# ============================================================
preprocessor = PIDImagePreprocessor()
detector = DFineSymbolDetector(confidence_threshold=0.5)
ocr_extractor = PIDOCRExtractor()
data_matcher = PIDDataMatcher()

# 분석 결과 임시 저장소 (운영 시 Redis/PostgreSQL로 교체)
analysis_store: dict[str, dict] = {}


# ============================================================
# 요청/응답 모델
# ============================================================
class AnalysisRequest(BaseModel):
    """도면 분석 요청"""
    confidence_threshold: float = Field(
        default=0.5, ge=0.1, le=1.0,
        description="심볼 감지 최소 신뢰도 (0.1~1.0)",
    )
    extract_text: bool = Field(
        default=True,
        description="텍스트 추출 (OCR) 수행 여부",
    )
    match_data: bool = Field(
        default=True,
        description="심볼-텍스트 매칭 수행 여부",
    )


class AnalysisResponse(BaseModel):
    """도면 분석 결과 응답"""
    analysis_id: str
    status: str
    filename: str
    processing_time_ms: float
    summary: dict
    symbols: list[dict]
    safety_devices: list[dict]
    equipments: list[dict]
    statistics: dict


class HealthResponse(BaseModel):
    """서비스 상태 응답"""
    service: str = "pid-parser"
    version: str = "1.0.0"
    status: str = "healthy"
    model: str = "D-Fine-L (Apache 2.0)"
    timestamp: str


# ============================================================
# API 엔드포인트
# ============================================================


@app.get("/api/v1/health", response_model=HealthResponse, tags=["시스템"])
async def health_check():
    """서비스 상태를 확인합니다.

    [용도]
    - 서비스가 정상 동작하는지 확인
    - 로드밸런서 헬스체크
    - 모니터링 시스템 연동
    """
    return HealthResponse(
        timestamp=datetime.now().isoformat(),
    )


@app.post("/api/v1/pid/analyze", tags=["도면 분석"])
async def analyze_pid(
    file: UploadFile = File(..., description="P&ID 도면 파일 (PNG, JPG, PDF)"),
    confidence_threshold: float = Query(
        default=0.5, ge=0.1, le=1.0,
        description="심볼 감지 최소 신뢰도",
    ),
    extract_text: bool = Query(default=True, description="OCR 텍스트 추출 여부"),
):
    """P&ID 도면을 업로드하고 자동 분석합니다.

    [전체 파이프라인]
    ```
    도면 업로드 → 이미지 전처리 → D-Fine 심볼 감지
                                → PaddleOCR 텍스트 추출
                                → 심볼-텍스트 매칭
                                → 결과 반환
    ```

    [지원 파일 형식]
    - PNG, JPG, JPEG: 스캔된 도면 이미지
    - PDF: 도면 PDF (첫 페이지 분석)
    - DWG/DXF: 추후 지원 예정

    [반환 정보]
    - 감지된 심볼 목록 (종류, 위치, 확신도)
    - 추출된 태그/텍스트 (태그번호, 크기, 재질)
    - PSM 안전장치 목록 (critical/high/normal)
    - 매칭 통계 (성공률, 미매칭 항목)
    """
    start_time = time.time()
    analysis_id = str(uuid.uuid4())[:8]

    # ── 파일 검증 ──
    allowed_types = {"image/png", "image/jpeg", "application/pdf"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다: {file.content_type}. "
                   f"지원: PNG, JPG, PDF",
        )

    logger.info(f"[{analysis_id}] 도면 분석 시작: {file.filename}")

    try:
        # ── 1단계: 파일 저장 ──
        upload_dir = Path("uploads") / analysis_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / file.filename

        content = await file.read()
        file_path.write_bytes(content)
        logger.info(f"[{analysis_id}] 파일 저장: {file_path} ({len(content)} bytes)")

        # ── 2단계: 이미지 전처리 (M01-01) ──
        preprocess_result = preprocessor.process(str(file_path))
        logger.info(
            f"[{analysis_id}] 전처리 완료: "
            f"{preprocess_result.metadata['num_segments']}개 세그먼트"
        )

        # ── 3단계: D-Fine 심볼 감지 (M01-03) ──
        detector.confidence_threshold = confidence_threshold

        if len(preprocess_result.segments) > 1:
            detection_result = detector.detect_from_segments(
                preprocess_result.segments
            )
        else:
            detection_result = detector.detect(preprocess_result.processed_image)

        logger.info(
            f"[{analysis_id}] 심볼 감지: {detection_result.symbol_count}개, "
            f"안전장치 {len(detection_result.safety_devices)}개"
        )

        # ── 4단계: OCR 텍스트 추출 (M01-04) ──
        texts = []
        tags = []
        if extract_text:
            texts = ocr_extractor.extract(preprocess_result.processed_image)
            tags = ocr_extractor.parse_tags(texts)
            logger.info(f"[{analysis_id}] OCR: {len(texts)}개 텍스트, {len(tags)}개 태그")

        # ── 5단계: 데이터 매칭 (M01-05) ──
        matching_result = data_matcher.match(
            symbols=detection_result.symbols,
            texts=texts,
            tags=tags,
        )

        # ── 결과 구성 ──
        elapsed_ms = (time.time() - start_time) * 1000

        result = {
            "analysis_id": analysis_id,
            "status": "completed",
            "filename": file.filename,
            "processing_time_ms": round(elapsed_ms, 1),
            "summary": {
                "total_symbols": detection_result.symbol_count,
                "total_texts": len(texts),
                "total_tags": len(tags),
                "safety_devices": len(detection_result.safety_devices),
                "match_rate": matching_result.statistics.get("match_rate", "N/A"),
                "symbol_breakdown": detection_result.symbol_summary,
            },
            "symbols": [
                {
                    "class": s.class_name,
                    "korean_name": s.korean_name,
                    "confidence": round(s.confidence, 3),
                    "bbox": s.bbox,
                    "center": s.center,
                }
                for s in detection_result.symbols
            ],
            "safety_devices": [
                {
                    "class": s.class_name,
                    "korean_name": s.korean_name,
                    "confidence": round(s.confidence, 3),
                    "bbox": s.bbox,
                    "psm_note": "독립방호계층(IPL) 대상 - LOPA 분석 필수",
                }
                for s in detection_result.safety_devices
            ],
            "equipments": [
                e.to_dict() for e in matching_result.equipments
            ],
            "statistics": matching_result.statistics,
            "model_info": {
                "detection_model": "D-Fine-L (HGNetV2, Apache 2.0)",
                "ocr_model": "PaddleOCR (Apache 2.0)",
                "nms_required": False,
            },
        }

        # 결과 저장
        analysis_store[analysis_id] = result

        logger.info(
            f"[{analysis_id}] 분석 완료: "
            f"{elapsed_ms:.0f}ms, {detection_result.symbol_count}개 심볼"
        )

        return JSONResponse(content=result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[{analysis_id}] 분석 실패: {e}")
        raise HTTPException(status_code=500, detail=f"도면 분석 중 오류: {str(e)}")


@app.get("/api/v1/pid/results/{analysis_id}", tags=["결과 조회"])
async def get_analysis_result(analysis_id: str):
    """이전 분석 결과를 조회합니다."""
    if analysis_id not in analysis_store:
        raise HTTPException(status_code=404, detail=f"분석 결과를 찾을 수 없습니다: {analysis_id}")
    return analysis_store[analysis_id]


@app.get("/api/v1/pid/safety/{analysis_id}", tags=["PSM 안전"])
async def get_safety_devices(analysis_id: str):
    """PSM 안전장치 목록만 조회합니다.

    [PSM 위험성평가에서 안전장치가 중요한 이유]
    - LOPA 분석에서 독립방호계층(IPL)으로 활용
    - 각 안전장치의 PFD(Probability of Failure on Demand) 값이
      잔여 위험 빈도 계산에 직접 반영됨
    - 안전장치 누락 = 위험성 과소평가 → 중대산업사고 위험!
    """
    if analysis_id not in analysis_store:
        raise HTTPException(status_code=404, detail="분석 결과 없음")

    result = analysis_store[analysis_id]
    return {
        "analysis_id": analysis_id,
        "safety_devices": result["safety_devices"],
        "total_count": len(result["safety_devices"]),
        "psm_guidance": {
            "note": "아래 안전장치는 PSM 위험성평가 시 IPL로 고려해야 합니다.",
            "required_analyses": ["LOPA (방호계층분석)", "SIL (안전계전시스템 등급)"],
        },
    }


@app.get("/api/v1/pid/symbol-classes", tags=["참조 데이터"])
async def get_symbol_classes():
    """지원하는 P&ID 심볼 클래스 목록을 반환합니다."""
    from src.config.settings import settings
    from src.modules.symbol_detector.detector import DetectedSymbol

    classes = []
    for i, name in enumerate(settings.PID_SYMBOL_CLASSES):
        temp = DetectedSymbol(
            class_id=i, class_name=name,
            confidence=1.0, bbox=(0, 0, 0, 0),
        )
        classes.append({
            "id": i,
            "name": name,
            "korean_name": temp.korean_name,
            "category": _categorize_symbol(name),
        })

    return {
        "total_classes": len(classes),
        "classes": classes,
    }


def _categorize_symbol(name: str) -> str:
    """심볼 카테고리 분류"""
    if "valve" in name:
        return "밸브류"
    if name in ("pump", "compressor", "tank", "vessel", "reactor",
                "heat_exchanger", "column", "drum", "filter"):
        return "장치류"
    if "indicator" in name or "transmitter" in name or name in ("control_loop", "alarm"):
        return "계장류"
    if name in ("pipe_line", "reducer", "tee", "elbow", "flange", "spectacle_blind"):
        return "배관류"
    if name in ("rupture_disc", "flame_arrestor", "safety_shower", "emergency_shutoff"):
        return "안전장치 (PSM)"
    return "기타"


# ============================================================
# 서버 실행
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
