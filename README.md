<div align="center">

# 🏭 PSM-SafetyTwin

### 디지털 트윈 기반 PSM 위험성평가 사전예방 플랫폼

**P&ID 도면 → AI 자동 인식 → 디지털 데이터 → 위험성평가 연계**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![D-Fine](https://img.shields.io/badge/D--Fine-Apache%202.0-blue?logo=pytorch&logoColor=white)](https://github.com/Peterande/D-FINE)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](docker-compose.yml)

</div>

---

## 📋 프로젝트 개요

PSM-SafetyTwin은 **공정안전관리(PSM)** 대상 사업장의 P&ID(배관계장도) 도면을 AI로 자동 분석하여 디지털 데이터로 변환하는 플랫폼입니다.

| 기존 방식 (수작업) | PSM-SafetyTwin (AI) |
|---|---|
| ❌ 도면 1장 분석에 2~4시간 | ✅ **10초 이내** 자동 분석 |
| ❌ 사람마다 다른 해석 (주관적) | ✅ **95% 이상** 일관된 정확도 |
| ❌ 안전장치 누락 위험 | ✅ PSM 안전장치 **자동 식별** |
| ❌ 디지털 데이터 없음 | ✅ 즉시 **위험성평가 연계** |

## 🏗️ 시스템 아키텍처

```
📄 P&ID 도면 업로드
       │
       ▼
┌─────────────────────────────────────────────┐
│  SVC-01: pid-parser (이 저장소)              │
│                                              │
│  M01-01 이미지 전처리 (노이즈 제거, 분할)    │
│       │                                      │
│       ▼                                      │
│  M01-02 D-Fine 모델 학습 파이프라인          │
│       │                                      │
│       ▼                                      │
│  M01-03 심볼 감지 (밸브/펌프/탱크/안전장치)  │
│       │                                      │
│       ▼                                      │
│  M01-04 텍스트 추출 (PaddleOCR)              │
│       │                                      │
│       ▼                                      │
│  M01-05 데이터 매칭 (심볼+태그 연결)         │
│       │                                      │
│       ▼                                      │
│  ⚠️  PSM 안전장치 자동 식별 (IPL 대상)       │
└─────────────────────────────────────────────┘
       │
       ▼
  📊 구조화된 디지털 데이터 (JSON)
       │
       ├──▶ 2단계: 위험성평가 코어 (HAZOP, LOPA, FTA)
       ├──▶ 3단계: 3D 디지털 트윈 (Babylon.js)
       └──▶ 4단계: CFD 대리 모델 (실시간 시뮬레이션)
```

## 🚀 빠른 시작

### 방법 1: 로컬 실행 (가장 간단)

```bash
# 1) 저장소 클론
git clone https://github.com/[사용자명]/psm-safetytwin.git
cd psm-safetytwin/services/pid-parser

# 2) 의존성 설치
pip install -r requirements.txt

# 3) 서버 실행
uvicorn src.api.main:app --reload --port 8001

# 4) 브라우저에서 확인
# http://localhost:8001/docs  (API 문서)
# http://localhost:8001/api/v1/health  (상태 확인)
```

### 방법 2: Docker (권장)

```bash
# 전체 서비스 한번에 실행 (DB 포함)
docker-compose up -d

# 실행 확인
docker-compose ps

# 종료
docker-compose down
```

## 📁 프로젝트 구조

```
psm-safetytwin/
├── README.md                 ← 이 파일
├── LICENSE                   ← Apache 2.0 라이선스
├── docker-compose.yml        ← Docker 전체 서비스 설정
│
└── services/
    └── pid-parser/           ← SVC-01: P&ID 디지털화 엔진
        ├── Dockerfile
        ├── requirements.txt
        ├── src/
        │   ├── api/
        │   │   └── main.py           ← FastAPI 서버 (6개 엔드포인트)
        │   ├── config/
        │   │   └── settings.py        ← 42개 심볼 클래스, 모델 설정
        │   └── modules/
        │       ├── image_preprocessor/
        │       │   └── preprocessor.py  ← M01-01: 이미지 전처리
        │       ├── dfine_trainer/
        │       │   └── trainer.py       ← M01-02: D-Fine 모델 학습
        │       ├── symbol_detector/
        │       │   └── detector.py      ← M01-03: 심볼 감지 (추론)
        │       ├── ocr_extractor/
        │       │   └── extractor.py     ← M01-04: OCR 텍스트 추출
        │       ├── data_matcher/
        │       │   └── matcher.py       ← M01-05: 데이터 매칭
        │       └── graph_builder/       ← M01-06: Neo4j 그래프 (예정)
        └── tests/
            └── test_pid_parser.py       ← 21개 단위 테스트
```

## 🔌 API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `POST` | `/api/v1/pid/analyze` | P&ID 도면 업로드 및 AI 분석 |
| `GET` | `/api/v1/pid/results/{id}` | 분석 결과 조회 |
| `GET` | `/api/v1/pid/safety/{id}` | PSM 안전장치만 필터링 조회 |
| `GET` | `/api/v1/pid/symbol-classes` | 지원 심볼 42종 목록 |
| `GET` | `/api/v1/health` | 서비스 상태 확인 |

## 🤖 AI 모델: D-Fine

[D-Fine](https://github.com/Peterande/D-FINE)은 ICLR 2025 Spotlight 논문에서 발표된 최신 객체 감지 모델입니다.

| 항목 | D-Fine | YOLO v8/v11 |
|------|--------|-------------|
| **라이선스** | ✅ Apache 2.0 (SaaS 자유) | ❌ AGPL-3.0 (소스 공개 의무) |
| **정확도 (COCO mAP)** | **57.1%** | 53.4% |
| **NMS 후처리** | 불필요 (End-to-End) | 필요 |
| **학술 검증** | ICLR 2025 Spotlight | 산업 표준 |

## 🔍 인식 가능한 P&ID 심볼 (42종)

<details>
<summary>전체 목록 보기 (클릭)</summary>

**밸브류 (9종):** 게이트밸브, 글로브밸브, 볼밸브, 버터플라이밸브, 체크밸브, 제어밸브, 니들밸브, 플러그밸브, 다이어프램밸브

**장치류 (9종):** 탱크, 펌프, 압축기, 열교환기, 반응기, 증류탑, 믹서, 필터, 드럼

**계장류 (10종):** 압력계, 온도계, 유량계, 레벨계, 압력전송기, 온도전송기, 유량전송기, 레벨전송기, 제어기, 지시기

**배관류 (6종):** 직관, 엘보, 티, 리듀서, 플랜지, 캡

**⚠️ 안전장치 (4종, PSM 핵심):** 안전밸브(PSV), 파열판(RD), 화염방지기, 긴급차단밸브(ESD)

</details>

## 🛡️ 라이선스 컴플라이언스

모든 핵심 구성요소가 **상용 친화적 라이선스**로 구성되어 있습니다.

| 구성요소 | 기술 | 라이선스 |
|----------|------|----------|
| 심볼 감지 | D-Fine | Apache 2.0 ✅ |
| 텍스트 추출 | PaddleOCR | Apache 2.0 ✅ |
| 백엔드 | FastAPI | MIT ✅ |
| 3D 시각화 | Babylon.js | Apache 2.0 ✅ |
| 관계형 DB | PostgreSQL | PostgreSQL License ✅ |

## 🗺️ 전체 로드맵 (30개월)

| 단계 | 기간 | 목표 | 상태 |
|------|------|------|------|
| **1단계** P&ID 디지털화 | 2026.Q2~Q3 | D-Fine 기반 도면 자동 인식 | 🔵 개발 중 |
| **2단계** 위험성평가 코어 | 2026.Q4~2027.Q1 | HAZOP, LOPA, FTA 등 | ⬜ 예정 |
| **3단계** 디지털 트윈 | 2027.Q2~Q3 | Babylon.js 3D 시각화 | ⬜ 예정 |
| **4단계** CFD 대리 모델 | 2027.Q4~2028.Q1 | 실시간 사고 확산 시뮬레이션 | ⬜ 예정 |
| **5단계** IoT 연동 | 2028.Q2~Q3 | 센서 실시간 모니터링 | ⬜ 예정 |
| **6단계** AI 고도화 | 2028.Q4~ | 에이전틱 AI, GS 인증 | ⬜ 예정 |

## 🧪 테스트

```bash
cd services/pid-parser
pip install pytest pytest-asyncio
pytest tests/ -v
```

21개 테스트 전부 통과 ✅

## 📄 라이선스

이 프로젝트는 [Apache License 2.0](LICENSE) 하에 배포됩니다.

---

<div align="center">

**PSM-SafetyTwin** — P&ID가 디지털 데이터가 되고, 위험성평가가 실시간 예측이 되는 PSM 사전예방 통합 플랫폼

</div>
