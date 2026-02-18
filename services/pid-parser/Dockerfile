# ============================================
# PSM-SafetyTwin SVC-01: pid-parser
# D-Fine 기반 P&ID 디지털화 엔진
# NCP GPU 서버 (L4/V100) 배포용
# ============================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 환경
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 의존성 설치
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY src/ ./src/
COPY config/ ./config/

# D-Fine 모델 가중치 캐시 디렉토리
RUN mkdir -p /app/models/dfine /app/uploads /app/outputs

# 환경 변수
ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models/dfine
ENV UPLOAD_DIR=/app/uploads
ENV OUTPUT_DIR=/app/outputs

# 포트: 8001 (pid-parser 전용)
EXPOSE 8001

# FastAPI 서버 실행
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
