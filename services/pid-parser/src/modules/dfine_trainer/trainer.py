"""
M01-02: dfine_trainer
━━━━━━━━━━━━━━━━━━━━
D-Fine 모델 커스텀 학습 파이프라인

[역할]
- D-Fine 모델을 P&ID 도면 심볼에 맞게 파인튜닝(재학습)합니다.
- Objects365 사전학습 가중치에서 시작하여,
  Digitize-PID 데이터셋으로 P&ID 특화 심볼을 학습합니다.

[D-Fine이란?]
- ICLR 2025 Spotlight 논문에서 발표된 최신 객체 감지 모델
- YOLO보다 정확하면서 속도도 비슷 (COCO mAP 57.1%)
- Apache 2.0 라이선스 → SaaS 상용화에 제약 없음
- NMS(Non-Maximum Suppression) 후처리 불필요 → End-to-End

[비유]
이미 '물건 찾기 달인'인 AI(Objects365 학습)에게
'도면 심볼 찾기'라는 새로운 전문 기술을 가르치는 과정입니다.

[학습 로드맵]
Phase 1) Objects365 사전학습 가중치 로드
Phase 2) Digitize-PID 500장으로 파인튜닝
Phase 3) 합성 데이터 2,000장 추가
Phase 4) 파일럿 고객 실제 도면으로 추가 학습
"""

import json
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


@dataclass
class TrainingConfig:
    """D-Fine 학습 설정

    [쉬운 설명]
    AI를 학습시킬 때 조정하는 '다이얼'들입니다.
    학습률(learning_rate)은 AI가 한 번에 얼마나 많이 배울지,
    에폭(epochs)은 교재를 몇 번 반복할지를 의미합니다.
    """
    # ── 모델 설정 ──
    model_name: str = "dfine_hgnetv2_l"          # D-Fine Large (HGNetV2 백본)
    pretrained_weights: str = "dfine_l_obj365.pth"  # Objects365 사전학습
    num_classes: int = 42                          # P&ID 심볼 클래스 수

    # ── 학습 하이퍼파라미터 ──
    learning_rate: float = 1e-4                    # 파인튜닝이므로 작은 값
    batch_size: int = 8                            # GPU 메모리에 따라 조절
    epochs: int = 50                               # 총 학습 에폭
    warmup_epochs: int = 3                         # 초기 워밍업 기간
    weight_decay: float = 1e-4                     # 과적합 방지

    # ── 데이터 설정 ──
    train_data_dir: str = "data/digitize_pid/train"
    val_data_dir: str = "data/digitize_pid/val"
    image_size: int = 640                          # D-Fine 최적 입력 크기

    # ── 데이터 증강 ──
    augmentation: dict = field(default_factory=lambda: {
        "horizontal_flip": True,                   # 좌우 반전
        "vertical_flip": False,                    # 상하 반전 (도면은 방향 중요)
        "rotation_range": 5,                       # 미세 회전 (스캔 기울기 대응)
        "brightness_range": (0.8, 1.2),            # 밝기 변화
        "noise_injection": True,                   # 노이즈 추가 (열화 도면 대응)
    })

    # ── 출력 설정 ──
    output_dir: str = "outputs/training"
    save_interval: int = 5                         # 5 에폭마다 체크포인트 저장
    device: str = "cuda"                           # GPU 사용


class DFineTrainer:
    """D-Fine 모델 학습 관리자

    [사용법]
    config = TrainingConfig(epochs=50)
    trainer = DFineTrainer(config)

    # 1) 데이터셋 준비 (COCO 포맷)
    trainer.prepare_dataset()

    # 2) 학습 실행
    trainer.train()

    # 3) 평가
    metrics = trainer.evaluate()
    print(f"mAP: {metrics['mAP']}")
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.training_log: list[dict] = []

        logger.info(f"D-Fine 학습기 초기화: {config.model_name}")
        logger.info(f"클래스 수: {config.num_classes}, 에폭: {config.epochs}")

    def prepare_dataset(self) -> dict:
        """학습 데이터셋을 COCO 포맷으로 준비합니다.

        [COCO 포맷이란?]
        AI 학습에 가장 널리 쓰이는 데이터 형식입니다.
        - images: 이미지 파일 목록
        - annotations: 각 이미지에서 심볼의 위치(bbox)와 종류(category)
        - categories: 심볼 종류 목록 (밸브, 펌프, 탱크 등)

        [P&ID 심볼 카테고리]
        1) 밸브류: gate_valve, globe_valve, ball_valve 등
        2) 장치류: pump, tank, heat_exchanger 등
        3) 계장류: pressure_indicator, flow_transmitter 등
        4) 안전장치: relief_valve, rupture_disc 등 (PSM 핵심!)
        """
        from src.config.settings import settings

        categories = [
            {"id": i, "name": name}
            for i, name in enumerate(settings.PID_SYMBOL_CLASSES)
        ]

        dataset_info = {
            "format": "COCO",
            "num_categories": len(categories),
            "categories": categories,
            "train_dir": self.config.train_data_dir,
            "val_dir": self.config.val_data_dir,
        }

        logger.info(f"데이터셋 준비 완료: {len(categories)}개 클래스")
        return dataset_info

    def build_model(self):
        """D-Fine 모델을 구성합니다.

        [구조 설명]
        D-Fine은 3가지 핵심 요소로 구성됩니다:
        1) HGNetV2 백본: 이미지에서 특징을 추출하는 '눈' 역할
        2) Encoder: 추출된 특징을 정리하는 '뇌' 역할
        3) FDR Decoder: 물체의 정확한 위치를 잡아내는 '손' 역할
           - Fine-grained Distribution Refinement (미세 분포 정제)
           - 이것이 D-Fine만의 핵심 기술!
        """
        logger.info("D-Fine 모델 구성 시작...")

        # ── D-Fine 모델 정의 ──
        # 실제 구현은 D-Fine 공식 저장소의 모델을 import하여 사용
        model_config = {
            "backbone": {
                "name": "HGNetV2-L",
                "pretrained": True,
                "freeze_at": 0,            # 전체 백본 파인튜닝
            },
            "encoder": {
                "name": "HybridEncoder",
                "in_channels": [512, 1024, 2048],
                "feat_strides": [8, 16, 32],
                "hidden_dim": 256,
                "num_encoder_layers": 1,
                "pe_temperature": 10000,
            },
            "decoder": {
                "name": "DFineDecoder",        # 핵심 디코더
                "hidden_dim": 256,
                "num_queries": 300,             # 최대 감지 객체 수
                "num_decoder_layers": 6,
                "num_head": 8,
                "reg_max": 32,                  # FDR 분포 해상도
                "num_classes": self.config.num_classes,
            },
        }

        logger.info(f"모델 구성: {json.dumps(model_config['decoder'], indent=2)}")
        self.model_config = model_config
        return model_config

    def build_training_pipeline(self) -> dict:
        """학습 파이프라인을 구성합니다.

        [파이프라인 구성]
        1) 데이터 로더: 이미지를 배치로 묶어서 GPU에 전달
        2) 옵티마이저: AdamW (가중치 감쇠 포함)
        3) 스케줄러: 학습률을 점진적으로 줄여가며 학습
        4) 손실 함수: D-Fine 특유의 FDR Loss + Hungarian Matching
        """
        pipeline = {
            "optimizer": {
                "type": "AdamW",
                "lr": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "betas": (0.9, 0.999),
            },
            "lr_scheduler": {
                "type": "CosineAnnealingLR",
                "T_max": self.config.epochs,
                "eta_min": self.config.learning_rate * 0.01,
            },
            "loss": {
                # D-Fine의 핵심 손실 함수
                "cls_loss": "FocalLoss",             # 클래스 분류 손실
                "reg_loss": "FDR_GIoU_Loss",         # FDR 분포 회귀 손실
                "matcher": "HungarianMatcher",       # 예측-정답 매칭
            },
            "data_loader": {
                "batch_size": self.config.batch_size,
                "num_workers": 4,
                "pin_memory": True,
                "shuffle": True,
            },
        }

        logger.info("학습 파이프라인 구성 완료")
        return pipeline

    def train(self) -> dict:
        """D-Fine 모델 학습을 실행합니다.

        [학습 프로세스]
        1) 사전학습 가중치 로드 (Objects365)
        2) P&ID 데이터로 파인튜닝 시작
        3) 매 에폭마다 검증 데이터로 성능 체크
        4) 최고 성능 모델 자동 저장

        Returns:
            학습 결과 요약 (최종 mAP, 학습 시간 등)
        """
        logger.info("=" * 60)
        logger.info("D-Fine P&ID 파인튜닝 학습 시작")
        logger.info(f"에폭: {self.config.epochs}, 배치: {self.config.batch_size}")
        logger.info(f"학습률: {self.config.learning_rate}")
        logger.info("=" * 60)

        # 모델 및 파이프라인 구성
        model_config = self.build_model()
        pipeline = self.build_training_pipeline()

        # 학습 실행 (프레임워크 구조)
        training_result = {
            "status": "ready",
            "model_config": model_config,
            "pipeline": pipeline,
            "training_plan": self._generate_training_plan(),
        }

        return training_result

    def _generate_training_plan(self) -> list[dict]:
        """에폭별 학습 계획을 생성합니다."""
        plan = []
        for epoch in range(1, self.config.epochs + 1):
            phase = "warmup" if epoch <= self.config.warmup_epochs else "training"
            lr_factor = min(1.0, epoch / self.config.warmup_epochs) if phase == "warmup" else 1.0

            plan.append({
                "epoch": epoch,
                "phase": phase,
                "lr": self.config.learning_rate * lr_factor,
                "save_checkpoint": epoch % self.config.save_interval == 0,
            })
        return plan

    def evaluate(self, model_path: Optional[str] = None) -> dict:
        """학습된 모델의 성능을 평가합니다.

        [평가 지표]
        - mAP@50: IoU 0.5 기준 평균 정밀도 (50% 이상 겹치면 정답)
        - mAP@50:95: 다양한 IoU 기준의 평균 (더 엄격한 평가)
        - Recall: 실제 심볼을 얼마나 잘 찾았는지
        - Precision: 찾은 것 중 실제 심볼인 비율

        [목표]
        심볼 인식 정확도 95% 이상 (mAP@50 기준)
        """
        metrics = {
            "mAP_50": 0.0,
            "mAP_50_95": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "per_class_ap": {},
            "evaluation_note": "모델 학습 후 실제 평가 수행 필요",
        }

        logger.info(f"평가 완료 (프레임워크): {metrics}")
        return metrics

    def export_model(self, format: str = "onnx") -> str:
        """학습된 모델을 배포용으로 변환합니다.

        [변환 포맷]
        - ONNX: 범용 포맷, NCP GPU 서버에서 최적 추론
        - TorchScript: PyTorch 네이티브, 빠른 추론
        - TensorRT: NVIDIA GPU 최적화 (가장 빠름, L4/V100용)

        Args:
            format: 변환 포맷 ("onnx", "torchscript", "tensorrt")
        """
        output_path = Path(self.config.output_dir) / f"dfine_pid.{format}"
        logger.info(f"모델 변환: {format} → {output_path}")
        return str(output_path)


def generate_dfine_training_yaml(config: TrainingConfig) -> str:
    """D-Fine 학습 YAML 설정 파일을 생성합니다.

    [설명]
    D-Fine은 YAML 파일로 학습 설정을 관리합니다.
    이 함수는 PSM-SafetyTwin에 맞는 설정 파일을 자동 생성합니다.
    """
    yaml_content = f"""# D-Fine P&ID 파인튜닝 설정
# PSM-SafetyTwin SVC-01
# 생성: dfine_trainer 모듈

task: detection
model: {config.model_name}

# 데이터 설정
train_dataloader:
  dataset:
    type: CocoDetection
    img_folder: {config.train_data_dir}/images
    ann_file: {config.train_data_dir}/annotations.json
  batch_size: {config.batch_size}
  num_workers: 4
  shuffle: true

val_dataloader:
  dataset:
    type: CocoDetection
    img_folder: {config.val_data_dir}/images
    ann_file: {config.val_data_dir}/annotations.json
  batch_size: {config.batch_size}
  num_workers: 4

# 모델 설정
model:
  backbone:
    type: HGNetV2
    name: L
    pretrained: true
    return_idx: [1, 2, 3]

  encoder:
    type: HybridEncoder
    in_channels: [512, 1024, 2048]
    feat_strides: [8, 16, 32]
    hidden_dim: 256
    nhead: 8
    dim_feedforward: 1024
    num_encoder_layers: 1

  decoder:
    type: DFineDecoder
    feat_channels: [256, 256, 256]
    feat_strides: [8, 16, 32]
    hidden_dim: 256
    num_levels: 3
    num_queries: 300
    num_decoder_layers: 6
    num_denoising: 100
    eval_idx: -1
    num_classes: {config.num_classes}
    reg_max: 32

# 학습 설정
optimizer:
  type: AdamW
  lr: {config.learning_rate}
  weight_decay: {config.weight_decay}
  betas: [0.9, 0.999]

lr_scheduler:
  type: CosineAnnealingLR
  T_max: {config.epochs}
  eta_min: {config.learning_rate * 0.01}

epochs: {config.epochs}
warmup_epochs: {config.warmup_epochs}

# 체크포인트
checkpoint:
  save_dir: {config.output_dir}/checkpoints
  save_interval: {config.save_interval}

# 이미지 크기
input_size: [{config.image_size}, {config.image_size}]

# 데이터 증강
augmentation:
  horizontal_flip: {str(config.augmentation['horizontal_flip']).lower()}
  rotation_range: {config.augmentation['rotation_range']}
  brightness_range: {list(config.augmentation['brightness_range'])}
"""
    return yaml_content
