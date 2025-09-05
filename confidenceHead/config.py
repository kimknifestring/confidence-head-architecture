# config.py

import torch
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()

# FILE_PATH = ROOT_DIR.parent.parent / '데이터셋.txt'

# Hugging Face 데이터셋 ID
DATASET_ID = "beomi/KoAlpaca-v1.1a"
# 사용할 사전 훈련된 토크나이저 ID
TOKENIZER_ID = "beomi/kcbert-base"

MODEL_DIR = ROOT_DIR / 'Model'
GRAPH_DIR = ROOT_DIR / 'Graph'
VOCAB_DIR = ROOT_DIR / 'Vocab'
MODEL_NAME = 'Transformer_model.pth'
BEST_NAME = 'BEST_model.pth'
MODEL_PATH = MODEL_DIR / MODEL_NAME
BEST_PATH = MODEL_DIR / BEST_NAME

# 하이퍼파라미터
BATCH_SIZE = 32          # 한 번에 처리할 데이터 묶음의 크기
BLOCK_SIZE = 30         # 모델이 한 번에 보는 문맥의 길이
MAX_ITERS = 15000         # 총 훈련 반복 횟수
EVAL_INTERVAL = 500      # 중간 평가를 하는 간격
LEARNING_RATE = 3e-4     # 학습률
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
T_RESTARTCYCLE = 1000
T_MULTIPLIER = 1
# 트랜스포머 모델 하이퍼파라미터
N_EMBD = 128             # 임베딩 차원의 크기
N_HEAD = 8               # 사용할 어텐션 헤드의 개수
N_LAYER = 4              # 쌓을 트랜스포머 블록의 개수
DROPOUT = 0.2            # 드롭아웃 비율

# 그냥 변수들
MAX_TOKEN=100

