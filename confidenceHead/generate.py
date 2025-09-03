# generate.py

import torch
import config
from model import TransformerLanguageModel
from dataset import Dataset
from transformers import AutoTokenizer

# 데이터 준비
print("토크나이저 준비하는 중...")
tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_ID)
vocab_size = tokenizer.vocab_size

# 모델 불러오기
model = TransformerLanguageModel(vocab_size)
m = model.to(config.DEVICE)

# 저장된 가중치(state_dict) 불러오기
print('가중치를 불러오는 중...')
print(f"Loading model from {config.BEST_PATH}...")
m.load_state_dict(torch.load(config.BEST_PATH, weights_only=True))
m.eval() # 모델을 평가 모드로 설정
print("Model loaded successfully.")

# 텍스트 생성
# start_context = '\n'
while True:
    start_context = input("예측할 문장을 입력:")
    # 사용자가 긴 문장을 입력할 경우를 대비해 자르기 옵션 추가
    context = torch.tensor([
        tokenizer.encode(
            start_context,
            max_length=config.BLOCK_SIZE,
            truncation=True
        )
    ], dtype=torch.long, device=config.DEVICE)

    print("\n--- 트랜스포머 아키텍쳐로 생성된 텍스트: ---")
    generated_tokens = []
    for token_tensor in m.generate(context, max_new_tokens=config.MAX_TOKEN):
        generated_tokens.append(token_tensor.item())

    # 모든 토큰 생성이 끝난 후, 한 번에 디코딩하여 출력
    full_text = tokenizer.decode(generated_tokens)
    print(start_context + full_text)

    print()