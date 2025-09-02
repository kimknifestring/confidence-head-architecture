# train.py

import torch
import config
from model import TransformerLanguageModel
from dataset import Dataset 

# 데이터 준비
dataset = Dataset()
vocab_size = dataset.vocab_size

# 모델 생성
model = TransformerLanguageModel(vocab_size)
m = model.to(config.DEVICE)

# 훈련 가능한 파라미터 수 출력
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters Model')

# 옵티마이저 생성
optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

# 훈련 루프
for iter in range(1,config.MAX_ITERS+1):
    # print(f"{iter} iter")
    # 데이터 배치 가져오기
    xb, yb = dataset.get_batch('train')
    xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
    
    # 모델 예측 및 손실 계산
    logits, loss = model(xb, yb)

    # 매 EVAL_INTERVAL 마다 손실 출력
    if iter % config.EVAL_INTERVAL == 0 or iter == config.MAX_ITERS - 1:
        # 평가할 땐 기울기 계산 필요 X
        with torch.no_grad():
            # 검증 데이터에 대한 손실을 계산
            xb_val, yb_val = dataset.get_batch('val')
            xb_val, yb_val = xb_val.to(config.DEVICE), yb_val.to(config.DEVICE)
            logits, val_loss = model(xb_val, yb_val)
            print(f"step {iter}: validation loss {val_loss.item()}")
            print(f"step {iter}: train loss {loss.item()}")
    
    # 역전파를 통한 파라미터 업데이트
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 모델 저장
config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), config.MODEL_PATH)
print(f"모델이 {config.MODEL_PATH} 파일에 저장되었습니다.")
