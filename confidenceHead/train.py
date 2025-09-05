# train.py

import torch
import config
from model import TransformerLanguageModel
from dataset import Dataset, collate_fn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import os
import json
import numpy as np

@torch.no_grad()
def evaluate(model, data_loader):
    model.eval() # 모델을 평가 모드로 설정
    losses = []
    for xb, yb in data_loader:
        xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
        _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train() # 모델을 다시 훈련 모드로 설정
    # 
    return np.mean(losses)


# 데이터 준비
print("데이터셋 준비 중...")
train_dataset = Dataset('train[:90%]') # 훈련 데이터로 90% 사용
val_dataset = Dataset('train[90%:]')  # 검증 데이터로 10% 사용

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
vocab_size = train_dataset.vocab_size
print("데이터셋 준비 완료.")

# 모델 생성
model = TransformerLanguageModel(vocab_size)
m = model.to(config.DEVICE)

if os.path.exists(config.MODEL_PATH):
    print(f"{config.MODEL_PATH} 파일에서 기존 가중치를 불러옵니다.")
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
else:
    print("새로운 모델의 훈련을 시작합니다.")

# 훈련 가능한 파라미터 수 출력
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters Model')

# 옵티마이저 생성
optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,weight_decay=0.05)

# 학습률 스케쥴러

# scheduler = CosineAnnealingLR(optimizer, T_max=config.MAX_ITERS, eta_min=0)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.T_RESTARTCYCLE, T_mult=config.T_MULTIPLIER , eta_min=1e-5)

# 손실 값을 저장할 리스트 생성
train_losses = []
val_losses = []
steps = []

# 가장 낮은 검증 손실을 추적하기 위한 변수
best_val_loss = float('inf') 

# 학습이 원할하지 않을 시 종료하기 위한 인내심 변수
patience_counter = 0

# 훈련 루프
model.train()
train_iterator = iter(train_loader)
for iter_num in range(1, config.MAX_ITERS + 1):
    try:
        xb, yb = next(train_iterator)
    except StopIteration:
        train_iterator = iter(train_loader)
        xb, yb = next(train_iterator)
    
    xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

    # 평가 및 조기 종료
    if iter_num % config.EVAL_INTERVAL == 0 or iter_num == config.MAX_ITERS:
        
        # 전체 검증 데이터로 평가
        current_val_loss = evaluate(model, val_loader)
        
        print(f"step {iter_num}: validation loss {current_val_loss:.4f}")
        print(f"step {iter_num}: train loss {loss.item():.4f}")

        # 그래프 데이터 저장
        train_losses.append(loss.item())
        val_losses.append(current_val_loss)
        steps.append(iter_num)

        # 최고 성능 모델 갱신 및 조기 종료 카운터 관리
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(model.state_dict(), config.BEST_PATH)
            print(f"최고 성능 모델 갱신 loss: {best_val_loss:.4f}")
            patience_counter = 0 
        else:
            patience_counter += 1 
            print(f"성능 개선 없음. Patience: {patience_counter}/{config.PATIENCE}")
        
        # 조기 종료 조건 확인
        if patience_counter >= config.PATIENCE:
            print(f"{config.PATIENCE}번의 평가 동안 성능 개선이 없어 훈련을 조기 종료합니다.")
            break

# 그래프 데이터 저장
loss_data = {
    'steps': steps,
    'train_losses': train_losses,
    'val_losses': val_losses
}
with open(config.GRAPH_DIR/'graph_data.json', 'w') as f:
    json.dump(loss_data, f, indent=4)
print("손실 데이터가 'graph_data.json' 파일로 저장되었습니다.")