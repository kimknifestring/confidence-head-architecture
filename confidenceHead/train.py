# train.py

import torch
import config
from model import TransformerLanguageModel
from dataset import Dataset, collate_fn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import os
import json

# 최고 성능 모델 저장
def bestSave(val_loss, best_val_loss):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # 최고 모델 저장 경로
        best_model_path = config.MODEL_DIR / 'best_model.pth'
        torch.save(model.state_dict(), best_model_path)
        print(f"최고 성능 모델이 갱신되었습니다. loss: {best_val_loss}")
    return best_val_loss

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
optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

# 학습률 스케쥴러

# scheduler = CosineAnnealingLR(optimizer, T_max=config.MAX_ITERS, eta_min=0)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.T_RESTARTCYCLE, T_mult=config.T_MULTIPLIER , eta_min=1e-5)

# 손실 값을 저장할 리스트 생성
train_losses = []
val_losses = []
steps = []

# 가장 낮은 검증 손실을 추적하기 위한 변수
best_val_loss = float('inf') 

# 훈련 루프
train_iterator = iter(train_loader)
for iter_num in range(1, config.MAX_ITERS + 1):
    # 데이터 로더에서 배치를 가져오고, 데이터가 소진되면 다시 처음부터 가져옴
    try:
        xb, yb = next(train_iterator)
    except StopIteration:
        train_iterator = iter(train_loader)
        xb, yb = next(train_iterator)
    
    xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
    logits, loss = model(xb, yb)

    if iter_num % config.EVAL_INTERVAL == 0 or iter_num == config.MAX_ITERS:
        with torch.no_grad():
            # 검증 데이터 로더에서 배치를 하나 가져와서 평가
            val_iterator = iter(val_loader)
            xb_val, yb_val = next(val_iterator)
            xb_val, yb_val = xb_val.to(config.DEVICE), yb_val.to(config.DEVICE)
            
            logits, val_loss = model(xb_val, yb_val)
            print(f"step {iter_num}: validation loss {val_loss.item():.4f}")
            print(f"step {iter_num}: train loss {loss.item():.4f}")

            best_val_loss = bestSave(val_loss.item(), best_val_loss)
            
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            steps.append(iter_num)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

# 그래프 데이터 저장
loss_data = {
    'steps': steps,
    'train_losses': train_losses,
    'val_losses': val_losses
}
with open(config.GRAPH_DIR/'graph_data.json', 'w') as f:
    json.dump(loss_data, f, indent=4)
print("손실 데이터가 'graph_data.json' 파일로 저장되었습니다.")

# 모델 저장
config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), config.MODEL_PATH)
print(f"모델이 {config.MODEL_PATH} 파일에 저장되었습니다.")
