# dataset.py

import torch
import config
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
class Dataset:
    def __init__(self, split):
        print(f"{split} 데이터셋 불러오는 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_ID)
        
        # 데이터셋의 구조를 명시하기 위한 특수 토큰 추가
        special_tokens_dict = {'sep_token': '[SEP]'}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        
        raw_dataset = load_dataset(config.DATASET_ID, split=f"{split}")

        self.instructions = []
        self.outputs = []
        for item in tqdm(raw_dataset, desc=f"{split} 데이터셋 처리 중..."):
            self.instructions.append(item['instruction'])
            self.outputs.append(item['output'])
        
        self.vocab_size = self.tokenizer.vocab_size
        print("완료됨.")

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        instruction = self.instructions[idx]
        output = self.outputs[idx]

        # <s> 지시문 [SEP] 결과문 </s>
        prompt = f"<s>{instruction}[SEP]{output}</s>"
        encoded = self.tokenizer.encode(prompt, add_special_tokens=False)

        # 토크나이저에 직접 최대 길이와 자르기(truncation) 옵션을 전달
        encoded = self.tokenizer.encode(
        prompt,
        add_special_tokens=False,
        max_length=config.BLOCK_SIZE, # 최대 길이를 BLOCK_SIZE로 설정
        truncation=True               # 길이가 초과되면 경고 없이 잘라내기 활성화
        )
            
        return torch.tensor(encoded, dtype=torch.long)

def collate_fn(batch):
    # 길이가 다르면 강제로 의미없는 [PAD]토큰으로 넣어 배치에 맞게 크기를 조절함
    pad_token_id = 0 # kcbert-base의 PAD 토큰 ID

    # 배치 내에서 가장 긴 시퀀스의 길이를 찾음
    max_len = max(len(seq) for seq in batch)
    
    padded_x = []
    padded_y = []

    for seq in batch:
        # 입력(x)과 목표(y) 생성
        x = seq[:-1]
        y = seq[1:]
        
        # 패딩 추가
        padded_x.append(torch.cat([x, torch.full((max_len - len(x),), pad_token_id, dtype=torch.long)]))
        # 손실 계산에서 패딩 부분은 무시하도록 -100으로 채움
        padded_y.append(torch.cat([y, torch.full((max_len - len(y),), -100, dtype=torch.long)]))

    return torch.stack(padded_x), torch.stack(padded_y)