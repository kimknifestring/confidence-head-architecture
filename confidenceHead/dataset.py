# dataset.py

import torch
import config
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
class Dataset:
    def __init__(self):
        print("데이터셋 불러오는 중...")
        # Hugging Face Hub에서 사전 훈련된 토크나이저 불러오기
        self.tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_ID)
        
        # Hugging Face Hub에서 데이터셋 불러오기
        raw_dataset = load_dataset(config.DATASET_ID)
        
        # 데이터셋의 각 항목을 하나의 긴 텍스트로 결합
        # Ko-Alpaca는 instruction, output 컬럼이 있으므로 이를 합침
        text_data = ""
        for item in tqdm(raw_dataset['train'], desc="데이터셋 텍스트 합치는 중..."):
            text_data += item['instruction'] + "\n" + item['output'] + "\n"
            
        # 전체 텍스트를 인코딩하고 데이터 분할
        data = torch.tensor(self.tokenizer.encode(text_data), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

        # 필요한 속성들 설정
        self.vocab_size = self.tokenizer.vocab_size
        self.block_size = config.BLOCK_SIZE
        self.batch_size = config.BATCH_SIZE
        print("완료됨.")

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y