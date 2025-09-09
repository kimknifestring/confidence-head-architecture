# model.py

import torch
import torch.nn as nn
from torch.nn import functional as F
import config
import math

class TransformerLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        print("모델을 불러오는 중...")
        super().__init__()
        # 단어들을 벡터 임베딩
        self.token_embedding_table = nn.Embedding(vocab_size, config.N_EMBD)
        # 문장에서의 위치 정보를 임베딩(이것과 임베딩된 벡터를 잘 사용하면 문맥을 파악하는 것이 가능해짐)
        self.position_embedding_table = nn.Embedding(config.BLOCK_SIZE, config.N_EMBD)

        self.blocks = nn.Sequential(*[Block(config.N_EMBD, n_head=config.N_HEAD) for _ in range(config.N_LAYER)])
        self.ln_f = nn.LayerNorm(config.N_EMBD)
        self.lm_head = nn.Linear(config.N_EMBD, vocab_size)

        for block in self.blocks:
            # 어텐션 서브층의 신뢰도 헤드 편향을 수정
            if hasattr(block, 'confidence_head_sa'):
                torch.nn.init.constant_(block.confidence_head_sa[0].bias, config.INITIAL_BIAS)
            # 피드포워드 서브층의 신뢰도 헤드 편향을 수정
            if hasattr(block, 'confidence_head_ffwd'):
                torch.nn.init.constant_(block.confidence_head_ffwd[0].bias, config.INITIAL_BIAS)
                
        print("완료됨")

    def forward(self, idx, targets=None,log_gates=False):
        B, T = idx.shape
        # 단어가 가지는 의미
        tok_emb = self.token_embedding_table(idx) # 결과 모양:(B,T,C)
        # 위치가 가지는 의미
        pos_emb = self.position_embedding_table(torch.arange(T, device=config.DEVICE)) # 결과 모양: (T, C)
        # 원랜 형식이 맞지 않아 더할 수 없지만 Pytorch의 브로드캐스팅(작은 텐서를 자동으로 확장하여 큰 텐서와 맞춰 연산이 가능하게 만듬)을 사용하여 더할 수 있음
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x, log_gates=log_gates)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.BLOCK_SIZE:]
            logits, loss = self(idx_cond,log_gates=False)

            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next
# 단일 어텐션 헤드 (Q,K,V)
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # 층 정규화(Layer Normalization)에서 bias와 유사한 역할을 하기 때문에 bias는 비활성화하였음
        self.query = nn.Linear(config.N_EMBD, head_size, bias=False)
        self.key = nn.Linear(config.N_EMBD, head_size, bias=False)
        self.value = nn.Linear(config.N_EMBD, head_size, bias=False)

        # 하삼각행렬을 사용, 어떤 토큰에서 자신보다 미래에 있는 토큰을 보는 것을 차단함
        # 그리고 이 하삼각행렬은 학습용 파라미터가 아니니 기울기 계산에서 제외
        self.register_buffer('tril',torch.tril(torch.ones(config.BLOCK_SIZE, config.BLOCK_SIZE)))

    def forward(self,x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # scaled_dot_product_attention 으로 스케일링함
        """
        벡터의 차원이 올라갈수록 내적의 결과가 더 커지는 경향성이 있는데,
        이렇게 매우 커진 내적 값이 Softmax에 입력되면
        그 값의 확률이 1에 수렴하고 나머지 값들의 확률이 0에 수렴하게 되는 문제가 있다. 
        """

        """
        Attention Is All You Need 에서 이를 해결하기 위해
        wei를 Key 벡터 차원의 제곱근으로 나누어 Softmax에 항상
        적당한 크기의 값이 들어가도록 조절했다.
        """

        """
        제곱근으로 나누는 이유는 내적 값의 표준편차가 C의 제곱근에 비례하여 커지므로
        이를 상쇄하기 위함
        """
        # 행렬곱을 위해 k를 알맞게 전치
        # 내 q(질문)을 기준으로 다른k(키워드)들이 어떤 연관관계가 있는지 구함
        wei = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1]) # (B, T, T)

        """
        이 시점에서 wei는 문장 전체에 대한 정보를 품고 있으니 이로 학습하면
        예측이 아니라 그냥 문장을 받아쓰는 법을 배움, 정보의 제한이 필요함
        """

        # Masking 기법
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, head_size)

        # q(자기 자신)에 대한 k(다른 단어)들의 관계와 v(자신의 뜻) 모두가 하나의 정보로 가공됨
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # 헤드의 출력값들을 적절히 섞을 가중치
        self.proj = nn.Linear(num_heads * head_size, config.N_EMBD)
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # 4배 확장 - 활성화함수 - 원상복구
            # 실험적으로 찾은 효과적인 설정값
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # 드롭아웃
        self.dropout = nn.Dropout(config.DROPOUT)
        
        # 신뢰도 헤드
        self.confidence_head_sa = nn.Sequential(nn.Linear(n_embd, 1), nn.Sigmoid())
        self.confidence_head_ffwd = nn.Sequential(nn.Linear(n_embd, 1), nn.Sigmoid())

    def forward(self, x, log_gates=False):
        processed_sa = self.sa(self.ln1(x))
        # 신뢰도 계산
        # gate_sa = self.confidence_head_sa(processed_sa)
        # x = x + self.dropout(gate_sa * processed_sa)
        x = x + self.dropout(processed_sa)

        processed_ffwd = self.ffwd(self.ln2(x))
        # gate_ffwd = self.confidence_head_ffwd(processed_ffwd)
        # x = x + self.dropout(gate_ffwd * processed_ffwd)
        x = x + self.dropout(processed_ffwd)

        # if log_gates:
        #     print(f"SA Gate Avg: {gate_sa.mean().item():.4f} FFWD Gate Avg: {gate_ffwd.mean().item():.4f}")
        return x