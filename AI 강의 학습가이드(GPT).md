# AI · 데이터 분석 학습 가이드 & 향후 로드맵

> 기준 자료: 지금까지 수행한 4개 Jupyter Notebook
> - `1-1 데이터 EDA 및 모델 학습`
> - `1-2 간단한 Perceptron 구현`
> - `2-1 토큰화·임베딩 심화`
> - `2-2 합성 데이터 생성`
>
> 향후 강의:
> - 3-1 Transfer Learning 기반의 CNN 모델 학습
> - 3-2 이미지 생성 및 평가와 모델 학습
> - 4-1 RAG 기반 Customer Service AI 에이전트 개발
> - 4-2-1 ReAct 기반 에이전트 서비스 개발
> - 4-2-2 Multi Agent 대표 패턴 학습
> - 5-1 PEFT 파라미터 효율적 튜닝
> - 5-2 Quantization

---

## 0. 전체 학습 흐름 한눈에 보기

지금까지의 과제는 서로 별개의 주제가 아니라 다음과 같이 단계적으로 확장되고 있습니다.

```text
데이터 이해
  ↓
EDA / 전처리 / 학습·평가
  ↓
Tensor / Autograd / PyTorch 학습 루프
  ↓
Neural Network 구조 직접 구현
  ↓
Tokenization / Embedding
  ↓
Attention / Transformer / GPT 구조
  ↓
[향후]
Transfer Learning / CNN
  ↓
생성모델 및 생성 결과 평가
  ↓
RAG
  ↓
ReAct Agent
  ↓
Multi-Agent
  ↓
PEFT / Quantization
```

핵심은 **“데이터를 모델 입력 형태로 바꾸고 → 모델이 예측하고 → loss를 계산하고 → 평가하고 → 실제 서비스 파이프라인으로 확장한다”**는 하나의 사고방식입니다.

---

# 1. 지금까지 배운 내용 요약

## 1-1. 데이터 EDA 및 전통적 ML 학습

### 핵심 개념

- 데이터셋 구조 파악
  - 샘플 수
  - feature 수
  - target 확인
- 기술통계
  - 평균, 중앙값, 표준편차, 최솟값, 최댓값
- 결측치 탐색
- 이상치 탐색
  - IQR
  - `Q1`, `Q3`, `IQR = Q3 - Q1`
  - 일반적인 경계: `Q1 - 1.5×IQR`, `Q3 + 1.5×IQR`
- 상관관계 분석
  - Pearson correlation
  - 상관행렬
  - heatmap
- 데이터 분할
  - train / test
  - `train_test_split`
  - target을 구간화한 뒤 `stratify` 적용
- 특성 스케일링
  - `StandardScaler`
  - train에 `fit`
  - test에는 `transform`
- 모델 학습
  - `LinearRegression`
- 회귀 평가
  - RMSE
  - MAE
  - R²
- 시각적 해석
  - 실제값 vs 예측값
  - 학습 결과 해석

### 사용한 주요 라이브러리 / 함수

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`
  - `fetch_openml`
  - `train_test_split`
  - `StandardScaler`
  - `LinearRegression`
  - `root_mean_squared_error`
  - `mean_absolute_error`
  - `r2_score`

### 데이터 처리 흐름

```text
원본 데이터
  ↓
EDA
  ├─ 결측치
  ├─ 이상치
  ├─ 분포
  └─ 상관관계
  ↓
X / y 분리
  ↓
Train / Test Split
  ↓
이상치 처리
  ↓
Scaling
  ↓
모델 학습
  ↓
예측
  ↓
RMSE / MAE / R²
  ↓
시각화 및 해석
```

### 특히 중요한 개념

#### 데이터 누수(Data Leakage)

현재 과제에서 가장 중요하게 복습할 부분 중 하나입니다.

```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

즉,

- train 데이터로만 전처리 규칙을 학습하고
- test 데이터는 그 규칙으로 변환만 해야 합니다.

이 원칙은 이후:

- CNN Transfer Learning
- 생성모델 평가
- RAG 검색 평가
- 모델 fine-tuning

등에서도 그대로 반복됩니다.

---

# 1-2. PyTorch Tensor / Autograd / Perceptron

## 핵심 개념

### Tensor

PyTorch에서 실제 모델 계산의 기본 데이터 구조입니다.

```text
numpy array
   ↓
torch.Tensor
   ↓
GPU 이동 가능
   ↓
autograd 추적 가능
```

### Autograd

`requires_grad=True`가 설정된 Tensor에 대해 계산 그래프를 만들고,

```python
loss.backward()
```

를 호출하면 파라미터의 gradient를 자동으로 계산합니다.

### 신경망의 기본 구조

과제에서는 다음 형태를 직접 구현했습니다.

```text
Input
 ↓
Linear
 ↓
ReLU
 ↓
Linear
 ↓
Logits
```

구현에 사용한 핵심 구성 요소:

- `nn.Module`
- `nn.Linear`
- `nn.ReLU`
- `forward()`

### DataLoader

```text
Dataset
   ↓
DataLoader
   ↓
mini-batch
   ↓
model
```

즉, 전체 데이터를 한꺼번에 넣는 대신 batch 단위로 학습할 수 있게 합니다.

사용 요소:

- `TensorDataset`
- `DataLoader`
- `batch_size`
- `shuffle`

### 핵심 학습 루프

반드시 머릿속으로 순서가 떠올라야 합니다.

```text
optimizer.zero_grad()
        ↓
forward
        ↓
loss 계산
        ↓
loss.backward()
        ↓
optimizer.step()
```

이 5단계를 정확히 설명할 수 있어야 합니다.

### 평가 루프

학습과 평가의 차이:

```python
model.train()
```

vs

```python
model.eval()
with torch.no_grad():
```

평가 시에는 파라미터 업데이트를 하지 않습니다.

---

# 1-3. Tokenization / Embedding

2-1 과제는 단순 NLP 사용법을 넘어 Transformer의 내부 구조를 직접 이해하는 단계입니다.

## Tokenization

텍스트를 모델이 처리할 수 있는 token sequence로 변환합니다.

```text
문장
 ↓
Tokenization
 ↓
token
 ↓
token ID
 ↓
Embedding
 ↓
dense vector
```

### 토큰화 비교

#### Word level

```text
"I'm a student of SSAFY!"
↓
단어 단위
```

장점:
- 단어가 의미 단위로 보존되기 쉬움

단점:
- vocabulary가 지나치게 커질 수 있음
- 새로운 단어 처리에 취약

#### Character level

장점:
- OOV 문제 완화

단점:
- sequence가 지나치게 길어질 수 있음

#### Subword level

현재 NLP에서 중요한 절충안입니다.

- 단어보다 작고
- 문자보다 큰 단위
- OOV 문제 완화
- vocabulary 크기와 sequence length 사이의 trade-off 조절

---

## BPE

과제에서는 `get_stats`, `merge_vocab`을 직접 구현했습니다.

핵심 아이디어:

```text
문자 단위
 ↓
인접 token pair 빈도 계산
 ↓
가장 많이 등장하는 pair 선택
 ↓
두 token 병합
 ↓
반복
```

예:

```text
l o w
l o w e r
n e w e s t
...
```

에서 자주 등장하는 pair를 계속 합쳐 새로운 subword token을 만듭니다.

여기서 중요한 것은 단순히 BPE 결과를 외우는 것이 아니라:

> “왜 token vocabulary가 데이터로부터 학습될 수 있는가?”

를 이해하는 것입니다.

---

# 1-4. Embedding

`nn.Embedding`을 사용하여 token ID를 dense vector로 변환했습니다.

```text
token id
   ↓
Embedding lookup
   ↓
H차원 벡터
```

예:

```text
token_id = 17
↓
[0.12, -0.31, ..., 0.08]
```

임베딩은 이후 모든 생성형 AI에서 사실상 기본 재료가 됩니다.

- GPT
- BERT
- RAG embedding
- 이미지/멀티모달 embedding
- vector database 검색

---

# 1-5. Transformer / GPT 내부 구현

2-1의 가장 중요한 부분입니다.

직접 다룬 핵심 요소:

- Multi-Head Self-Attention
- Q / K / V
- Scaled Dot-Product Attention
- Causal Mask
- Layer Normalization
- MLP
- Residual Connection
- Decoder Block
- Token Embedding
- Position Embedding
- Language Head

## Attention 흐름

현재 과제의 구현은 다음 구조입니다.

```text
X
 ↓
Linear
 ↓
Q, K, V
 ↓
QKᵀ / √d
 ↓
Causal Mask
 ↓
Softmax
 ↓
Attention Probability
 ↓
× V
 ↓
Multi-Head 결합
 ↓
Linear Projection
```

### 반드시 이해해야 할 핵심

Attention은 단순히 “중요한 단어를 고르는 기능”이 아니라,

> 각 위치의 Query가 모든 Key와의 관계를 계산하고, 그 결과를 이용해 Value를 가중합하여 새로운 표현을 만드는 과정

이라고 이해하는 것이 좋습니다.

---

## Decoder Block

과제에서는 Pre-Norm 구조를 직접 구현했습니다.

```text
X
 ↓
LayerNorm
 ↓
Self-Attention
 ↓
Residual Add
 ↓
LayerNorm
 ↓
MLP
 ↓
Residual Add
 ↓
Y
```

수식으로는:

```text
X' = X + MHA(LN(X))
Y  = X' + MLP(LN(X'))
```

---

## GPT2 전체 흐름

```text
Token IDs
   ↓
Token Embedding
   +
Position Embedding
   ↓
N개의 Decoder Block
   ↓
Final LayerNorm
   ↓
Language Head
   ↓
Logits
```

출력 shape도 중요합니다.

```text
Input
(B, T)

Embedding
(B, T, H)

Decoder Block
(B, T, H)

Language Head
(B, T, V)
```

- `B`: batch size
- `T`: sequence length
- `H`: hidden size
- `V`: vocabulary size

**향후 모델 구조를 이해할 때 shape 추적 능력이 매우 중요합니다.**

---

# 1-6. Prompt Engineering / 합성 데이터 생성 / LLM as a Judge

2-2에서는 모델 자체를 학습시키기보다는 LLM을 활용해 데이터를 만들고 평가하는 방법을 실습했습니다.

## Prompting

실습한 유형:

- Zero-shot
- Few-shot
- Chain-of-Thought
- Persona 기반 prompting

핵심은:

```text
하나의 모델
+
서로 다른 지시문
↓
서로 다른 출력
```

즉, 모델의 성능은 모델 자체뿐 아니라 **입력 설계(prompt design)**에도 크게 영향을 받습니다.

---

## 합성 데이터 생성

실습 흐름:

```text
원본 데이터
 ↓
Persona 여러 개 설정
 ↓
LLM에 각각 다른 prompt 전달
 ↓
다양한 변형 생성
 ↓
JSON 구조로 저장
```

과제에서는 Text-to-SQL 데이터셋을 사용하여 번역/변형된 데이터를 생성하는 구조를 만들었습니다.

---

## Structured Output

응답을 자유로운 텍스트로 받는 대신 JSON schema를 지정했습니다.

예:

```text
{
  "korean": "..."
}
```

이 방식은 이후 Agent 개발에서 매우 중요합니다.

왜냐하면 Agent는 LLM의 자연어 출력보다:

```text
행동 이름
파라미터
검색어
도구 인자
최종 답변
```

등을 구조적으로 받아야 하기 때문입니다.

---

## LLM as a Judge

생성 결과를 또 다른 LLM에게 평가시키는 구조를 경험했습니다.

```text
원본
 +
생성 결과
 ↓
Judge LLM
 ↓
점수
 +
이유
 ↓
Threshold filtering
 ↓
최종 데이터
```

즉,

```text
Generate → Evaluate → Filter
```

라는 파이프라인을 경험한 것입니다.

이 구조는 앞으로:

- 생성 모델 평가
- RAG 답변 평가
- Agent 평가
- synthetic data 품질관리

로 직접 연결됩니다.

---

# 2. 이전 내용과 향후 내용의 연계성

# 2-1. Transfer Learning 기반 CNN

## 연결되는 기존 개념

현재까지 배운:

- 데이터 분할
- 전처리
- batch
- Tensor
- `nn.Module`
- loss
- optimizer
- 학습 / 평가 모드
- overfitting
- evaluation

이 그대로 다시 등장합니다.

새롭게 추가되는 핵심은:

```text
CNN architecture
+
Pretrained model
+
Transfer Learning
```

### 특히 복습할 것

```text
Dataset
→ Transform
→ DataLoader
→ Pretrained CNN
→ classifier head
→ loss
→ backward
→ optimizer.step()
```

즉, **학습 루프는 거의 같은데 모델의 입력이 이미지가 되고 구조가 더 커지는 것**이라고 이해하면 부담이 줄어듭니다.

---

# 2-2. 이미지 생성 및 평가와 모델 학습

현재 2-2에서 배운:

- 생성
- Prompt Engineering
- 데이터 증강
- LLM as a Judge
- 평가 결과 기반 filtering

이 그대로 연결됩니다.

차이가 있다면 생성 대상이 텍스트에서 이미지로 확장됩니다.

```text
Text Prompt
 ↓
Image Generation Model
 ↓
Generated Image
 ↓
Evaluation
 ↓
Filtering / Model Improvement
```

따라서 2-2는 생각보다 중요한 선행 내용입니다.

---

# 2-3. RAG 기반 Customer Service AI Agent

여기서부터는 “모델 하나를 학습하는 것”에서 “AI 시스템을 조립하는 것”으로 관점이 바뀝니다.

핵심 구조:

```text
사용자 질문
 ↓
질문 임베딩
 ↓
Vector Search
 ↓
관련 문서 검색
 ↓
Context 구성
 ↓
LLM
 ↓
답변
```

## 지금까지와 연결

### Embedding

2-1에서 배운 `Embedding` 개념

↓

RAG의 벡터 검색으로 연결

### Tokenization

LLM 입력 처리 구조 이해에 연결

### Transformer

LLM이 문맥을 처리하는 근본 구조 이해에 연결

### Data Processing

1-1의 데이터 전처리 경험

↓

문서 chunking / cleaning / metadata 처리로 연결

### Structured Output

2-2 경험

↓

Agent의 tool call / structured response로 연결

---

# 2-4. ReAct Agent

여기서는 LLM이 단순히 답변을 생성하는 것이 아니라,

```text
Reasoning
 ↓
Action
 ↓
Observation
 ↓
Reasoning
 ↓
Action
...
 ↓
Final Answer
```

형태의 반복적인 문제 해결 구조를 다룹니다.

기존 지식과 연결하면:

```text
Prompt Engineering
        +
Structured Output
        +
LLM
        +
외부 도구
        =
Agent
```

즉, 2-2가 **LLM에게 잘 시키는 법**이라면,

4장은 **LLM이 외부 세계와 상호작용하도록 만드는 법**으로 볼 수 있습니다.

---

# 2-5. Multi-Agent

여기서는 Agent 하나가 모든 일을 하는 대신 역할을 나눕니다.

예:

```text
User
 ↓
Manager Agent
 ├─ Search Agent
 ├─ Analysis Agent
 └─ Response Agent
```

또는:

```text
Planner
 ↓
Executor
 ↓
Reviewer
```

이때 지금까지 배운 데이터 흐름 사고방식이 그대로 필요합니다.

각 Agent에 대해 항상 확인해야 합니다.

```text
Input
 ↓
Processing
 ↓
Output
 ↓
Next Agent
```

---

# 2-6. PEFT

PEFT는 모델 자체를 전부 업데이트하지 않고 일부 파라미터만 효율적으로 학습하는 방법입니다.

지금까지 배운:

- `nn.Module`
- 파라미터
- gradient
- optimizer
- backward
- model training

이해도가 높을수록 훨씬 쉽게 이해됩니다.

특히 중요한 질문:

> “학습할 때 실제로 어떤 파라미터의 gradient가 계산되고 업데이트되는가?”

이 질문을 이해하면 PEFT가 단순 암기에서 구조 이해로 바뀝니다.

대표적으로 LoRA 같은 방식은:

```text
기존 거대 모델
   ↓
대부분 parameter freeze
   ↓
작은 추가 학습 parameter만 train
   ↓
메모리 / 계산량 절감
```

이라는 관점으로 접근하면 됩니다.

---

# 2-7. Quantization

Quantization은 모델의 파라미터/계산 표현을 더 낮은 비트 수로 표현하여:

- 메모리 감소
- 추론 비용 감소
- 배포 효율 향상

등을 노리는 방법입니다.

여기서도 지금까지 배운 Tensor와 파라미터 개념이 그대로 필요합니다.

```text
모델 parameter
 ↓
표현 정밀도 낮춤
 ↓
메모리/연산 비용 감소
```

---

# 3. 앞으로의 공부 로드맵 & 우선순위

# 핵심 TOP 3

## TOP 1. Tensor Shape + 데이터 흐름 추적

가장 중요합니다.

앞으로 모델이 복잡해질수록 코드를 전부 외우는 것보다:

```text
입력 shape
→ 각 레이어
→ 출력 shape
```

를 추적하는 능력이 중요합니다.

### 반드시 익숙해질 shape

```text
Tabular
(N, F)

Image
(B, C, H, W)

Text IDs
(B, T)

Embedding
(B, T, H)

Attention score
(B, heads, T, T)

GPT logits
(B, T, V)
```

### 공부 방법

노트북에서 모든 중요한 줄에 직접 주석을 달아보세요.

```python
print(x.shape)
```

그리고 각 줄마다:

```text
왜 이 shape이 되는가?
```

를 설명합니다.

---

# TOP 2. PyTorch 학습 루프

다음 흐름을 코드 없이도 설명할 수 있어야 합니다.

```text
DataLoader
 ↓
batch
 ↓
model(x)
 ↓
logits
 ↓
loss
 ↓
backward()
 ↓
optimizer.step()
```

그리고 다음을 정확히 구별해야 합니다.

```text
train()
eval()
no_grad()
zero_grad()
backward()
step()
```

이 개념들은 CNN, 생성모델, PEFT 등에서 계속 재등장합니다.

---

# TOP 3. Attention → Transformer → LLM 연결

2-1의 내용을 가장 확실히 복습하세요.

특히 다음 질문에 답할 수 있어야 합니다.

### Q1. Q / K / V의 역할은?

```text
Q: 무엇을 찾고 싶은가?
K: 내가 어떤 정보를 가지고 있는가?
V: 실제로 전달할 정보는 무엇인가?
```

### Q2. 왜 `QKᵀ`을 계산하는가?

각 Query와 Key의 관계 정도를 점수화하기 위해서입니다.

### Q3. 왜 `√d`로 나누는가?

dot-product 값의 크기가 커지는 것을 완화하여 softmax가 지나치게 뾰족해지는 문제를 줄이는 스케일링입니다.

### Q4. Causal Mask가 필요한 이유는?

다음 토큰을 예측할 때 미래 토큰을 미리 볼 수 없도록 하기 위해서입니다.

### Q5. Residual Connection의 역할은?

깊은 네트워크에서 정보와 gradient가 더 안정적으로 흐를 수 있도록 도와줍니다.

### Q6. Attention과 MLP의 역할 차이는?

```text
Attention = 토큰 간 정보교환
MLP       = 각 토큰의 표현 변환/강화
```

이 구분을 확실히 잡으세요.

---

# 4. 지금 미리 복습해 두면 과부하가 줄어드는 개념

## 반드시 복습

- `numpy array` vs `torch.Tensor`
- Tensor shape
- `device`
- `nn.Module`
- `forward`
- `nn.Linear`
- activation function
- `loss`
- `backward`
- optimizer
- `zero_grad`
- `step`
- `train()` / `eval()`
- `torch.no_grad()`
- Dataset / DataLoader
- train / validation / test
- scaling
- data leakage
- overfitting
- Embedding
- token ID
- vocabulary
- Q / K / V
- self-attention
- causal mask
- residual connection
- LayerNorm

---

# 5. 지금 단계에서 굳이 깊게 외울 필요 없는 것

향후 강의를 따라가는 데 아래 내용은 “코드를 보고 이해할 수 있는 정도”면 충분합니다.

- seaborn의 세부 옵션
- matplotlib의 세부 문법
- BPE의 모든 구현 세부사항
- 특정 Hugging Face API의 모든 인자
- 특정 모델의 모든 configuration 값
- Transformer 전체 소스코드 암기

중요한 것은 **구조와 데이터 흐름**입니다.

---

# 6. 효율적인 공부 방법 및 Jupyter 실습 팁

## 실습 1. 코드를 복붙하지 말고 “shape 추적 노트북” 만들기

새 노트북을 하나 만들고 코드 사이에 아래를 계속 기록하세요.

```python
print("input:", x.shape)
print("q:", q.shape)
print("k:", k.shape)
print("attention:", attn_probs.shape)
print("output:", output.shape)
```

목표:

> 코드 실행 결과를 보는 것이 아니라 shape 변화를 예측한 뒤 확인하기

---

## 실습 2. 학습 루프를 빈 코드에서 재작성

현재 Perceptron 코드를 보지 않고 아래 틀부터 직접 작성해보세요.

```python
for x_batch, y_batch in train_loader:
    optimizer.zero_grad()

    logits = model(x_batch)
    loss = criterion(logits, y_batch)

    loss.backward()
    optimizer.step()
```

그리고 각 줄의 역할을 자신의 말로 주석 처리합니다.

---

## 실습 3. Transformer를 “작게” 다시 구현

GPT를 처음부터 다시 만들 필요는 없습니다.

아주 작은 숫자로:

```text
B = 1
T = 4
H = 8
heads = 2
```

정도로 설정하여

```text
Q
K
V
→ attention score
→ mask
→ softmax
→ weighted sum
```

만 직접 계산해보세요.

이 실습의 목표는 성능이 아니라 **행렬의 shape와 계산 의미 이해**입니다.

---

## 실습 4. Attention 시각화

2-1에서 `return_attn=True` 구조가 이미 있기 때문에 이를 활용하면 좋습니다.

각 head의 attention matrix를 시각화하고:

```text
행 = Query 위치
열 = Key 위치
```

로 해석해 보세요.

특히 causal mask 때문에:

```text
미래 영역
= attention 불가능
```

이 되는 것을 직접 확인해보면 이해도가 크게 올라갑니다.

---

## 실습 5. “기존 코드 → 한 단계 변형” 연습

새 기술을 배울 때 완전히 새로운 프로젝트를 만드는 것보다 현재 과제를 조금씩 변형하는 방법을 추천합니다.

예:

```text
Perceptron
→ hidden layer 변경
→ activation 변경
→ optimizer 변경
→ dropout 추가
```

또는

```text
GPT
→ head 수 변경
→ sequence 길이 변경
→ hidden size 변경
→ causal mask 확인
```

이렇게 한 번에 하나만 바꾸세요.

---

## 실습 6. 합성 데이터 과제는 “평가 기준”을 바꿔보기

2-2에서 단순히 데이터를 생성하는 데서 끝내지 말고:

```text
생성
→ 평가
→ 필터링
→ 다시 생성
```

의 반복 구조를 생각하세요.

예:

```text
score >= 8
→ 통과

score < 8
→ 폐기 또는 재생성
```

이 사고방식은 앞으로 RAG/Agent 품질관리와도 연결됩니다.

---

# 7. 강의별 추천 공부 포인트

| 강의 | 가장 중요한 선행 개념 | 수업 중 집중할 것 |
|---|---|---|
| 3-1 CNN Transfer Learning | Tensor, DataLoader, train/eval, loss/optimizer | pretrained model의 parameter와 classifier 교체 |
| 3-2 이미지 생성/평가 | 생성/평가, prompt, 데이터 증강 | 생성 결과 평가 기준 |
| 4-1 RAG | Embedding, 전처리, LLM | chunk → embedding → retrieval → context → answer |
| 4-2-1 ReAct | Prompt, structured output | Thought/Action/Observation 흐름 |
| 4-2-2 Multi-Agent | Agent 입출력 설계 | Agent 간 역할 분리와 orchestration |
| 5-1 PEFT | parameter, gradient, optimizer | 무엇을 freeze하고 무엇을 학습하는가 |
| 5-2 Quantization | Tensor/parameter | 정밀도 감소가 메모리/성능에 미치는 영향 |

---

# 8. 추천 학습 순서

강의 순서는 그대로 따라가되, 개인 복습에서는 다음 순서를 추천합니다.

```text
[복습 A]
PyTorch 학습 루프
        ↓
[복습 B]
Embedding
        ↓
[복습 C]
Q/K/V + Attention
        ↓
[복습 D]
Transformer Decoder Block
        ↓
3-1 CNN
        ↓
3-2 생성/평가
        ↓
4-1 RAG
        ↓
4-2 Agent
        ↓
5-1 PEFT
        ↓
5-2 Quantization
```

---

# 9. 매 강의마다 사용할 수 있는 5단계 공부법

각 강의가 끝날 때 아래 5단계만 반복하세요.

### 1단계 — 개념 설명

노트를 보지 않고 직접 설명합니다.

> “이 기술은 무엇을 해결하기 위해 존재하는가?”

### 2단계 — 데이터 흐름

```text
Input
→ Processing
→ Model
→ Output
```

을 직접 그립니다.

### 3단계 — 핵심 코드 재작성

전체 코드를 외우지 말고 핵심 10~20줄을 직접 작성합니다.

### 4단계 — 하나만 변경

예:

```text
batch size 변경
head 수 변경
embedding dimension 변경
threshold 변경
```

### 5단계 — 결과 해석

반드시:

> “왜 결과가 이렇게 나왔는가?”

를 한두 문장으로 적습니다.

---

# 10. 현재 시점에서의 최종 학습 전략

## 가장 먼저 할 것

1. **PyTorch 학습 루프 완전 복습**
   - `zero_grad → forward → loss → backward → step`

2. **Tensor shape 완전 복습**
   - 특히 `(B, T, H)`와 `(B, heads, T, T)`

3. **Attention 완전 복습**
   - Q/K/V
   - `QKᵀ`
   - scaling
   - causal mask
   - softmax
   - weighted sum

## 그다음

- CNN에서는 새로운 모델 구조보다 기존 학습 파이프라인과의 공통점을 먼저 찾기
- 생성모델에서는 “생성 자체”보다 **평가와 품질관리**에 집중
- RAG에서는 `Embedding → Retrieval → Context → LLM` 흐름을 확실히 잡기
- Agent에서는 `LLM + Tool + 반복적인 의사결정` 구조를 이해하기
- PEFT에서는 “어떤 파라미터가 실제로 학습되는가?”를 중심으로 보기
- Quantization에서는 “모델 정확도와 효율의 trade-off”를 중심으로 보기

---

# 11. 최종 체크리스트

## 기초 ML / PyTorch

- [ ] train / validation / test의 차이를 설명할 수 있다.
- [ ] data leakage가 왜 발생하는지 설명할 수 있다.
- [ ] Tensor와 numpy의 차이를 설명할 수 있다.
- [ ] `forward → loss → backward → optimizer.step()`을 설명할 수 있다.
- [ ] `train()`과 `eval()`의 차이를 설명할 수 있다.
- [ ] DataLoader가 왜 필요한지 설명할 수 있다.

## NLP / Transformer

- [ ] Tokenization이 필요한 이유를 설명할 수 있다.
- [ ] BPE의 기본 동작을 설명할 수 있다.
- [ ] Embedding이 무엇인지 설명할 수 있다.
- [ ] Q / K / V 역할을 설명할 수 있다.
- [ ] Causal mask가 필요한 이유를 설명할 수 있다.
- [ ] Attention score의 shape을 계산할 수 있다.
- [ ] Decoder Block 구조를 그릴 수 있다.
- [ ] GPT의 입력부터 logits까지 흐름을 설명할 수 있다.

## 생성 / Agent

- [ ] Zero-shot / Few-shot 차이를 설명할 수 있다.
- [ ] 합성 데이터 생성 파이프라인을 설명할 수 있다.
- [ ] LLM as a Judge의 목적을 설명할 수 있다.
- [ ] RAG 전체 흐름을 그릴 수 있다.
- [ ] ReAct의 Action / Observation 개념을 설명할 수 있다.
- [ ] Multi-Agent를 사용하는 이유를 설명할 수 있다.

## 효율적 학습

- [ ] 코드의 입력/출력 shape을 먼저 예측할 수 있다.
- [ ] 코드를 보지 않고 핵심 학습 루프를 다시 작성할 수 있다.
- [ ] 작은 숫자의 toy example로 Attention을 직접 계산해볼 수 있다.
- [ ] 모델 결과가 좋은지뿐 아니라 왜 좋은지/나쁜지 설명할 수 있다.

---

# 결론

현재까지의 학습은 단순히

> “데이터 분석 → 머신러닝 → NLP”

로 따로 배운 것이 아닙니다.

실제로는 다음 역량을 단계적으로 만들고 있습니다.

```text
데이터를 이해한다
        ↓
데이터를 모델이 먹을 수 있는 형태로 만든다
        ↓
모델을 학습시킨다
        ↓
모델 내부 동작을 이해한다
        ↓
LLM을 활용한다
        ↓
LLM을 외부 데이터와 연결한다
        ↓
LLM이 도구를 사용하게 만든다
        ↓
여러 Agent로 시스템을 확장한다
        ↓
큰 모델을 효율적으로 튜닝하고 배포한다
```

따라서 지금 가장 중요한 것은 새로운 개념을 계속 추가하는 것보다,

**① Tensor/데이터 흐름 → ② PyTorch 학습 루프 → ③ Attention/Transformer**

이 세 축을 확실히 고정하는 것입니다.

이 세 가지가 잡혀 있으면 이후 CNN, RAG, Agent, PEFT, Quantization이 서로 완전히 다른 기술처럼 보이지 않고 **하나의 AI 시스템이 확장되는 과정**으로 보이기 시작할 것입니다.
