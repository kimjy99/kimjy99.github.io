---
title: "[논문리뷰] Differential Transformer"
last_modified_at: 2024-10-23
categories:
  - 논문리뷰
tags:
  - Transformer
  - NLP
  - AI
  - Microsoft
excerpt: "Diff Transformer 논문 리뷰"
use_math: true
classes: wide
---

> arXiv 2024. [[Paper](https://arxiv.org/abs/2410.05258)] [[Github](https://github.com/microsoft/unilm/tree/master/Diff-Transformer)]  
> Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang, Furu Wei  
> Microsoft Research | Tsinghua University  
> 7 Oct 2024  

## Introduction
Transformer는 최근 몇 년 동안 상당한 관심을 모았으며, Transformer가 LLM의 사실상 표준으로 부상했다. Transformer의 핵심은 attention 메커니즘으로, softmax 함수를 사용하여 시퀀스에서 다양한 토큰의 중요성을 평가한다. 그러나 최근 연구에 따르면 LLM은 컨텍스트에서 핵심 정보를 정확하게 검색하는 데 어려움을 겪는다.

<center><img src='{{"/assets/img/diff-transformer/diff-transformer-fig1.PNG" | relative_url}}' width="100%"></center>
<br>
위 그림은 Transformer에 의해 컨텍스트의 각 부분에 할당한 정규화된 attention score를 시각화한 것이다. Task는 문서 더미의 중간에 포함된 답을 검색하는 것이다. Transformer는 정답에 attention score의 작은 비율만 할당하는 경향이 있고, 관련 없는 컨텍스트에 불균형하게 집중하는 경향이 있음을 알 수 있다. 이 문제는 관련 없는 컨텍스트에 할당된 무시할 수 없는 attention score에서 발생하며, 궁극적으로 정답을 압도한다. 이러한 관계 없는 score를 **attention noise**라고 부른다.

본 논문에서는 LLM을 위한 기반 아키텍처인 **Differential Transformer (Diff Transformer)**를 소개한다. Differential attention 메커니즘은 differential denoising을 통해 attention noise를 제거한다. 구체적으로, query와 key 벡터를 두 그룹으로 분할하고 두 개의 별도 softmax attention map을 계산한다. 그런 다음 이 두 map을 뺀 결과를 attention score로 간주한다. 

이 메커니즘은 attention noise를 제거하여 모델이 중요한 정보에 집중하도록 한다. 이 접근 방식은 두 신호의 차이로 노이즈를 제거하는 노이즈캔슬링 헤드폰과 유사하다. 위 그림에서 볼 수 있듯이, Diff Transformer는 Transformer에 비해 정답에 상당히 높은 score를 할당하고 관계 없는 컨텍스트에 훨씬 낮은 score를 할당한다. 이를 통해 검색 능력에서 눈에 띄는 개선을 달성했다. 

Transformer와 비교하여 Diff Transformer는 다음과 같은 이점이 있다. 

1. Transformer와 비슷한 언어 모델링 성능을 달성하기 위해 Transformer에 필요한 모델 크기 또는 학습 토큰의 약 65%만 필요하다.
2. 다양한 다운스트림 task에서 Transformer보다 성능이 뛰어나다. 
3. 긴 컨텍스트를 활용하는 데 매우 효과적이다. 
4. 핵심 정보 검색, hallucination 완화, in-context learning에서 Transformer보다 상당히 성능이 뛰어나다.
5. 모델 activation의 outlier를 줄여 quantization에 대한 새로운 기회를 제공한다. 

## Method
본 논문은 LLM과 같은 시퀀스 모델링을 위한 기반 아키텍처로 Diff Transformer를 제안하였다. 모델은 $L$개의 Diff Transformer 레이어가 쌓인 형태이다. 입력 시퀀스 $x = x_1 \cdots x_N$이 주어지면 입력 임베딩을 $$X^0 = [x_1, \cdots, x_N] \in \mathbb{R}^{N \times d_\textrm{model}}$$로 패킹한다. 여기서 $d_\textrm{model}$은 모델의 hidden dimension이다. $X^0$는 Diff Transformer를 $L$번 통과하여 최종적으로 출력 $X^L$을 얻는다. 

각 레이어는 두 개의 모듈, 즉 differential attention 모듈과 feed-forward network 모듈로 구성된다. Transformer와 비교할 때 주요 차이점은 레이아웃을 동일하게 유지하면서 기존 softmax attention을 differential attention으로 대체한다는 것이다. 또한 저자들은 LLaMA를 따라 pre-RMSNorm과 SwiGLU를 채택하였다. 

### 1. Differential Attention
<center><img src='{{"/assets/img/diff-transformer/diff-transformer-fig2.PNG" | relative_url}}' width="80%"></center>
<br>
Differential attention 메커니즘은 query, key, value 벡터를 출력에 매핑한다. Query 벡터와 key 벡터를 사용하여 attention score를 계산한 다음 value 벡터의 가중 합을 계산한다. 중요한 디자인은 softmax 함수 쌍을 사용하여 attention score의 noise를 상쇄시킨다는 것이다. 

구체적으로, 입력 $$X \in \mathbb{R}^{N \times d_\textrm{model}}$$이 주어지면 먼저 이를 query $Q_1, Q_2 \in \mathbb{R}^{N \times d}$, key $K_1, K_2 \in \mathbb{R}^{N \times d}$, value $V \in \mathbb{R}^{N \times 2d}$로 projection시킨다. 그런 다음 differential attention 연산자 $\textrm{DiffAttn}(\cdot)$은 다음과 같이 출력을 계산한다. 

$$
\begin{equation}
[Q_1; Q_2] = XW^Q, \quad [K_1; K_2] = XW^K, \quad V = XW^V \\
\textrm{DiffAttn} (X) = (\textrm{softmax} (\frac{Q_1 K_1^\top}{\sqrt{d}}) - \lambda \textrm{softmax} (\frac{Q_2 K_2^\top}{\sqrt{d}})) V
\end{equation}
$$

($W^Q, W^K, W^V \in \mathbb{R}^{d_\textrm{model} \times 2d}$는 파라미터, $\lambda$는 학습 가능한 스칼라)

학습을 동기화하기 위해 스칼라 $\lambda$를 다음과 같이 re-parameterize한다.

$$
\begin{equation}
\lambda = \exp (\lambda_{\mathbf{q}_1} \cdot \lambda_{\mathbf{k}_1}) - \exp (\lambda_{\mathbf{q}_2} \cdot \lambda_{\mathbf{k}_2}) + \lambda_\textrm{init}
\end{equation}
$$

($$\lambda_{\mathbf{q}_1}, \lambda_{\mathbf{k}_1}, \lambda_{\mathbf{q}_2}, \lambda_{\mathbf{k}_2} \in \mathbb{R}^d$$는 학습 가능한 벡터, $$\lambda_\textrm{init}$$은 $\lambda$의 초기화를 위해 사용되는 상수)

저자들은 경험적으로 $\lambda_\textrm{init} = 0.8 - 0.6 \times \exp (-0.3 \cdot (l-1))$으로 설정하면 잘 작동한다는 것을 발견했다. ($l \in [1, L]$은 레이어 인덱스)

##### Multi-Head Differential Attention
Differential Transformer는 multi-head 메커니즘도 사용한다. $h$를 attention head의 수라 하자. Head에 대해 서로 다른 projection matrix $W_i^Q, W_i^K, W_i^V$를 사용한다 ($i \in [1, h]$). 스칼라 $\lambda$는 같은 레이어 내의 head 간에 공유된다. 그런 다음 head 출력은 정규화되고 다음과 같이 최종 결과로 projection된다. 

$$
\begin{aligned}
\textrm{head}_i &= \textrm{DiffAttn} (X; W_i^Q, W_i^K, W_i^V, \lambda) \\
\overline{\textrm{head}_i} &= (1 - \lambda_\textrm{init}) \cdot \textrm{LN} (\textrm{head}_i) \\
\textrm{MultiHead} (X) &= \textrm{Concat} (\overline{\textrm{head}_1}, \ldots, \overline{\textrm{head}_h}) W^O
\end{aligned}
$$

($W^O \in \mathbb{R}^{d_\textrm{model} \times d_\textrm{model}}$은 학습 가능한 projection matrix, $\textrm{LN}(\cdot)$는 각 head에 대한 RMSNorm, $\textrm{Concat}(\cdot)$은 채널 차원으로 concat)

$\textrm{LN}(\cdot)$에 고정된 값 $$(1 - \lambda_\textrm{init})$$를 곱하여 Transformer에 gradient를 맞춘다. 이를 통해 기존 Transformer와 유사한 hyperparameter를 직접 사용하여 학습 안정성을 보장할 수 있다. Head 수는 $h = d_\textrm{model} / 2d$로 설정한다. 여기서 $d$는 Transformer의 head 차원과 같다. 따라서 파라미터 수와 계산 복잡도를 Transformer와 맞출 수 있다. 

##### Headwise Normalization 
Differential attention은 보다 sparse한 패턴을 갖는 경향이 있으므로 통계적 정보는 head 간에 더 다양하다. $\textrm{LN}(\cdot)$ 연산자는 연결 전에 각 head를 정규화하여 gradient를 개선한다.

### 2. Overall Architecture
전체 아키텍처는 $L$개의 레이어를 쌓으며, 각 레이어에는 multi-head differential attention 모듈과 feed-forward network 모듈이 포함된다. 

$$
\begin{aligned}
Y^l &= \textrm{MultiHead} (\textrm{LN} (X^l)) + X^l \\
X^{l+1} &= \textrm{SwiGLU} (\textrm{LN} (Y^l)) + Y^l
\end{aligned}
$$

$\textrm{LN}(\cdot)$은 RMSNorm이고, $\textrm{SwiGLU}(\cdot)$는 다음과 같이 정의된다. 

$$
\begin{equation}
\textrm{SwiGLU}(X) = (\textrm{swish}(XW^G) \odot XW_1) W_2 \\
\textrm{where} \quad W^G, W_1 \in \mathbb{R}^{d_\textrm{model} \times \frac{8}{3} d_\textrm{model}}, \; W_2 \in \mathbb{R}^{\frac{8}{3} d_\textrm{model} \times d_\textrm{model}}
\end{equation}
$$

($W^G$, $W_1$, $W_2$는 학습 가능한 행렬)

## Experiments
### 1. Language Modeling Evaluation
다음은 Transformer를 사용한 언어 모델들과 Eval Harness 정확도를 비교한 표이다. 

<center><img src='{{"/assets/img/diff-transformer/diff-transformer-table1.PNG" | relative_url}}' width="88%"></center>

### 2. Scalability Compared with Transformer
다음은 파라미터 수와 학습 토큰 수에 따른 언어 모델링 loss를 비교한 그래프이다. 

<center><img src='{{"/assets/img/diff-transformer/diff-transformer-fig3.PNG" | relative_url}}' width="84%"></center>

### 3. Long-Context Evaluation
다음은 시퀀스 위치에 따른 negative log-likelihood (NLL) 누적 평균을 비교한 그래프이다. (낮을수록 좋음)

<center><img src='{{"/assets/img/diff-transformer/diff-transformer-fig4.PNG" | relative_url}}' width="40%"></center>

### 4. Key Information Retrieval
- **Multi-needle retrieval**: 컨텍스트에 $N$개의 숫자-도시 쌍을 넣어두고 $R$개의 도시를 검색하는 task

다음은 4K 길이의 컨텍스트에 대한 multi-needle retrieval의 평균 정확도를 비교한 표이다. (3B 모델로 평가)

<center><img src='{{"/assets/img/diff-transformer/diff-transformer-table2.PNG" | relative_url}}' width="35%"></center>
<br>
다음은 8K에서 64K 길이의 컨텍스트에 대한 multi-needle retrieval의 평균 정확도를 비교한 표이다. (3B 모델로 평가)

<center><img src='{{"/assets/img/diff-transformer/diff-transformer-fig5.PNG" | relative_url}}' width="90%"></center>
<br>
다음은 정보 검색 task에서 답변과 noise 컨텍스트에 할당된 attention score를 비교한 표이다. 

<center><img src='{{"/assets/img/diff-transformer/diff-transformer-table3.PNG" | relative_url}}' width="68%"></center>

### 5. In-Context Learning
다음은 many-shot in-context learning 정확도를 비교한 표이다. 

<center><img src='{{"/assets/img/diff-transformer/diff-transformer-fig6.PNG" | relative_url}}' width="74%"></center>
<br>
다음은 TREC 데이터셋에서 in-context learning의 robustness를 비교한 그래프이다. 

<center><img src='{{"/assets/img/diff-transformer/diff-transformer-fig7.PNG" | relative_url}}' width="78%"></center>

### 6. Contextual Hallucination Evaluation
다음은 (a) 텍스트 요약 데이터셋들과 (b) QA 데이터셋들에서 hallucination을 비교한 표이다. 

<div style="display: flex; align-items: start; justify-content: center">
  <img src='{{"/assets/img/diff-transformer/diff-transformer-table4a.PNG" | relative_url}}' width="34%">
  &nbsp;&nbsp;&nbsp;
  <img src='{{"/assets/img/diff-transformer/diff-transformer-table4b.PNG" | relative_url}}' width="36%">
</div>

### 7. Activation Outliers Analysis
다음은 attention logit과 hidden state의 activation 최대값을 비교한 표이다. 

<center><img src='{{"/assets/img/diff-transformer/diff-transformer-table5.PNG" | relative_url}}' width="70%"></center>
<br>
다음은 HellaSwag 데이터셋에서 zero-shot 정확도를 비교한 그래프이다. 

<center><img src='{{"/assets/img/diff-transformer/diff-transformer-fig8.PNG" | relative_url}}' width="37%"></center>

### 8. Ablation Studies
다음은 1.4B 모델에 대한 ablation study 결과이다. 

<center><img src='{{"/assets/img/diff-transformer/diff-transformer-table6.PNG" | relative_url}}' width="55%"></center>