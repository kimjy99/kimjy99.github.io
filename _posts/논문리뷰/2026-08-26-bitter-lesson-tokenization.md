---
title: "[논문리뷰] You Can Learn Tokenization End-to-End with Reinforcement Learning"
last_modified_at: 2026-08-26
categories:
  - 논문리뷰
tags:
  - NLP
  - Reinforcement Learning
  - Tokenization
  - ICML
excerpt: "You Can Learn Tokenization End-to-End with Reinforcement Learning 논문 리뷰 (ICLR 2026 Spotlight)"
use_math: true
classes: wide
---

> ICLR 2026 Spotlight. [[Paper](https://arxiv.org/abs/2602.13940)] [[Github](https://github.com/SamD770/bitter-lesson-tokenization)]  
> Sam Dauncey, Roger Wattenhofer  
> ETH Zurich  
> 15 Feb 2026  

## Introduction
본 논문은 end-to-end 학습을 통해 tokenization 자체를 학습함으로써 이러한 휴리스틱한 토큰 경계 규칙을 개선하는 것을 목표로 하였다. 기존 접근 방식은 straight-through estimator에 초점을 맞추었다. 반면, 본 논문에서는 토큰 경계에 대한 loss의 gradient를 직접 근사하는 score function estimator에 초점을 맞추었다. Score function estimator는 더 높은 분산을 감수하더라도 더 강력한 이론적 보장을 제공한다.

본 논문에서는 RL의 분산 감소 기법을 적용한 score function estimator를 사용하여 명시적인 prior 구조나 inductive bias 없이도 semantic 경계와 밀접하게 일치하는 tokenization 전략을 학습할 수 있음을 보여주었다. 또한, straight-through estimator를 사용하는 기존 방식보다 성능이 우수하다.

## Method
### 1. Desiderata for Designing a End-to-End Tokenization Method
본 논문은 tokenization 프로세스를 LLM의 아키텍처 및 학습 과정 내부에 통합하는 데 관심이 있다. 이러한 방법이 실용적이고 일반화되기 위해서는 다음과 같은 요구사항을 만족해야 한다.

- **End-to-end tokenizer 학습**: 토큰 경계 결정은 loss를 최소화하도록 학습되어야 한다.
- **End-to-end 아키텍처**: Byte-level에서 학습된 표현은 토큰 수준에서 재사용되어야 한다.
- **효율성**: 기존 하드웨어에서 byte pair encoding (BPE) 기반 tokenization 방식보다 사전 학습 연산량을 0.1% 미만으로 추가해야 한다.

### 2. Autoregressive U-Net Architecture and Setup
본 논문에서는 autoregressive U-net 아키텍처를 사용한다. End-to-end 아키텍처의 요구 사항을 충족하고 byte-level 표현을 token-level에서 재사용하기 위해, 모델은 다운샘플링과 업샘플링 단계를 거쳐 byte-level에서 token-level로, 그리고 다시 byte-level로 변환한다.

<center><img src='{{"/assets/img/bitter-lesson-tokenization/bitter-lesson-tokenization-fig1.webp" | relative_url}}' width="57%"></center>
<br>
이 모델은 autoregressive하므로 각 토큰은 토큰 경계인 cutoff index까지만 byte 스트림을 요약할 수 있다. 따라서 byte 스트림을 tokenization하는 것은 이러한 점프가 발생하는 위치를 선택하는 것과 같다. 이는 각 토큰이 byte를 어떻게 모으는지와는 무관한 본질적으로 discrete한 결정이다.

$x_1 \ldots x_N$을 입력 byte 시퀀스라고 하고, $d_\textrm{enc}$, $d_\textrm{mid}$, $d_\textrm{dec}$를 모델 차원에 대한 hyperparameter라고 하자. Forward pass는 다음과 같은 구조로 되어 있다.

1. 입력 byte를 byte-level의 표현 $X$로 autoregressive하게 인코딩한다.

$$
\begin{equation}
X = \textrm{encode}(x) \in \mathbb{R}^{N \times d_\textrm{enc}}
\end{equation}
$$

{:start="2"}
2. Byte-level 표현에서 토큰 경계를 예측한다. $a_i = 1$은 $x_i$가 토큰 경계인 경우이고, $a_i = 0$은 그렇지 않은 경우이다.

$$
\begin{equation}
a \sim \pi (X, x)
\end{equation}
$$

{:start="3"}
3. Byte-level 표현을 token-level 표현 $X^\prime$으로 다운샘플링한다.

$$
\begin{equation}
X^\prime = \textrm{downsample} (X, a) \in \mathbb{R}^{M \times d_\textrm{mid}}, \quad \textrm{where} \; $M = \sum_{i=0}^N a_i
\end{equation}
$$

{:start="4"}
4. Token-level 표현을 autoregressive feedforward network로 보강한다.

$$
\begin{equation}
Y^\prime = \textrm{mid} (X^\prime) \in \mathbb{R}^{M \times d_\textrm{mid}}
\end{equation}
$$

{:start="5"}
5. 인코딩된 byte-level 정보를 포함할 수 있는 $Y$로 업샘플링한다.

$$
\begin{equation}
Y = \textrm{upsample} (Y^\prime, X, a) \in \mathbb{R}^{N \times d_\textrm{dec}}
\end{equation}
$$

{:start="6"}
6. 결과로 나온 byte-level 표현을 디코딩하여 다음 byte에 대한 예측값 $y_i = x_{i+1}$로 변환한다.

$$
\begin{equation}
y \sim \textrm{decode} (Y)
\end{equation}
$$

위의 모든 연산은 autoregressive해야 한다. 예를 들어 $\textrm{decode}$ 함수는 위치 $j$에서의 다운샘플링된 표현인 $$X_j^\prime$$가 이전 byte $X \le i$에만 의존해야 한다. 여기서 $i$는 $j = \sum_{k=0}^i a_k$를 만족하는 최소 토큰 인덱스이다.

Score function 추정치는 위 함수들의 모든 구현에 유연하게 적용할 수 있다. 구체적으로, 저자들은 $\textrm{mid}$는 full attention을 사용하는 decoder-only transformer이고, $\textrm{encode}$와 $\textrm{decode}$는 sliding window attention과 linear embedding/unembedding 행렬을 사용하는 decoder-only transformer이다. $\textrm{downsample}$ 구현에서는 단순히 $a_i = 1$ 값에 해당하는 $X^\prime$ 값을 선택한다.

$$
\begin{equation}
\textrm{downsample}(X, a)_i = X_j^\prime \quad \textrm{for } j \textrm{ the minimum value such that } j = \sum_{k=0}^i a_k
\end{equation}
$$

$\textrm{upsample}$의 경우, 간단한 distribute-then-add 연산을 적용한다.

$$
\begin{equation}
\textrm{upsample}(Y^\prime, X, a)_j = X_j + Y_i, \quad \textrm{for } i = \sum_{k=0}^j a_k
\end{equation}
$$

### 3. Score Function Estimation for Tokenization
Byte 스트림을 tokenization하는 것은 토큰 경계를 어디에 배치할지 결정하는 discrete한 선택이다. 따라서 loss는 이 선택에 대해 미분 가능하지 않으므로, end-to-end 학습이라는 목표를 달성하기 위해서는 모델이 이러한 토큰 경계를 확률적으로 설정하는 전략을 탐색해야 한다.

따라서 출력되는 모델의 next-token cross-entropy loss는 샘플링된 tokenization 전략 $$a \sim \pi_\theta$$에 조건부이므로, autoregressive 모델 $$p_\theta$$와 tokenization 전략 $$\pi_\theta$$의 파라미터를 동시에 학습하는 문제를 모든 $a$에 대해 marginalize된 next-token cross-entropy를 최소화하는 문제로 생각할 수 있다.

$$
\begin{equation}
\log p_\theta (y \vert x) = \mathbb{E}_{a \sim \pi_\theta} \log p_\theta (y \vert a, x)
\end{equation}
$$

이 likelihood의 gradient는 두 개의 개별 gradient의 합의 기대값으로 계산할 수 있다. 하나는 tokenization 전략에 따라 컨디셔닝된 표준 next-token cross-entropy loss이고, 다른 하나는 tokenization 전략 $$\pi_\theta$$에 REINFORCE를 적용한 것으로 해석할 수 있는 보정 항이다. 이 보정 항은 next-token cross-entropy를 reward로 한다.

$$
\begin{aligned}
\nabla_\theta \mathbb{E}_{a \sim \pi_\theta} \log p_\theta (y \vert a, x)
&= \nabla_\theta \sum_a \log p_\theta (y \vert a, x) \pi_\theta (a \vert x) \\
&= \sum_a (\nabla_\theta \log p_\theta (y \vert a, x)) \pi_\theta (a \vert x) + \sum_a \log p_\theta (y \vert a, x) (\nabla_\theta \pi_\theta (a \vert x)) \\
&= \sum_a (\nabla_\theta \log p_\theta (y \vert a, x)) \pi_\theta (a \vert x) + \sum_a \log p_\theta (y \vert a, x) (\pi_\theta (a \vert x) \nabla_\theta \log \pi_\theta (a \vert x)) \\
&= \sum_a (\nabla_\theta \log p_\theta (y \vert a, x)  + \log p_\theta (y \vert a, x) \nabla_\theta \log \pi_\theta (a \vert x)) \pi_\theta (a \vert x) \\
&= \mathbb{E}_{a \sim \pi_\theta} (\underbrace{\nabla_\theta \log p_\theta (y \vert a, x)}_{\textrm{cross entropy loss}} + \underbrace{\log p_\theta (y \vert a, x) \nabla_\theta \log \pi_\theta (a \vert x)}_{\textrm{policy gradient}}) \\
\end{aligned}
$$

이러한 유형의 estimator는 score function estimator다. 특히, 이는 대규모 컴퓨팅 및 데이터 제약 조건에서 위의 policy gradient 항에 대해 gradient descent를 수행하면 로컬하게 최적인 tokenization 전략을 얻을 수 있음을 의미한다.

### 4. Reducing the Variance of the Score Function Estimate
저자들은 효율성을 위해, 각 시퀀스당 하나의 샘플만 사용하여 policy gradient 항에 대한 몬테카를로 추정치를 사용하고자 하였다. 단순한 REINFORCE policy gradient는 이러한 설정에서 효율적으로 학습하기에는 너무 많은 노이즈를 포함하고 있다. 저자들은 RL의 표준 테크닉을 사용하여 이 추정치의 노이즈를 제거하였다. 이를 통해, 어떤 토큰 경계 결정이 다음 토큰의 loss를 증가시키거나 감소시키는지 연결하는 **reward attribution** 문제를 해결하였다.

모델 차원을 $$d_\textrm{model}$$로, vocabulary 크기를 고유한 UTF-8 byte 및 특수 문자의 개수 $V$로 정의한다. 또한 $i$는 토큰 인덱스, $b$는 batch 인덱스이다.

##### Early exit relative rewards
모델은 autoregressive하므로 byte에서의 토큰 경계 결정 $a_i$가 이전 byte $x_{< i}$와 이전 토큰 경계 결정 $a_{< i}$에만 의존할 수 있는 경우로 분석을 제한한다. 따라서 policy gradient 항은 토큰 경계 policy $$\pi_\theta (a_i \vert x_{\le i}, a_{< i})$$에 대해 $j > i$에 해당하는 reward $$\log p_\theta (x_j \vert x_{< j}, a_{< j})$$를 갖는 것으로 취급한다.

$$
\begin{equation}
\log p_\theta (y \vert a, x) \nabla_\theta \log \pi_\theta (a \vert x) = \sum_i \left( \sum_{j > i} \log p_\theta (x_j \vert x_{< j}, a_{< j}) \right) \nabla_\theta \log \pi_\theta (a_i \vert x_{\le i}, a_{< i})
\end{equation}
$$

$$\mathbb{E}_{a \sim \pi_\theta} \nabla_\theta \log \pi_\theta (a_i \vert x_{\le i}, a_{< i}) = 0$$이므로, 이러한 reward에 $a_i$와 무관한 임의의 항을 추가하여 유효한 policy gradient 추정치를 얻을 수 있다. 구체적으로, early byte-level embedding을 사용하여 다음 토큰 확률을 추정함으로써 tokenization에 독립적인 다음 토큰 예측의 기본 난이도를 추정한다.

$$
\begin{equation}
\log p_\theta^\textrm{early} (x_i = t_j \vert x_{< i}) = \log \textrm{softmax} (W_\textrm{early} X_{i-1})_j \\
\textrm{where} \quad W_\textrm{early} \in \mathbb{R}^{d_\textrm{model} \times V}
\end{equation}
$$

($W_\textrm{early}$는 unembedding 행렬)

이를 통해 tokenization과 무관한 노이즈의 상당 부분이 제거된 reward를 얻을 수 있다.

$$
\begin{equation}
R_i = \log p_\theta (x_i \vert x_{< i}, a_{< i}) - \log p_\theta^\textrm{early} (x_i \vert x_{< i})
\end{equation}
$$

$$W_\textrm{early}$$의 가중치를 최종 출력 head의 가중치와 동일하게 초기화하여 쉬운 전송을 가능하게 한다.

##### Time discounting
위에서 언급한 reward들을 합산하여 advantage를 계산하는 방식은 여전히 ​​분산이 너무 크고 reward attribution 문제로 어려움을 겪는다. long-horizon RL에서 이 문제를 해결하는 일반적인 방법은 advantage를 계산할 때 reward에 time discounting을 적용하는 것이다. 즉, policy gradient에 약간의 편향을 도입하여 분산을 크게 줄이는 것이다. 직관적으로, 이는 시퀀스의 멀리 떨어진 부분에 주어지는 advantage를 분리하여 시퀀스당 토큰 경계에 대해 거의 독립적인 여러 학습 자극을 제공한다.

$$
\begin{equation}
G_i = \sum_{j=0}^{N-i-1} \gamma^j R_{i+j+1}
\end{equation}
$$

($\gamma = 0.99$)

##### Batch-relative advantages
최종 모델 $$p_\theta$$가 조기 종료 모델 $$p_\theta^\textrm{early}$$보다 우수한 성능을 보이는 경향이 있기 때문에, $G_i$는 양수 값을 갖는 경향이 있다. 이러한 편향은 토큰 인덱스에 따라 달라지며, 토큰 인덱스가 나중일수록 격차가 더 커진다.

이를 해결하기 위해, batch 내의 $G_{i,1}, \ldots, G_{i,B}$를 활용하여 advantage 추정치를 중심화할 수 있다.

$$
\begin{equation}
A_{i,b} = G_{i,b} - \bar{G}_i, \quad \textrm{where} \quad \bar{G}_i = \frac{1}{B} \sum_{b=1}^B G_{i,b}
\end{equation}
$$

최종 policy loss는 다음과 같다.

$$
\begin{equation}
\mathcal{L}^\pi = -\sum_{i=0}^N \log \pi_\theta (a_i \vert x_{< i}, a_{< i}) \cdot \textrm{detach}(A_i)
\end{equation}
$$

### 5. Defining the Token Boundary Function
토큰 경계 policy $$\pi_\theta$$를 이전 byte와 토큰 경계를 조건으로 주어진 인덱스에서 토큰 경계일 확률을 나타내는 함수로 정의한다. 이 확률은 해당 logit $l_i$의 sigmoid로 계산된다.

$$
\begin{equation}
a_i \vert x_{\le i}, a_{< i} \sim \textrm{Bernoulli}(p_i) \\
p_i = \pi_\theta (a_i = 1 \vert x_{\le i}, a_{< i}) = \sigma (l_i)
\end{equation}
$$

저자들은 $l_i$의 계산 비용이 전체 forward pass에 비해 무시할 수 있을 정도로 작으면서도, 토큰 경계 휴리스틱을 모델이 학습할 수 있을 만큼 충분히 표현력이 높도록 설계했다. 명시적인 학습이 없더라도 모델은 이미 내부 표현에 시퀀스에 대한 풍부한 정보를 인코딩하고 있다. 그럼에도 불구하고, 고정된 striding 전략을 표현하기 위해서는 모델이 이전 토큰 경계의 sliding window에 접근할 수 있어야 한다.

구체적으로, linear projection $$W_j \in \mathbb{R}^{1 \times d_\textrm{model}}$$을 byte-level 표현 $X_i$에 적용하여 logit의 기본값과 window 내 각 토큰 경계에 따른 조건부 항들을 얻어 raw logit $$l_i^\textrm{raw}$$를 계산한다. 이 연산은 모든 $i, k$에 대해 단일 행렬 곱셈으로 $W_k X_i$를 미리 계산한 후 빠른 scan 연산을 수행함으로써 최신 하드웨어에서 빠른 계산이 가능하다.

$$
\begin{equation}
l_i^\textrm{raw} = W_0 X_i +  \sum_{j=1}^w a_{i-j} W_k X_i
\end{equation}
$$

(window 크기 $w = 8$)

다운샘플링 비율에 제약이 없으면 모델은 모든 byte를 토큰 경계로 분리하는 계산 비용이 많이 드는 전략을 사용하게 된다. 이를 방지하기 위해 모델을 목표 다운샘플링 비율 $$\bar{\pi}_\textrm{target}$$으로 조정한다. Attention 모델에서와 마찬가지로 초기화 시 raw logit을 대략적으로 균일하게 scaling해야 하므로, 초기화 시 $$p_i \approx \bar{\pi}_\textrm{target}$$이 되도록 $$\sigma^{-1}(\bar{\pi}_\textrm{target})$$항을 추가하여 안정성을 더욱 향상시킨다.

$$
\begin{equation}
l_i^\textrm{scaled} = \frac{l_i^\textrm{raw}}{D} + \sigma^{-1} (\bar{\pi}_\textrm{target})
\end{equation}
$$

(Scaling factor $D = 16$, $$\bar{\pi}_\textrm{target} = 0.2$$)

Logit의 폭발로 인한 수치적 문제를 방지하기 위해 $$l_i^\textrm{scaled}$$에 [softcapping](https://arxiv.org/abs/2408.00118)을 적용한다. 평가 과정에서는 이 softcapping 단계를 건너뛰고 $$l_i = l_i^\textrm{scaled}$$로 설정한다.

$$
\begin{equation}
l_i = \textrm{softcap} (l_i^\textrm{scaled})
\end{equation}
$$

### 6. Downsample Rate Targeting
본 논문에서는 batch 내 모든 logit $l_{i,b}$에 걸쳐 균일한 압력을 가함으로써 다운샘플링 비율을 목표값 $$\bar{\pi}_\textrm{target}$$에 가깝게 유지하는 메커니즘을 제시하였다. 토큰 경계 확률에 직접 압력을 가하는 방식은 sigmoid 함수의 gradient 크기가 고르지 않아 불안정한 결과를 초래할 수 있으므로, 본 논문에서는 토큰 경계 확률에 직접 압력을 가하는 방식을 제안하였다.

구체적으로, batch 평균 logit $\bar{l}$에 대해 평균 토큰 경계 확률 $\bar{p}$가 목표 다운샘플링 비율 $$\bar{\pi}_\textrm{target}$$를 초과하면 음의 압력을, 미달하면 양의 압력을 가한다. 이러한 메커니즘은 다음 loss를 통해 구현된다.

$$
\begin{equation}
\mathcal{L}^\textrm{target} = \bar{l} \cdot \textrm{detach} (\bar{p} - \bar{\pi}_\textrm{target}) \\
\textrm{where} \quad \bar{l} = \frac{1}{NB} \sum_{i,b} l_{i,b}, \quad \bar{p} = \frac{1}{NB} \sum_{i,b} p_{i,b}
\end{equation}
$$

### 7. Full Loss Formula
메인 모델과 조기 종료 모델을 학습시키기 위한 autoregressive loss는 각각 다음과 같다.

$$
\begin{aligned}
\mathcal{L}^\textrm{auto} &= -\sum_{i=0}^N \log p_\theta (x_i \vert x_{< i}, a_{< i}) \\
\mathcal{L}^\textrm{early} &= -\sum_{i=0}^N \log p_\theta^\textrm{early} (x_i \vert x_{< i})
\end{aligned}
$$

전체 loss는 다음과 같다.

$$
\begin{equation}
\mathcal{L} = \mathcal{L}^\textrm{auto} + \lambda_\pi \mathcal{L}^\pi + \lambda_\textrm{target} \mathcal{L}^\textrm{target} + \lambda_\textrm{early} \mathcal{L}^\textrm{early}
\end{equation}
$$

($$\lambda_\pi = \lambda_\textrm{target} = 0.01$$, $$\lambda_\textrm{early} = 0.1$$)

## Expertiments
- 데이터셋: FineWeb

### 1. Learned Tokenization Strategies for Natural Language
다음은 생성된 토큰 경계를 시각화한 예시이다.

<center><img src='{{"/assets/img/bitter-lesson-tokenization/bitter-lesson-tokenization-fig2.webp" | relative_url}}' width="85%"></center>

### 2. Natural Language Performance
다음은 147M 모델에 대한 validation loss curve를 비교한 결과이다.

<center><img src='{{"/assets/img/bitter-lesson-tokenization/bitter-lesson-tokenization-fig3.webp" | relative_url}}' width="55%"></center>

### 3. Comparison to BPE-Guided Downsampling
다음은 147M 모델의 성능을 다양한 downstream task에서 비교한 결과이다.

<center><img src='{{"/assets/img/bitter-lesson-tokenization/bitter-lesson-tokenization-table1.webp" | relative_url}}' width="82%"></center>

### 4. Python Code
다음은 CodeParrot 데이터셋으로 학습시킨 90M 모델에 대한 validation loss curve를 비교한 결과이다.

<center><img src='{{"/assets/img/bitter-lesson-tokenization/bitter-lesson-tokenization-fig4.webp" | relative_url}}' width="55%"></center>