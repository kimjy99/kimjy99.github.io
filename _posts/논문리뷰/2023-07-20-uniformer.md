---
title: "[논문리뷰] UniFormer: Unified Transformer for Efficient Spatiotemporal Representation Learning"
last_modified_at: 2023-07-20
categories:
  - 논문리뷰
tags:
  - ViT
  - Video Classification
  - Computer Vision
  - AI
  - ICLR
excerpt: "UniFormer 논문 리뷰 (ICLR 2022)"
use_math: true
classes: wide
---

> ICLR 2022. [[Paper](https://arxiv.org/abs/2201.04676)] [[Github](https://github.com/Sense-X/UniFormer)]  
> Kunchang Li, Yali Wang, Peng Gao, Guanglu Song, Yu Liu, Hongsheng Li, Yu Qiao  
> Chinese Academy of Sciences | University of Chinese Academy of Sciences | Shanghai AI Laboratory | SenseTime Research | The Chinese University of Hong Kong  
> 12 Jan 2022  

## Introduction
시공간 표현 학습은 동영상 이해를 위한 기본 task이다. 기본적으로 두 가지 뚜렷한 과제가 있다. 

1. 동영상에는 큰 시공간 중복성이 포함되어 있어 로컬한 인접 프레임에 걸친 모션이 미묘하게 다르다.  
2. 동영상은 장거리 프레임 간의 관계가 동적이기 때문에 복잡한 시공간 의존성을 포함한다.

동영상 분류의 발전은 대부분 3D CNN과 시공간 trasnformer에 의해 주도되었다. 불행하게도 이 두 가지 프레임워크는 각각 앞서 언급한 과제 중 하나에 초점을 맞추고 있다. 3D convolution은 작은 3D neighborhood (ex. 3$\times$3$\times$3)의 컨텍스트로 각 픽셀을 처리하여 상세하고 로컬한 시공간적 feature를 캡처할 수 있다. 따라서 인접한 프레임에서 시공간 중복성을 줄일 수 있다. 그러나 제한된 receptive field로 인해 3D convolution은 장거리 의존성을 학습하는 데 어려움을 겪는다. 

또는 vision transformer는 시각적 토큰 간의 self-attention을 통해 글로벌 의존성을 포착하는 데 능숙하다. 최근에 이 디자인은 시공간 attention 메커니즘을 통해 동영상 분류에 도입되었다. 그러나 video transformer는 얕은 레이어의 로컬한 시공간 feature를 인코딩하는 데 종종 비효율적이다. 다음 그림은 잘 알려진 일반적인 TimeSformer를 시각화한 것이다. 

<center><img src='{{"/assets/img/uniformer/uniformer-fig1.PNG" | relative_url}}' width="90%"></center>
<br>
위 그림에서 볼 수 있듯이 TimeSformer는 실제로 초기 레이어에서 상세한 동영상 표현을 학습하지만 매우 중복된 공간적 및 시간적 attention을 가진다. 특히, 공간적 attention는 주로 이웃한 토큰에 초점을 맞추는 반면, 동일한 프레임의 나머지 토큰에서는 아무것도 학습하지 않는다. 유사하게, 시간적 attention은 대부분 인접한 프레임의 토큰만 집계하고 먼 프레임의 토큰은 무시한다. 더 중요한 것은 이러한 로컬 표현은 모든 레이어에서 글로벌 토큰 간 유사성 비교를 통해 학습되므로 많은 계산 비용이 필요하다. 이 사실은 이러한 video transformer의 계산량-정확도 균형을 분명히 악화시킨다 (아래 그림 참조).

<center><img src='{{"/assets/img/uniformer/uniformer-fig2.PNG" | relative_url}}' width="85%"></center>
<br>
본 논문은 이러한 어려움을 해결하기 위해 간결한 transformer 형식으로 3D convolution과 시공간적 self-attention을 효과적으로 통합할 것을 제안하였다. 따라서 효율성과 효과 사이에서 바람직한 균형을 달성할 수 있는 네트워크를 **Unified transFormer (UniFormer)**라고 부른다. 보다 구체적으로 UniFormer는 Dynamic Position Embedding (DPE), Multi-Head Relation Aggregator (MHRA), Feed-Forward Network (FFN)의 세 가지 핵심 모듈로 구성된다. UniFormer와 기존 video transFormer의 주요 차이점은 relation aggregator의 독특한 디자인이다. 

첫째, 모든 레이어에서 self-attention 메커니즘을 사용하는 대신 제안된 relation aggregator는 각각 동영상 중복성과 의존성을 처리한다. 얕은 레이어에서 aggregator는 학습 가능한 작은 파라미터 행렬로 로컬 관계를 학습한다. 이는 작은 3D 환경에서 인접한 토큰의 컨텍스트를 집계하여 계산 부담을 크게 줄일 수 있다. 깊은 레이어에서 aggregator는 유사도 비교를 통해 글로벌한 관계를 학습하여 동영상의 먼 프레임에서 장거리 토큰 의존성을 유연하게 구축할 수 있다. 둘째, 전통적인 transformer의 공간적 및 시간적 attention 분리와 달리 relation aggregator는 모든 레이어에서 시공간 컨텍스트를 공동으로 인코딩하므로 공동 학습 방식으로 동영상 표현을 더욱 향상시킬 수 있다. 마지막으로 UniFormer 블록을 계층적 방식으로 점진적으로 통합하여 모델을 구축한다. 이 경우 동영상에서 효율적인 시공간 표현 학습을 위해 로컬 UniFormer 블록과 글로벌 UniFormer 블록의 협력을 확대한다.

## Method
### 1. Overview of UniFormer Block
<center><img src='{{"/assets/img/uniformer/uniformer-fig3.PNG" | relative_url}}' width="100%"></center>
<br>
본 논문은 시공간 중복성 및 의존성 문제를 극복하기 위해 위 그림과 같은 새롭고 간결한 Unified transFormer (UniFormer)를 제안하였다. 기본 transformer 형식을 사용하지만 효율적이고 효과적인 시공간 표현 학습을 위해 특별히 디자인되었다. 특히 UniFormer 블록은

1. Dynamic Position Embedding (DPE)
2. Multi-Head Relation Aggregator (MHRA)
3. Feed-Forward Network (FFN)

의 세 가지 주요 모듈로 구성된다.

$$
\begin{aligned}
X &= \textrm{DPE} (X_\textrm{in}) + X_\textrm{in} \\
Y &= \textrm{MHRA} (\textrm{Norm} (X)) + X \\
Z &= \textrm{FFN} (\textrm{Norm} (Y)) + Y
\end{aligned}
$$

입력 토큰 텐서 (프레임 볼륨) $X_\textrm{in} \in \mathbb{R}^{C \times T \times H \times W}$를 고려하여 먼저 DPE를 도입하여 3D 위치 정보를 모든 토큰에 동적으로 통합하여 동영상 모델링을 위해 토큰의 시공간적 순서를 효과적으로 사용한다. 그런 다음 MHRA를 활용하여 각 토큰을 컨텍스트 토큰과 함께 집계한다. 일반적인 Multi-Head Self-Attention (MHSA)와 달리 MHRA는 얕은 레이어와 깊은 레이어에서 토큰 유사도 학습의 유연한 디자인을 통해 동영상의 로컬 중복성과 글로벌 의존성을 현명하게 처리한다. 마지막으로 각 토큰의 점별 향상을 위해 두 개의 linear layer가 있는 FFN을 추가한다.

### 2. Multi-Head Relation Aggregator
효율적이고 효과적인 시공간 표현 학습을 위해 큰 로컬 중복성과 복잡한 글로벌 의존성을 해결해야 한다. 인기 있는 3D CNN과 시공간 transformer는 이 두 가지 문제 중 하나에만 초점을 맞춘다. 이러한 이유로 3D convolution과 시공간적 self-attention을 간결한 transformer 형식으로 유연하게 통합할 수 있는 대안적인 Relation Aggregator (RA)를 설계하여 얕은 레이어와 깊은 레이어에서 중복성과 의존성을 각각 해결한다. 구체적으로, MHRA는 multi-head fusion을 통해 토큰 관계를 학습한다.

$$
\begin{aligned}
R_n (X) &= A_n V_n (X) \\
\textrm{MHRA} (X) &= \textrm{Concat} (R_1 (X); R_2 (X); \cdots; R_N (X)) U
\end{aligned}
$$

입력 텐서 $X \in \mathbb{R}^{C \times T \times H \times W}$가 주어지면 먼저 이를 토큰 시퀀스

$$
\begin{equation}
X \in \mathbb{R}^{L \times C}, \quad L = T \times H \times W
\end{equation}
$$

로 재구성한다. $R_n (\cdot)$은 $n$번째 head의 relation aggregator (RA)이고, $U \in \mathbb{R}^{C \times C}$는 $N$개의 head를 통합하기 위한 학습 가능한 파라미터 행렬이다. 또한 각 RA는 토큰 컨텍스트 인코딩과 토큰 선호도 학습으로 구성된다. 선형 변환을 통해 원래 토큰을 컨텍스트 $V_n (X) \in \mathbb{R}^{L \times \frac{C}{N}}$으로 변환할 수 있다. 결과적으로 relation aggregator는 토큰 선호도 $A_n \in \mathbb{R}^{L \times L}$의 guidance로 컨텍스트를 요약할 수 있다. RA의 핵심은 동영상에서 $A_n$을 배우는 방법이다.

#### Local MHRA
얕은 레이어에서는 작은 3D 이웃의 로컬 시공간 컨텍스트에서 자세한 동영상 표현을 학습하는 것을 목표로 한다. 이는 우연히 3D convolution 필터 디자인과 유사한 통찰력을 공유한다. 결과적으로 로컬 3D 이웃에서 작동하는 학습 가능한 파라미터 행렬로 토큰 친화도를 디자인한다. 즉, 하나의 앵커 토큰 $X_i$가 주어지면 RA는 이 토큰과 작은 튜브 $\Omega_i^{t \times h \times w}$의 다른 토큰 간의 로컬 시공간 선호도를 학습한다.

$$
\begin{equation}
A_n^\textrm{local} (X_i, X_j) = a_n^{i-j}, \quad \textrm{where} \quad j \in \Omega_i^{t \times h \times w}
\end{equation}
$$

여기서 $a_n \in \mathbb{R}^{t \times h \times w}$와 $X_j$는 $\Omega_i^{t \times h \times w}$의 이웃 토큰을 나타낸다. $(i − j)$는 합산 가중치를 결정하는 상대적인 토큰 인덱스를 의미한다. 얕은 레이어에서 인접한 토큰 사이의 동영상 콘텐츠는 미묘하게 다르므로 중복성을 줄이기 위해 로컬 연산자와 함께 세부 feature을 인코딩하는 것이 중요하다. 따라서 토큰 선호도는 토큰 간의 상대적인 3D 위치에만 값이 의존하는 학습 가능한 로컬 파라미터 행렬로 설계된다.

#### Comparison to 3D Convolution Block
저자들은 흥미롭게도 로컬 MHRA가 MobileNet 블록의 시공간적 확장으로 해석될 수 있음을 발견했다. 구체적으로, 선형 변환 $V(\cdot)$은 pointwise convolution (PWConv)으로 인스턴스화될 수 있다. 또한 로컬 토큰 선호도 $A_n^\textrm{local}$은 각 출력 채널 (또는 head) $V_n (X)$에서 작동하는 시공간 행렬이므로 relation aggregator $R_n (X) = A_n^\textrm{local} V_n (X)$는 depthwise convolution (DWConv)으로 설명할 수 있다. 마지막으로 모든 head는 선형 행렬 $U$에 의해 연결 및 융합되며 pointwise convolution (PWConv)으로도 인스턴스화할 수 있다. 

결과적으로 이 로컬 MHRA는 MobileNet 블록에서 PWConv-DWConv-PWConv 방식으로 재구성될 수 있다. 저자들은 실험에서 UniFormer가 경량 동영상 분류를 위한 계산 효율성을 물려받을 수 있도록 채널로 분리된 시공간 convolution과 같은 로컬 MHRA를 유연하게 인스턴스화한다. MobileNet 블록과 달리 UniFormer 블록은 일반 transformer 형식으로 설계되었으므로 MHRA 뒤에 추가 FFN이 삽입되어 각 시공간 위치에서 토큰 컨텍스트를 추가로 혼합하여 분류 정확도를 높일 수 있다.

#### Global MHRA
깊은 레이어에서는 글로벌한 동영상 클립에서 장기적인 토큰 의존성을 캡처하는 데 중점을 둔다. 이것은 자연스럽게 self-attention의 디자인과 유사한 통찰력을 공유한다. 따라서 글로벌한 관점에서 모든 토큰 간의 콘텐츠 유사성을 비교하여 토큰 선호도를 설계한다.

$$
\begin{equation}
A_n^\textrm{global} (X_i, X_j) = \frac{\exp (Q_n (X_i)^\top K_n (X_j))}{\sum_{j' \in \Omega_{T \times H \times W}} \exp (Q_n (X_i)^\top K_n (X_{j'}))}
\end{equation}
$$

여기서 $X_j$는 크기가 $T \times H \times W$인 글로벌 3D 튜브의 모든 토큰이 될 수 있으며 $Q_n (\cdot)$과 $K_n (\cdot)$은 두 가지 다른 선형 변환이다. 대부분의 video transformer는 모든 단계에서 self-attention을 적용하여 많은 양의 계산을 도입한다. 내적 계산을 줄이기 위해 이전 연구들은 공간적 attention과 시간적 attention을 나누는 경향이 있지만 토큰 간의 시공간적 관계를 악화시킨다. 대조적으로, MHRA는 초기 레이어에서 로컬 relation aggregator를 수행하여 토큰 비교 계산을 크게 절약한다. 따라서 시공간 attention을 분해하는 대신 바람직한 계산량-정확도 균형을 달성하기 위해 모든 단계에 대해 MHRA에서 시공간 관계를 공동으로 인코딩한다.

#### Comparison to Transformer Block
깊은 레이어에서 UniFormer 블록에는 글로벌 MHRA $A_n^\textrm{global}$이 장착되어 있다. 그것은 $Q_n (\cdot)$, $K_n (\cdot)$, $V_n (\cdot)$이 transformer에서 query, key, value가 되는 시공간적 self-attention으로 인스턴스화할 수 있다. 따라서 장기 의존성을 효과적으로 학습할 수 있다. 이전 video transformer의 공간적 및 시간적 분해 대신, 글로벌 MHRA는 보다 판별적인 동영상 표현을 생성하기 위해 시공간 공동 학습을 기반으로 한다. 또한 순열 불변성을 극복하기 위해 동적 위치 임베딩 (DPE)을 채택하여 변환 불변성을 유지할 수 있고 다양한 입력 클립 길이에 친숙하다.

### 3. Dynamic Position Embedding
동영상은 공간적, 시간적 변이체이기 때문에 토큰 표현을 위한 시공간적 위치 정보를 인코딩하는 것이 필요하다. 이전 방법은 주로 이 문제를 해결하기 위해 절대 또는 상대 위치 임베딩을 조정한다. 그러나 더 긴 입력 클립으로 테스트할 때는 절대 위치 임베딩을 fine-tuning하여 대상 입력 크기로 보간해야 한다. 게다가 상대 위치 임베딩은 self-attention을 수정하고 절대 위치 정보가 부족하여 성능이 더 나쁘다. 위의 문제를 극복하기 위해 본 논문은 조건부 위치 인코딩 (conditional position encoding, CPE)을 확장하여 DPE를 설계하였다.

$$
\begin{equation}
\textrm{DPE} (X_\textrm{in}) = \textrm{DWConv} (X_\textrm{in})
\end{equation}
$$

여기서 DWConv는 zero padding을 포함한 간단한 3D depthwise convolution을 의미한다. 공유 파라미터와 convolution의 locality 덕분에 DPE는 순열 불변성을 극복할 수 있고 임의의 입력 길이에 적합하다. 또한 CPE에서 zero padding은 경계에 있는 토큰이 절대 위치를 인식하도록 도와주므로 모든 토큰이 이웃 쿼리를 통해 절대 시공간 위치 정보를 점진적으로 인코딩할 수 있다. 

### 4. Model Architecture
UniFormer 블록을 계층적으로 쌓아 시공간 학습을 위한 네트워크를 구축한다. 네트워크는 4단계로 구성되어 있으며 채널 수는 각각 64, 128, 320, 512이다. UniFormer 블록 수에 따라 두 가지 모델 변형이 있다.

1. **UniFormer-S**: {3, 4, 8, 3}
2. **UniFormer-B**: {5, 8, 20, 7}

처음 두 단계에서는 로컬 토큰 선호도와 함께 MHRA를 활용하여 단기 시공간 중복성을 줄인다. 튜브 크기는 5$\times$5$\times$5로 설정되고 head 수 $N$은 해당 채널 수와 동일하다. 마지막 두 단계에서는 장기 의존성을 포착하기 위해 head 차원이 64인 글로벌 토큰 선호도와 함께 MHRA를 적용한다. 로컬 MHRA에는 BN이 사용되며 글로벌 MHRA에는 LN이 사용되는다. DPE의 커널 크기는 3$\times$3$\times$3이고 모든 레이어에서 FFN의 확장 비율은 4이다. 

첫 번째 단계 전에 stride가 2$\times$4$\times$4인 3$\times$4$\times$4 컨벌루션을 적용한다. 이는 공간 및 시간 차원이 모두 다운샘플링됨을 의미한다. 다른 단계 전에 stride가 1$\times$2$\times$2인 1$\times$2$\times$2 convolution을 적용한다. 마지막으로 시공간 average pooling과 fully connected layer을 사용하여 최종 예측을 출력한다.

#### Comparison to Convolution+Transformer Network
<center><img src='{{"/assets/img/uniformer/uniformer-table1.PNG" | relative_url}}' width="90%"></center>
<br>
이전 연구들은 self-attention이 convolution을 수행할 수 있음을 입증했지만 convolution을 결합하는 대신 convolution을 대체할 것을 제안하였다. 최근 연구들은 ViT에 convolution을 도입하려는 시도가 있으나 동영상 이해를 위한 시공간적 고려 없이 주로 동영상 인식에 초점을 맞추고 있다. 더욱이 이전의 video transformer에서는 조합이 거의 간단하다. 예를 들어 transformer를 글로벌 attention으로 사용하거나 convolution을 patch stem으로 사용한다. 반대로 UniFormer는 통찰력 있는 통합 프레임워크를 통해 동영상 중복성과 의존성을 모두 해결한다. 로컬 및 글로벌 토큰 선호도 학습을 통해 동영상 분류를 위한 바람직한 계산량-정확도 균형을 달성할 수 있다.

## Experiments
- 데이터셋: Kinetics-400, Kinetics-600

### 1. Comparison to State-of-the-art
다음은 Kinetics-400과 600에서 SOTA 방법들과 비교한 표이다.

<center><img src='{{"/assets/img/uniformer/uniformer-table2.PNG" | relative_url}}' width="90%"></center>
<br>
다음은 Something-Something V1과 V2에서 SOTA 방법들과 비교한 표이다.

<center><img src='{{"/assets/img/uniformer/uniformer-table3.PNG" | relative_url}}' width="90%"></center>

### 2. Ablation Studies
다음은 구조 디자인에 대한 ablation study 결과이다. (Kinetics-400에서 50 epochs 동안 학습)

<center><img src='{{"/assets/img/uniformer/uniformer-table4a.PNG" | relative_url}}' width="90%"></center>
<br>
다음은 튜브 크기에 대한 ablation study 결과이다.

<center><img src='{{"/assets/img/uniformer/uniformer-table4b.PNG" | relative_url}}' width="22%"></center>
<br>
다음은 transfer learning 결과이다. 

<center><img src='{{"/assets/img/uniformer/uniformer-table4c.PNG" | relative_url}}' width="45%"></center>
<br>
다음은 샘플링 방법에 대한 ablation study 결과이다. 

<center><img src='{{"/assets/img/uniformer/uniformer-table4d.PNG" | relative_url}}' width="29%"></center>
<br>
다음은 다양한 데이터셋에서 Multi-clip/crop 테스트 결과를 비교한 것이다. 

<center><img src='{{"/assets/img/uniformer/uniformer-fig4.PNG" | relative_url}}' width="100%"></center>