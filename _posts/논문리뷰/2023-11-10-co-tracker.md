---
title: "[논문리뷰] CoTracker: It is Better to Track Together"
last_modified_at: 2023-11-10
categories:
  - 논문리뷰
tags:
  - Transformer
  - Computer Vision
  - AI
  - Meta AI
excerpt: "CoTracker 논문 리뷰"
use_math: true
classes: wide
---

> arXiv 2023. [[Paper](https://arxiv.org/abs/2307.07635)] [[Page](https://co-tracker.github.io/)] [[Github](https://github.com/facebookresearch/co-tracker)]  
> Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, Christian Rupprecht  
> Meta AI | Visual Geometry Group, University of Oxford  
> 14 Jul 2023  

<center><img src='{{"/assets/img/co-tracker/co-tracker-fig1.PNG" | relative_url}}' width="100%"></center>

## Introduction
점 대응을 설정하는 것은 컴퓨터 비전의 근본적인 문제이며 많은 다운스트림 task에 필요하다. 본 논문은 동적 객체와 움직이는 카메라가 포함된 동영상에서 점 대응을 추정하는 데 관심이 있다. 이 task를 흔히 모션 추정이라고 하며, 3D 장면의 특정 물리적 포인트의 projection인 하나 이상의 2D 포인트가 주어지면 동영상의 다른 모든 프레임에서 동일한 물리적 포인트의 위치를 찾는 것이 목표이다. 

동영상 모션 추정 문제에는 두 가지 변형이 있다. Optical flow의 목적은 동영상 프레임 내 모든 포인트의 속도를 추정하는 것이다. 이 추정은 모든 점에 대해 공동으로 수행되지만 동작은 아주 작은 거리에서만 예측된다. 추적의 목표는 장기간에 걸쳐 점의 움직임을 추정하는 것이다. 효율성과 모델링 단순성을 위해 추적 방법은 일반적으로 sparse한 점 선택에 초점을 맞추고 이를 통계적으로 독립적인 것으로 처리한다. 최신 심층망을 사용하고 가려짐이 있는 경우에도 점을 추적할 수 있는 [TAP-Vid](https://arxiv.org/abs/2211.03726)와 [Particle Video Revisited](https://arxiv.org/abs/2204.04153)와 같은 최신 기술조차도 모델 추적을 독립적으로 수행한다. 이 근사는 점들이 강한 상관관계를 갖는 경우가 많기 때문에 조잡하다. 

본 논문에서는 추적된 포인트 간의 상관관계를 설명하면 추적 정확도가 크게 향상될 수 있다는 가설을 세웠다. 따라서 긴 동영상 시퀀스에서 여러 포인트의 공동 추적을 지원하는 새로운 neural tracker를 제안한다. 본 논문의 디자인은 optical flow와 포인트 추적을 위한 심층망에 대한 이전 연구들에서 영감을 얻었지만 여러 트랙의 공동 추정 및 긴 동영상 처리를 위한 몇 가지 변경 사항이 있다. 

본 논문의 신경망은 하나의 동영상과 여러 개의 시작 트랙 위치를 입력으로 받아 전체 트랙을 출력한다. 동영상의 모든 공간적 위치와 시간에서 선택된 임의의 포인트를 추적할 수 있으므로 디자인이 유연하다.

네트워크는 트랙의 대략적인 초기 버전을 가져온 다음 동영상 콘텐츠와 더 잘 일치하도록 점진적으로 개선하는 방식으로 작동한다. 트랙은 간단하게 초기화된다. 트랙 포인트나 트랙 조각이 주어지면 트랙의 나머지 부분은 포인트가 고정되어 있다고 가정하여 초기화된다. 이러한 방식으로 트랙은 동영상 중간에서도 어느 포인트에서나 시작하거나 sliding-window 방식으로 작동할 때 추적기 자체의 출력에서 시작하여 초기화할 수 있다. 이러한 모든 경우는 동일한 아키텍처에서 원활하게 지원된다. 

네트워크 자체는 토큰의 2D 그리드에서 작동하는 transformer이다. 첫 번째 그리드 차원은 시간을 나타내고 두 번째 그리드 차원은 추적된 포인트들의 집합이다. 적합한 self-attention 연산자를 통해 transformer는 window의 지속 시간 동안 각 트랙을 전체적으로 고려할 수 있으며 트랙 간 정보를 교환하여 상관 관계를 활용할 수 있다. 이 연산에 특히 효과적인 것으로 입증된 합성 [TAP-Vid-Kubric](https://arxiv.org/abs/2211.03726)을 기반으로 신경망을 학습시킨다. 

저자들은 세 가지 벤치마크를 통해 트래커를 평가하였다. 저자들은 평가 프로토콜의 공정성을 보장하기 위해 평가 프로토콜의 설계에 특별한 주의를 기울였다. 특히, 기존 벤치마크에는 몇 가지 전경 객체에 집중된 실제 트랙이 포함되어 있어 잠재적으로 이러한 객체의 존재를 공동 추적기에 공개할 수 있다는 점에 주목하였다. 실제 정보가 추적기로 유출되지 않도록 하기 위해 추적된 포인트의 다양한 분포를 테스트하였다. 특히, 저자들은 한 번에 하나의 벤치마크 트랙을 추적하지만 모델이 공동 추적을 수행할 수 있도록 그리드에 포인트들을 추가하였다. 이러한 방식으로 단일 포인트 추적기와 공평하게 비교하였다. 

본 논문의 아키텍처는 단일 포인트 추적에 적합하고 포인트들의 그룹에도 탁월하여 여러 벤치마크에서 SOTA 추적 성능을 얻었다. 특히 긴 트랙의 경우에도 TAP-Vid와 FastCapture 데이터셋에서의 성능을 크게 향상시켰다.

## CoTracker
본 논문의 목표는 동영상이 재생되는 동안 2D 포인트를 추적하는 것이다. 문제를 다음과 같이 공식화한다. 동영상 $V = (I_t)_{t=1}^T$는 $T$개의 RGB 프레임 $I_t \in \mathbb{R}^{3 \times H \times W}$의 시퀀스이다. 추적은 $N$개의 포인트 각각에 대하여 트랙

$$
\begin{equation}
P_t^i = (x_t^i, y_t^i) \in \mathbb{R}^2, \quad t = t^i, \ldots, T, \; i = 1, \ldots, N
\end{equation}
$$

를 생성하는 것과 같다. 여기서 $$t^i \in \{1, \ldots, T\}$$는 트랙이 시작되는 시간을 나타낸다. 또한 visibility flag $$v_t^i \in \{0, 1\}$$는 주어진 프레임에서 포인트가 보이는지 또는 가려지는지 여부를 알려준다. 트랙의 시작점이 대응하는 물리적 포인트를 명확하게 식별한다고 가정한다. 이를 위해서는 포인트가 시작 시 표시되어야 한다 (즉, $v_{t^i}^i = 1$). 트랙은 동영상 도중 언제든지 $1 \le t^i \le T$에서 시작할 수 있다. 

추적기는 $N$개의 트랙의 시작 위치와 시간 $(P_{t^i}^i, t^i)_{i=1}^N$뿐만 아니라 동영상 $V$를 입력으로 취하고 해당 트랙의 추정 $$\hat{P}_t^i = (\hat{x}_t^i, \hat{y}_t^i)$$를 출력하는 알고리즘이며, 모든 유효한 시간 ($t \ge t^i$)을 추적한다. 또한 visibility flag의 추정값 $$\hat{v}_i^t$$를 예측하는 task를 추적기에 맡긴다. 이 값 중 초기 값인 $$\hat{P}_{t^i}^i = P_{t^i}^i$$와 $$\hat{v}_{t^i}^i = v_{t^i}^i = 1$$만 추적기에 알려지고 나머지는 예측해야 한다.

### 1. Transformer formulation
<center><img src='{{"/assets/img/co-tracker/co-tracker-fig2.PNG" | relative_url}}' width="100%"></center>
<br>
본 논문은 **CoTracker**라고 부르는 transformer 네트워크 $\Psi : G \mapsto O$를 사용하여 이 예측 문제에 접근한다. Transformer의 목표는 주어진 트랙 추정치를 향상시키는 것이다. 트랙은 각 트랙 $i = 1, \ldots, N$와 시간 $t = 1, \ldots, T$에 대해 하나씩 입력 토큰 $G_t^i$의 그리드로 인코딩된다. 업데이트된 트랙은 출력 토큰 $O_t^i$의 해당 그리드로 표현된다. 이러한 토큰은 optical flow를 위한 [RAFT](https://arxiv.org/abs/2003.12039)와 추적을 위한 [PIP](https://arxiv.org/abs/2204.04153)에서 영감을 받은 디자인을 사용하여 구축되었다. 전체 개요는 위 그림에 나와 있고 모델 구성 요소는 아래 그림에 나와 있다. 

<center><img src='{{"/assets/img/co-tracker/co-tracker-fig3.PNG" | relative_url}}' width="100%"></center>

##### Image features
각 동영상 프레임 $I_t$에서 transformer와 함께 end-to-end로 학습한 CNN을 사용하여 $d$차원 외관 feature $\phi (I_t) \in \mathbb{R}^{d \times \frac{H}{k} \times \frac{W}{k}}$를 추출한다. 여기서 $k = 8$은 효율성을 위해 활용되는 downsampling factor이다. 이러한 feature 중 stride $s = 1, \ldots, S$를 갖는 몇 가지 스케일링된 버전 $\phi (I_t; s) \in \mathbb{R}^{d \times \frac{H}{sk} \times \frac{W}{sk}}$를 고려한다. 이러한 축소된 feature들은 $s \times s$ 이웃들의 기본 feature들에 average pooling을 적용하여 얻는다. 저자들은 $S = 4$ 스케일을 사용하였다. 

##### Track features
트랙의 모양은 feature 벡터 $Q_t^i \in \mathbb{R}^d$에 의해 캡처된다. 이는 트랙 모양의 변화를 수용하기 위해 시간에 따라 달라진다. 처음에는 이미지 feature들을 샘플링하여 초기화된 다음 신경망에 의해 업데이트된다.

##### Correlation features
트랙과 이미지의 매칭을 용이하게 하기 위해 상관 관계 feature $C_t^i \in \mathbb{R}^S$를 도입한다. 이 feature는 트랙 feature $Q$와 현재 트랙 위치 $$\hat{P}_iˆt$$ 주변의 이미지 feature $\phi (I_t; s)$를 비교하여 얻는다. 구체적으로, 벡터 $C_t^i$는 내적

$$
\begin{equation}
\langle Q_t^i, \phi(I_t; s) [\hat{P}_t^i / ks + \delta] \rangle \\
s = 1, \ldots, S, \; \delta \in \mathbb{Z}^2, \; \| \delta \|_1 \le \Delta, \; \Delta \in \mathbb{N}
\end{equation}
$$

을 쌓아서 얻는다. 여기서 오프셋 $\delta$는 점 $$\hat{P}_i^t$$의 이웃을 정의한다. 이미지 feature $\phi (I_t; s)$는 bilinear interpolation과 zero padding을 사용하여 정수가 아닌 위치에서 샘플링된다. $S = \Delta = 4$인 경우 $C_t^i$의 차원은 $(2(\Delta^2 + \Delta) + 1)S = 164$이다. 

##### Tokens
입력 토큰 $G(\hat{P}, \hat{v}, Q)$는 트랙의 위치, visibility, 모양, 상관 관계에 대한 코드이다. 이 정보는 해당 feature들을 쌓아서 표시된다.

$$
\begin{equation}
G_t^i = (\hat{P}_t^i, \textrm{logit}(\hat{v}_t^i), Q_t^i, C_t^i, \eta (\hat{P}_t^i - \hat{P}_1^i))
\end{equation}
$$

마지막 성분을 제외한 모든 성분은 위에서 소개되었다. 마지막 성분은 추정된 위치에서 파생된다. 이는 시간 $t = 1$의 초기 위치에 대한 트랙 위치의 sinusoidal 위치 인코딩 $\eta$이다. 후자의 정보는 $$\hat{P}_t^i$$만 관찰하여 transformer에 의해 추론될 수 있지만 저자들은 직접 전달하는 것도 유익하다는 것을 알았다. 

출력 토큰 $O(\hat{P}^\prime, Q^\prime)$에는 업데이트된 위치와 모양 feature만 포함된다.

$$
\begin{equation}
O_t^i = (\hat{P}_t^{i \prime}, Q_t^{i \prime})
\end{equation}
$$

##### Iterated transformer applications
트랙 추정을 점진적으로 개선하기 위해 transformer를 $M$번 적용한다. $m = 0, 1, \ldots, M$은 추정값을 인덱싱하며, $m = 0$은 초기화를 나타낸다. 

$$
\begin{equation}
O (\hat{P}^{(m+1)}, Q^{(m+1)}) = \Psi (G (\hat{P}^{(m)}, \hat{v}^{(0)}, Q^{(m)}))
\end{equation}
$$

Visibility 마스크 $\hat{v}$는 transformer에 의해 업데이트되지 않는다. 대신 $\hat{v} (M) = \sigma (W Q^{(M)})$로 transformer의 $M$번 적용이 끝나면 한 번 업데이트된다. 여기서 $\sigma$는 sigmoid activation 함수이고 $W$는 학습된 가중치 행렬이다. 저자들은 visibility에 대한 반복적인 업데이트가 성능을 더 이상 향상시키지 않는다는 것을 발견했다. 이는 visibility가 먼저 정확한 위치를 예측하는 데 크게 좌우된다는 사실 때문일 것이다.

$$\hat{P}ˆ{(0)}$$, $$v^{(0)}$$, $$Q^{(0)}$$는 트랙의 시작 위치 및 시간부터 초기화된다. 모든 트랙 $i = 1, \ldots, N$과 시간 $t = 1, \ldots, T$에 대하여, 저자들은 다음과 같이 단순하게 설정하였다. 

$$
\begin{equation}
\hat{P}_t^{i, (0)} \leftarrow P_{t^i}^i, \quad \hat{v}_t^{i, (0)} \leftarrow 1, \quad [Q_t^{i, (0)}]_s \leftarrow \phi (I_{t^i}; s) [P_{t^i}^i/ks]
\end{equation}
$$

이를 통해 $t^i$를 다른 모든 시간 $t = 1, \ldots, T$에 효과적으로 broadcasting한다. 

### 2. Windowed inference
본 논문의 transformer 디자인의 장점은 매우 긴 동영상을 처리하기 위해 windowed application을 쉽게 지원할 수 있다는 것이다. 특히 아키텍처에서 지원하는 최대 창 길이 $T$보다 긴 길이 $T^\prime > T$의 동영상 $V$를 고려하자. 전체 동영상 $V$ 전체에서 포인트들을 추적하기 위해 동영상을 길이 $T$의 $J = \langle 2T^\prime / T − 1 \rangle$개의 window로 분할하고 $T/2$개의 프레임이 겹친다. 

동영상을 처리하기 위해 transformer를 $MJ$번 적용한다. 첫 번째 window의 출력은 두 번째 window의 입력으로 사용된다. 위첨자 $(m, j)$는 $j$번째 윈도우에 적용된 transformer의 $m$번째 업데이트를 나타낸다. 따라서 transformer 반복과 window에 걸쳐 $(\hat{P}^{(m,j)}, \hat{V}^{(m,j)}, \hat{Q}^{(m,j)})$의 $M \times J$ 그리드를 갖게 된다. $m = 0$, $j = 1$부터 시작하여 이러한 값들을 초기화한다. 그런 다음 transformer를 $M$번 적용하여 상태 $(\hat{P}^{(M,1)}, \hat{v}^{(M,1)}, Q^{(M,1)})$를 얻는다. 이것으로부터 두 번째 window에 대해 $(\hat{P}^{(0,2)}, \hat{v}^{(0,2)}, Q^{(0,2)})$를 초기화한다. 구체적으로, $\hat{P}^{(0,2)}$의 처음 $T/2$개의 성분은 $\hat{P}^{(M,1)}$의 마지막 $T/2$개의 성분의 복사본이다. $\hat{P}^{(0,2)}$의 마지막 $T/2$개의 성분은 $\hat{P}ˆ{(M,1)}$의 마지막 시간 $t = T/2 − 1$의 복사본이다. $\hat{v}^{(0,2)}$에는 동일한 업데이트 규칙이 사용되는 반면 $Q^{(0,j)}$는 항상 초기 추적 feature $Q$로 초기화된다. $(\hat{P}ˆ{(0,2)}, \hat{v}^{(0,2)}, Q^{(0,2)})$를 초기화한 후, transformer는 두 번째 window에 $M$번 더 적용되고 다음 window에도 이 과정이 반복된다. 마지막으로 초기화를 사용하여 토큰 그리드를 확장하면 새로운 트랙이 추가된다.

### 3. Unrolled learning
반 겹쳐진 window를 적절하게 처리하기 위해 펼쳐진 방식으로 windowed transformer를 학습하는 것이 중요하다. 주요 loss는 반복된 transformer 적용과 window에 대해 합산된 트랙 regression에 대한 것이다. 

$$
\begin{equation}
\mathcal{L}_1 (\hat{P}, P) = \sum_{j=1}^J \sum_{m=1}^M \gamma^{M-m} \| \hat{P}^{(m,j)} - P^{(j)} \|
\end{equation}
$$

여기서 $\gamma = 0.8$은 초기 transformer 업데이트를 감가한다. 여기서 $P^{(j)}$는 window $j$로 제한된 ground-truth 궤적을 포함한다 (window 중앙에서 시작하는 궤적은 뒤로 패딩된다). 두 번째 loss는 visibility flag의 cross entropy이다. 

$$
\begin{equation}
\mathcal{L}_2 (\hat{v}, v) = \sum_{j=1}^J \textrm{CE} (\hat{v}^{(M,j)}, v^{(j)})
\end{equation}
$$

계산 비용으로 인해 학습 중 loss에는 적당한 수의 window만 사용되지만, 테스트 시에는 windowed transformer 적용을 임의로 펼쳐서 원칙적으로 모든 동영상 길이를 처리할 수 있다. 

펼쳐진 inference를 통해 동영상 후반부에 나타나는 추적 포인트들을 사용할 수 있다. 먼저 나타나는 sliding window에서만 포인트 추적을 시작한다. 또한 시퀀스의 중간 프레임에서 보이는 포인트를 샘플링하여 이러한 포인트가 학습 데이터에 존재하는지 확인한다. 

### 4. Transformer
본 논문의 transformer는 입력과 출력에 두 개의 linear layer가 적용된 시간 및 그룹 attention 블록으로 구성된다. 시간 및 포인트 트랙에 걸쳐 attention을 고려하면 모델이 계산적으로 다루기 쉬워진다. 즉, 복잡도가 $O(N^2 T^2)$에서 $O(N^2 + T^2)$로 감소된다. Transformer가 입력으로 사용하는 추정된 궤적에 대한 위치 인코딩 $\eta$ 외에도 표준 sinusoidal 위치 인코딩을 추가한다 (시간의 경우 1차원, 공간의 경우 2차원). 

### 5. Point Selection
여러 포인트를 동시에 추적하면 모델이 동영상의 모션과 트랙 간의 상관 관계에 대해 더 나은 추론을 할 수 있다. 

그러나 벤치마크 데이터셋에서 방법을 평가할 때는 주의가 필요하다. 이러한 데이터셋에서 사람이 주석을 추가한 포인트들은 움직이는 물체의 돌출된 위치에 있는 경우가 많다. 포인트 선택과 관련하여 성능 수치가 견고함을 보장하고 기존 방법과 엄격하게 공정한 비교를 보장하기 위해 한 번에 하나의 타겟 ground truth 포인트만 사용하여 모델을 평가한다. 이는 데이터셋의 포인트 분포에서 모델 성능을 분리한다.

<center><img src='{{"/assets/img/co-tracker/co-tracker-fig5.PNG" | relative_url}}' width="100%"></center>
<br>
위 그림에 시각화된 두 가지 점 선택 전략을 실험한다. "글로벌" 전략을 사용하면 전체 이미지에 걸쳐 일반 그리드에서 추가 점을 선택하기만 하면 된다. "로컬" 전략을 사용하면 타겟 포인트 주변의 일반 그리드를 사용하여 타겟 포인트에 가까운 추가 포인트를 선택하므로 모델이 해당 인근 지역에 집중할 수 있다. 포인트 선택은 inference 시에만 사용된다.

## Experiments
- 데이터셋: FlyingThings++, TAP-Vid-Kubric

### 1. Results
다음은 다양한 보조 그리드에 대한 성능을 비교한 표이다. 

<center><img src='{{"/assets/img/co-tracker/co-tracker-table1.PNG" | relative_url}}' width="70%"></center>
<br>
다음은 TAP-Vid에서 성능을 비교한 표이다. 

<center><img src='{{"/assets/img/co-tracker/co-tracker-table2.PNG" | relative_url}}' width="73%"></center>
<br>
다음은 포인트 추적 벤치마크에서의 비교 결과이다. 

<center><img src='{{"/assets/img/co-tracker/co-tracker-table3.PNG" | relative_url}}' width="40%"></center>
<br>
다음은 Kubric에서 학습한 PIP와 성능을 비교한 표이다. 

<center><img src='{{"/assets/img/co-tracker/co-tracker-table4.PNG" | relative_url}}' width="80%"></center>
<br>
다음은 다른 방법들과 정성적으로 비교한 결과이다. 

<center><img src='{{"/assets/img/co-tracker/co-tracker-fig6.PNG" | relative_url}}' width="100%"></center>
<br>
다음은 sliding window를 펼치는 것에 대한 ablation 결과이다. 

<center><img src='{{"/assets/img/co-tracker/co-tracker-table5.PNG" | relative_url}}' width="42%"></center>

### 2. Limitations
1. Sliding window 기반 방법이므로 하나의 window 크기보다 길게 가려지면 포인트를 추적할 수 없다. 
2. Transformer 복잡도는 추적된 포인트 수에 따라 2차이므로 dense prediction에 이 기술을 쉽게 적용할 수 없다. 