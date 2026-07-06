---
title: "[논문리뷰] FlashVGGT: Efficient and Scalable Visual Geometry Transformers with Compressed Descriptor Attention"
last_modified_at: 2026-07-07
categories:
  - 논문리뷰
tags:
  - 3D Vision
  - 3D Reconstruction
  - Pose Estimation
  - Monocular Depth Estimation
  - CVPR
excerpt: "FlashVGGT 논문 리뷰 (CVPR 2026)"
use_math: true
classes: wide
---

> CVPR 2026. [[Paper](https://arxiv.org/abs/2512.01540)] [[Page](https://wzpscott.github.io/flashvggt_page/)] [[Github](https://github.com/wzpscott/flashvggt)]  
> Zipeng Wang, Dan Xu  
> The Hong Kong University of Science and Technology  
> 1 Dec 2025  

<center><img src='{{"/assets/img/flash-vggt/flash-vggt-fig1.webp" | relative_url}}' width="100%"></center>

## Introduction
본 논문은 [VGGT](https://kimjy99.github.io/논문리뷰/vggt)에서 글로벌 추론을 위해 full self-attention이 정말로 필요한가라는 핵심 질문에서 출발하였다. 저자들은 두 가지 중요한 통찰력을 바탕으로 접근 방식을 설계했다.

1. 기존 방법들은 sparse한 keypoint들과 descriptor들로부터 정확한 프레임 간 연관성을 추론할 수 있음을 보여주며, 이는 dense한 토큰 간 attention이 불필요할 수 있음을 시사한다.
2. VGGT의 global attention map은 본질적으로 sparse하며, 대부분의 score가 0에 가깝다. 이는 계산의 상당 부분이 관련 없는 토큰 쌍에 소모됨을 의미한다.

이러한 관찰을 통해 저자들은 긴 입력 시퀀스에 대한 scaling을 유지하면서 글로벌 추론을 더욱 효율적으로 수행할 수 있는 대안을 모색하게 되었다.

본 논문에서는 압축된 descriptor attention을 통해 VGGT의 계산 병목을 극복하는 효율적인 아키텍처인 **FlashVGGT**를 제안하였다. 핵심 혁신은 공간 리샘플링을 통해 각 프레임의 주요 정보를 캡슐화하는 압축된 descriptor 토큰 세트를 생성함으로써 global attention block을 재구성하는 것이다. Global attention은 이미지 토큰에서 descriptor로의 cross-attention을 통해 근사화된다. 이러한 설계는 global attention의 계산 복잡도를 $\mathcal{O}(N^2)$에서 $\mathcal{O}(N^2/r^2)$으로 줄인다. FlashVGGT는 1,000개 이미지 시퀀스에서 VGGT와 유사한 정확도를 유지하면서 inference 속도를 90% 이상 향상시켰다.

또한, descriptor 토큰의 압축성 덕분에 청크 재귀 방식을 통해 매우 긴 시퀀스(ex. 이미지 3,000개)에 대한 scalable inference가 가능하다. 메모리 제한을 초과하는 시퀀스를 처리할 때는 입력을 순차적인 청크로 나눈다. 이전 청크의 descriptor 토큰을 캐싱하고 재사용함으로써, 이후 청크는 전체 시퀀스에 걸쳐 global receptive field를 유지하면서 이전 청크의 컨텍스트를 통합할 수 있다. 특히, 모든 transformer layer의 고해상도 토큰을 캐싱하는 [StreamVGGT](https://arxiv.org/abs/2507.11539)와 달리, 본 방법은 압축된 descriptor만 저장한다. 이를 통해 최대 메모리 사용량을 $r^2$만큼 줄여 훨씬 더 큰 입력과 리소스 제약이 있는 시나리오에서도 scalable reconstruction을 가능하게 한다.

## Method
### 1. Descriptor-Based Global Attention
본 접근 방식은 global attention block에서 표준 self-attention을 descriptor 기반의 cross-attention 메커니즘으로 대체한다. 반면, 인코더, frame attention, reconstruction head들은 최소한의 계산 오버헤드만을 가지므로 변경하지 않는다.

<center><img src='{{"/assets/img/flash-vggt/flash-vggt-fig3.webp" | relative_url}}' width="100%"></center>

##### 공간적으로 압축된 descriptor 토큰
Global block $$\textbf{G} = \textrm{Reshape}(\{\textbf{F}_i^\prime\}_{i=1}^S) \in \mathbb{R}^{K \times C}$$에 대한 입력이 주어지면, 먼저 이를 $$\mathbb{R}^{S \times H \times W \times C}$$로 reshape하여 공간 구조를 복원한다. 여기서 $H$와 $W$는 각 프레임에 대한 2D 패치 토큰 그리드의 높이와 너비이다. 그런 다음, bilinear interpolation을 사용하여 각 프레임의 공간 차원 $(H, W)$을 더 낮은 해상도 $$(\lfloor H/r \rfloor, \lfloor W/r \rfloor)$$로 리샘플링하고, reshape하여 압축된 descriptor 토큰 집합 $\textbf{D}$를 생성한다. ($r$은 압축 계수)

$$
\begin{equation}
\textbf{D} = \textrm{Reshape} (\textrm{Interp} (\textbf{G}, (\lfloor H/r \rfloor, \lfloor W/r \rfloor))) \in \mathbb{R}^{K_d \times C}
\end{equation}
$$

여기서 $K_d = S \times \lfloor H/r \rfloor \times \lfloor W/r \rfloor$이다.

##### 보조 descriptor 토큰
기하학적 일관성을 유지하기 위해 압축된 descriptor에 세 가지 유형의 보조 토큰을 추가한다.

1. 모든 프레임의 카메라 및 register 토큰
2. 첫 번째 이미지의 모든 토큰
3. 평균 프레임 토큰에 대한 k-means clustering을 통해 선택된 키프레임의 모든 토큰

키프레임 선택은 개별 토큰이 아닌 프레임별 평균을 사용하기 때문에 매우 효율적이며, NVIDIA H800 GPU 1개에서 1,000개 이미지에 대해 2초 이내에 수렴한다. 이러한 보조 토큰은 기하학적 기준점 역할을 하여 카메라 파라미터, 월드 좌표계, 대표 시점의 고품질 정보를 보존한다. 이를 통해 descriptor 압축 중 중요한 디테일 손실을 방지하고 전체 시퀀스에 걸쳐 robust한 기하학적 추론을 보장한다.

##### Descriptor attention
Global attention 연산을 cross-attention layer로 재구성한다. 원래의 전체 해상도 토큰 $\textbf{G}$는 query로 사용되고, descriptor $\textbf{D}$는 공유 key와 value로 사용된다. 이를 통해 전체 해상도 토큰은 글로벌 컨텍스트를 나타내는 간결한 descriptor 집합에 의해 업데이트될 수 있다.

$$
\begin{equation}
\textbf{H} = \textrm{CrossAttn}(\textbf{Q}=\textbf{G}, \textbf{KV} = \textbf{D})
\end{equation}
$$

이 연산은 global receptive field를 유지하여 모든 입력 이미지에서 장거리 의존성을 포착하는 모델의 능력을 보존한다.

##### 계산 복잡도
이 디자인은 global block의 계산 복잡도를 크게 줄인다. 표준 self-attention은 $$\mathcal{O}(K^2) = \mathcal{O}(S^2 N^2)$$ 연산을 필요로 한다. Descriptor 기반 cross-attention은 이를 $$\mathcal{O}(K \times K_d) = \mathcal{O}(S^2 N^2 / r^2)$$로 줄인다. $r = 4$인 경우 계산 복잡도 감소는 약 16배이다.

### 2. Chunk-Recursive Inference
<center><img src='{{"/assets/img/flash-vggt/flash-vggt-fig4.webp" | relative_url}}' width="70%"></center>
<br>
저자들은 GPU 메모리 제약을 초과하는 시퀀스에 대한 scalable reconstruction을 위해 청크 재귀 추론 방식을 제안하였다. 이 방식은 이전에 처리된 모든 청크에 걸쳐 글로벌 컨텍스트를 유지하면서 긴 시퀀스를 순차적으로 처리한다.

##### 문제 정의
$S$개의 입력 이미지 시퀀스를 $T$개의 연속적인 청크 $$\{\mathcal{C}_1, \ldots, \mathcal{C}_T\}$$로 나눈다고 하자. 그리고 $t$번째 청크에 대해, 청크 $$\mathcal{C}_t$$에서 생성된 descriptor 토큰을 $$\textbf{D}_t$$라고 하자. $t$번째 단계까지 처리된 모든 청크의 글로벌 정보를 누적하는 메모리 토큰 집합 $$\textbf{M}_t$$를 유지하며, 이를 $$\textbf{M}_0 = \varnothing$$로 초기화한다.

##### 메모리를 갖춘 descriptor attention
청크 $t$에 대해 global attention 계산은 이전에 처리된 모든 청크의 정보를 유지하는 메모리 메커니즘을 통해 과거 컨텍스트를 통합한다. Query는 현재 청크 $$\mathcal{C}_t$$의 고해상도 이미지 토큰 $$\textbf{G}_t$$이다. key와 value는 현재 청크의 descriptor $$\textbf{D}_t$$와 이전 청크의 메모리 토큰 $$\textbf{M}_{t-1}$$을 concat하여 생성된다.

$$
\begin{equation}
\textbf{H}_t = \textrm{CrossAttn}(\textbf{Q} = \textbf{G}_t, \textbf{KV} = [\textbf{M}_{t-1}, \textbf{D}_t])
\end{equation}
$$

($[\cdot, \cdot]$은 시퀀스 차원에 대한 concat 연산)

이 디자인은 현재 청크의 각 토큰이 로컬로 압축된 컨텍스트 $$\textbf{D}_t$$와 글로벌하게 누적된 기록 $$\textbf{M}_{t-1}$$ 모두에 attention할 수 있도록 하여 개별 청크에서 작동하는 동안 전체 시퀀스에 걸쳐 global receptive field를 효과적으로 유지한다.

##### 메모리 업데이트
청크 $t$를 처리한 후, 현재 청크의 descriptor를 추가하여 메모리를 업데이트한다. 긴 시퀀스 동안 메모리 증가를 제한하기 위해, 매 $p$번째 프레임의 descriptor 토큰만 유지하는 dropping 메커니즘을 사용한다. 현재 청크 내의 매 $p$번째 프레임의 descriptor 부분집합을 $$\textbf{D}_t^\textrm{retain} = \textbf{D}_t [::p]$$라고 하면, 메모리는 다음과 같이 업데이트된다.

$$
\begin{equation}
\textbf{M}_t = [\textbf{M}_{t-1}, \textbf{D}_t^\textrm{retain}]
\end{equation}
$$

이 선택적 업데이트 규칙은 메모리 $$\textbf{M}_t$$가 전체 시퀀스 기록을 간결하게 표현하는 동시에 크기가 프레임 수에 따라 선형적으로 증가하지 않도록 제한한다.

##### 계산 복잡도
청크 재귀 방식은 [StreamVGGT](https://arxiv.org/abs/2507.11539)의 단순한 KV 캐싱 방식보다 상당히 메모리 효율성이 향상된다. Global attention block이 $L$개 있을 때, StreamVGGT의 메모리 사용량은 $\mathcal{O}(KL)$이지만, 본 논문에서 제시하는 접근 방식은 이를 $\mathcal{O}(KL/(pr^2))$로 줄인다.

## Experiment
- 데이터셋: BlendedMVS, CO3Dv2, ScanNet, Mapillary, Arkitscenes, MVSSynth, VirtualKitti
- 구현 디테일
  - $r = 4$, $p = 5$
  - 먼저 2~24개의 뷰로 학습한 후, 시퀀스들에 대해 finetuning

### 1. Monocular and Sparse Reconstruction
다음은 카메라 포즈 추정 결과이다.

<center><img src='{{"/assets/img/flash-vggt/flash-vggt-table1.webp" | relative_url}}' width="75%"></center>
<br>
다음은 monocular depth estimation 결과이다.

<center><img src='{{"/assets/img/flash-vggt/flash-vggt-table2.webp" | relative_url}}' width="72%"></center>

### 2. Long-sequence Dense 3D Reconstruction
다음은 대규모 장면에 대한 3D reconstruction 결과이다.

<center><img src='{{"/assets/img/flash-vggt/flash-vggt-table3.webp" | relative_url}}' width="100%"></center>
<span style="display: block; margin: 1px 0;"></span>
<center><img src='{{"/assets/img/flash-vggt/flash-vggt-fig5.webp" | relative_url}}' width="100%"></center>

### 3. Online Dense 3D Reconstruction
다음은 online reconstruction 결과이다.

<center><img src='{{"/assets/img/flash-vggt/flash-vggt-table4.webp" | relative_url}}' width="63%"></center>
<span style="display: block; margin: 1px 0;"></span>
<center><img src='{{"/assets/img/flash-vggt/flash-vggt-fig6.webp" | relative_url}}' width="58%"></center>

### 4. Model Analysis and Discussion
다음은 공간 압축 방법에 대한 ablation 결과이다.

<center><img src='{{"/assets/img/flash-vggt/flash-vggt-table5.webp" | relative_url}}' width="49%"></center>
<br>
다음은 압축 비율 $r$에 대한 ablation 결과이다.

<center><img src='{{"/assets/img/flash-vggt/flash-vggt-fig7.webp" | relative_url}}' width="49%"></center>
<br>
다음은 보조 descriptor 토큰에 대한 ablation 결과이다.

<center><img src='{{"/assets/img/flash-vggt/flash-vggt-fig8.webp" | relative_url}}' width="80%"></center>
<br>
다음은 신뢰도 맵을 VGGT와 비교한 결과이다.
  
<center><img src='{{"/assets/img/flash-vggt/flash-vggt-fig9.webp" | relative_url}}' width="60%"></center>