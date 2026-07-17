---
title: "[논문리뷰] ArtiFixer: Enhancing and Extending 3D Reconstruction with Auto-Regressive Diffusion Models"
last_modified_at: 2026-07-17
categories:
  - 논문리뷰
tags:
  - Diffusion
  - 3D Reconstruction
  - 3D Vision
  - NVIDIA
  - SIGGRAPH
excerpt: "ArtiFixer 논문 리뷰 (SIGGRAPH 2026)"
use_math: true
classes: wide
---

> SIGGRAPH 2026. [[Paper](https://arxiv.org/abs/2603.00492)] [[Page](https://research.nvidia.com/labs/sil/projects/artifixer/)] [[Github](https://github.com/nv-tlabs/artifixer)]  
> Riccardo de Lutio, Tobias Fischer, Yen-Yu Chang, Yuxuan Zhang, Jay Zhangjie Wu, Xuanchi Ren, Tianchang Shen, Katarina Tothova, Zan Gojcic, Haithem Turki  
> NVIDIA | ETHZ | Cornell University | University of Toronto | Vector Institute  
> 28 Feb 2026  

<center><img src='{{"/assets/img/arti-fixer/arti-fixer-fig1.webp" | relative_url}}' width="100%"></center>

## Introduction
3D reconstruction과 3D 생성을 독립적으로 취급하는 대신, 본 논문은 이들의 상호 보완적인 강점을 결합하고자 하였다. 생성 모델은 불완전한 reconstruction을 복구하고 완성하는 강력한 prior 역할을 하며, 명시적이지만 노이즈가 많고 부분적인 3D 표현은 생성을 뒷받침하고 장기적인 드리프트를 완화하며 착시 현상을 억제하는 강력한 컨디셔닝 신호를 제공한다.

최근 연구들은 손상된 새로운 시점의 렌더링을 깨끗한 이미지로 매핑하는 생성 모델을 학습시키고, 그 결과로 얻은 개선 사항을 다시 3D 표현으로 추출하는 방식을 사용하였다. 그러나 이러한 접근 방식은 두 가지 근본적인 trade-off를 해결해야 한다.

1. 시간적 일관성과 효율성 사이의 trade-off
  - 동영상 생성 모델을 사용하면 시간적 일관성이 강력하지만 계산 비용이 높다.
  - 이미지 기반 생성 모델을 사용하면 효율적이지만 시간적 일관성이 제한된다.
2. 컨디셔닝 강도와 생성 용량 사이의 trade-off
  - Concat이나 cross-attention을 통해 손상된 렌더링을 컨디셔닝하면 관찰된 장면 콘텐츠를 변경할 위험이 있다.
  - 손상된 렌더링을 깨끗한 이미지에 직접 매핑하도록 학습시키면 모든 입력 픽셀이 검은색인 완전히 관찰되지 않은 영역에서 mode collapse가 발생하여 누락된 콘텐츠를 합성할 수 없다.

본 논문에서는 사전 학습된 양방향 video diffusion model을 카메라 제어 가능한 생성 모델로 적용하여 손상된 렌더링 이미지를 깨끗한 이미지로 매핑하였다. 앞서 언급한 한계를 극복하기 위해 다음과 같은 방법을 사용하였다.

1. 양방향 모델을 few-step causal autoregressive 생성 모델로 distillation하여 기존 이미지 기반 방법의 효율성에 근접하면서 임의의 시간 동안 시간적으로 일관된 동영상을 생성할 수 있도록 하였다.
2. 불투명도를 고려한 noise 혼합 전략을 통해 불투명도가 낮은 영역에 Gaussian noise를 주입하여 mode collapse를 방지하고 관찰되지 않은 영역의 생성 능력을 보존한다.

본 논문에서 제시하는 방법은 reconstruction 조건이 생성 모델을 개선하고, 생성 모델은 다시 reconstruction을 향상시키고 확장하여 더 높은 품질의 동영상 합성 및 향상된 3D 장면 완성도를 가능하게 한다. 결과적으로, 이 프레임워크는 3D reconstruction을 효율적으로 개선하고 다양한 벤치마크에서 baseline들을 크게 능가하는 성능을 보여준다.

## Method
<center><img src='{{"/assets/img/arti-fixer/arti-fixer-fig2.webp" | relative_url}}' width="100%"></center>
<br>
소수의 이미지로 구성된 초기 3D reconstruction이 주어졌을 때, 본 논문의 목표는 입력 이미지에서 관찰되지 않은 영역을 포함하여 임의의 카메라 시점에서 아티팩트 없는 렌더링을 빠르게 생성하는 것이다. 본 논문에서는 제어 가능한 autoregressive 동영상 생성 모델을 사용했으며, 임의의 길이의 새로운 시점 렌더링을 직접 수행하거나 3D reconstruction을 개선하기 위한 pseudo-supervision을 제공할 수 있다.

### 1. Bidirectional Training
<center><img src='{{"/assets/img/arti-fixer/arti-fixer-fig3.webp" | relative_url}}' width="55%"></center>

##### 아키텍처
사전 학습된 Wan 2.1 T2V-14B에서 시작하여 VAE와 텍스트 인코더를 고정하고 나머지 구성 요소를 fine-tuning한다. 저하된 렌더링은 고정된 VAE로 인코딩되고 $(t, h, w) = (1, 2, 2)$로 patchify된다. 렌더링된 opacity map $\textbf{O}$를 통해 장면 콘텐츠를 생성할 위치를 지정하고, per-pixel Plücker raymap $\textbf{R}$을 통해 완전히 관찰되지 않은 영역에서 카메라 제어를 가능하게 한다. $\textbf{R}$은 각 픽셀에 광선 방향 $\textbf{d}$와 카메라 중심 $\textbf{o}$로 구성된 6차원 벡터 $(\textbf{d}, \textbf{o} \times \textbf{d})$를 할당한다. 두 신호 모두 VAE를 완전히 우회한다. PixelUnshuffle 연산을 통해 공간 차원을 VAE의 공간 압축률과 일치하도록 축소하고, block별 linear layer $f_o$와 $f_r$을 통해 인코딩한 다음, 임베딩을 비주얼 토큰에 더한다.

$$
\begin{aligned}
T_r &:= T_s + f_r (\textrm{PixelUnshuffle}(\textbf{R})) \\
T_o &:= T_r + f_o (\textrm{PixelUnshuffle}(\textbf{O}))
\end{aligned}
$$

($T_s$는 self-attention과 layer-normalization을 적용한 후의 토큰 집합)

이 전략은 $\textbf{R}$과 $\textbf{O}$를 VAE 인코딩하는 대안보다 계산 효율성이 더 높으며, 입력 렌더링이 완전히 비어 있는 경우에도 카메라 제어를 제공한다.

추가적인 장면 컨텍스트를 제공하기 위해, batch 차원을 따라 이미지별로 patchify된 고정된 VAE를 사용하여 깨끗한 레퍼런스 뷰를 인코딩한다. 각 transformer block은 타겟 토큰 $Q$에서 concat된 레퍼런스 토큰으로 cross-attention을 수행하며, 이 레퍼런스 토큰은 추가 linear projection $K_n$과 $V_n$을 통해 key와 value로 매핑된다. Cross-attention 출력은 Wan 2.1의  image-to-video 모델을 따라 타겟 토큰 $Q$에 다시 더해진다. 저자들은 [PRoPE](https://arxiv.org/abs/2507.10496)를 이 cross-attention 내에서만 적용하였으며, $Q$에는 타겟 intrinsics/extrinsics을, $K_n$, $V_n$에는 레퍼런스 intrinsics/extrinsics들을 사용한다. $f_r$, $f_o$, $V_n$은 모두 사전 학습된 가중치와의 호환성을 보장하기 위해 0으로 초기화된다.

##### 불투명도 혼합
기존 방법 대부분은 Gaussian noise $$\boldsymbol{\epsilon} \sim \mathcal{N}(0, \textbf{I})$$에서 시작하여 채널 concat 또는 [classifier-free guidance](https://kimjy99.github.io/논문리뷰/cfdg)를 통해 초기 열화된 렌더링의 latent $$\textbf{z}_\textrm{deg}$$로 생성 프로세스를 컨디셔닝한다. 결과적으로 생성된 latent $$\textbf{z}_\textrm{enh}$$는 $$\textbf{z}_\textrm{deg}$$와 semantic하게 유사한 경향이 있지만, 특히 아티팩트가 많은 영역에서는 상당한 불일치가 발생한다. 몇몇 방법은 noise 대신 $$\textbf{z}_\textrm{deg}$$에서 직접 시작하여 더 강력한 일관성을 보장하였지만, 완전히 처음 보는 영역에서 mode collapse 문제가 발생하고, 렌더링을 extrapolation하는 능력을 저해한다.

<center><img src='{{"/assets/img/arti-fixer/arti-fixer-fig4.webp" | relative_url}}' width="100%"></center>
<br>
이를 해결하기 위해, 저자들은 $$\textbf{z}_\textrm{deg}$$의 공간 차원과 일치하도록 max pooling을 통해 $\textbf{O}$를 $$\textbf{O}_z$$로 downscaling하여 낮은 불투명도 영역에 Gaussian noise를 혼합하였다.

$$
\begin{equation}
\textbf{z}_\textrm{mix} = \textbf{O}_z \textbf{z}_\textrm{deg} + (1 - \textbf{O}_z) \boldsymbol{\epsilon}
\end{equation}
$$

Max pooling에서 소스 정보가 손실되지 않으므로, 이 접근 방식은 $$\textbf{z}_\textrm{deg}$$에서 시작하는 일관성 이점을 유지하면서 완전히 새로운 영역에서 Gaussian prior로 자연스럽게 interpolation한다.

##### 데이터 큐레이션
본 논문의 목표는 관측이 부족한 영역의 아티팩트를 수정하는 것뿐만 아니라, 전혀 관측되지 않은 영역에서도 그럴듯한 콘텐츠를 생성하는 것이다. 이를 위해, 모델이 채워 넣어야 할 빈 영역이 많은 sparse reconstruction을 유도하는 카메라 선택 전략을 사용하여 DL3DV-10K 데이터셋에서 reconstruction-GT 샘플 쌍을 생성하였다.

Rotation $$\textbf{R}_i$$와 translation $$\textbf{t}_i$$를 갖는 카메라 포즈 세트가 주어졌을 때, 먼저 다음과 같이 카메라 포즈 거리 $d_{ij}$를 측정한다.

$$
\begin{equation}
d_{ij} = \frac{\theta_{ij}}{\pi} + \frac{\| \textbf{t}_i - \textbf{t}_j \|}{\bar{r}} \\
\textrm{where} \quad \bar{r} = \frac{1}{N} \sum_k \| \textbf{t}_k \|_2
\end{equation}
$$

($$\theta_{ij} \in [0, \pi]$$는 $$\textbf{R}_i$$와 $$\textbf{R}_j$$ 사이의 SO(3) geodesic angle)

그런 다음, 가장 먼 거리를 가진 카메라 쌍 $(P_1, P_2)$를 찾아 시드 그룹 $G_1$과 $G_2$를 생성한다. 나머지 카메라들은 $P_1$, $P_2$와의 거리를 기준으로 $G_1$ 또는 $G_2$에 할당되고, 각 그룹 내에서 카메라 간 거리가 가장 먼 2~12개의 카메라를 샘플링하여 sparsity가 다양한 reconstruction들을 생성한다. 각 reconstruction의 카메라 scale을 사전 학습된 metric depth 추정 모델을 사용하여 대략적으로 정렬하고, 장면 설명을 위해 Qwen3-VL을 실행한다.

##### 최적화
초기 latent 인코딩 렌더링 $$\textbf{z}_\textrm{deg}$$를 $$\textbf{z}_\textrm{mix}$$로 변환한 후, conditional flow matching loss $$\mathcal{L}_\textrm{cfm}$$을 사용하여 개선된 $$\textbf{z}_\textrm{enh}$$를 예측하도록 모델을 학습시킨다. Reconstruction-GT 쌍을 구성하기 위해, $N = 81$개의 프레임과 해당 카메라 포즈, 텍스트 프롬프트, 균일하게 변화하는 레퍼런스 뷰 개수(0~12)를 샘플링한다. 모델의 생성 능력과 시점 제어 능력을 향상시키기 위해, 입력 데이터의 마지막 $K \le N$ 프레임에서 RGB 렌더링과 불투명도 맵을 모두 0으로 설정하고 Plücker raymap은 유지한다. 따라서 모델은 프롬프트, 레퍼런스 뷰, 카메라 조건만을 사용하여 GT를 재구성해야 한다.

### 2. Causal Distillation
##### 초기화
양방향 teacher 모델의 가중치를 사용하여 causal model을 초기화한다. 학습 안정화를 위해, teacher 모델로부터 ODE 궤적 데이터셋을 생성해야 하는 기존 방법의 ODE 초기화 프로토콜보다 간단한 전략을 사용한다. 구체적으로, block-causal mask를 적용하고, [Diffusion Forcing](https://kimjy99.github.io/논문리뷰/diffusion-forcing)에서처럼 각 입력 프레임에 서로 다른 noise level을 적용한다. 그 외에는 teacher 모델과 동일한 입력 및 학습 프로토콜을 사용한다.

##### Autoregressive rollout
초기화 후, [Self Forcing](https://arxiv.org/abs/2506.08009)과 유사한 학습 전략을 채택하였다. 동영상 청크를 순차적으로 생성하고 KV caching을 통해 이전에 생성된 청크로 컨디셔닝한다. 단, 카메라 제어 및 순수 noise로부터의 생성이 성능 저하를 초래하기 때문에 dropout을 계속 적용한다. 모델을 4-step으로 변환하기 위해 [Distribution Matching Distillation (DMD)](https://kimjy99.github.io/논문리뷰/dmd)을 적용한다.

##### 긴 동영상 생성
기존 방법들은 긴 동영상 rollout에서 오차 누적을 최소화하기 위해 long-horizon training에 의존한다. 이러한 전략들을 본 방법에도 적용할 수 있지만, 실제로는 컨디셔닝 신호만으로도 오차 누적을 방지하기에 충분하다. 따라서 저자들은 양방향 모델 학습과 동일한 프레임 수로 학습하고 inference 시에 rolling KV cache를 사용한다. 이 접근 방식은 간단하지만 주어진 계산 예산 내에서 더 다양하고 짧은 동영상 세트를 사용하여 학습하기 때문에 학습 수렴 속도를 높이고, 임의 길이의 동영상에도 일반화 성능을 보인다.

##### 3D distillation
기존 방법들은 일관성 유지를 위해 diffusion model 출력을 3D 표현으로 distillation했다. 이는 기존 방식이 시간적 불안정성을 보이거나 양방향 모델이 단일 pass로 생성할 수 있는 프레임 수에 제한이 있기 때문이다. 본 논문의 autoregressive 모델은 임의 길이의 렌더링을 순차적으로 생성할 수 있으므로 이러한 제약에서 자유롭다.

그러나 3D distillation은 효율성 측면에서 여전히 유용할 수 있는데, 이러한 표현은 렌더링 속도를 몇 배나 향상시켜 주기 때문이다. 기존 방식에서는 뷰 생성과 3D reconstruction을 번갈아 수행하는 점진적 distillation 과정을 사용하므로 상당한 학습 시간이 소요된다. 본 논문에서는 일관된 방식으로 임의 개수의 프레임을 생성할 수 있으므로, 표준 3D reconstruction을 적용하기 전에 원하는 모든 새로운 뷰를 단일 pass로 생성하는 보다 효율적인 접근 방식을 채택했다.

## Experiments
- 구현 디테일
  - GPU: H100 128개
  - batch size: GPU당 1, 총 128
  - optimizer: AdamW
  - learning rate & iteration
    - 양방향 모델 학습: $1 \times 10^{-5}$ & 15,000
    - Causal model 학습: $1 \times 10^{-5}$ & 5,000
    - DMD 학습: generator $2 \times 10^{-6}$, fake score $4 \times 10^{-7}$ & 2,000
  - 3D reconstruction 모델: [3DGUT](https://kimjy99.github.io/논문리뷰/3dgut) + [MCMC](https://kimjy99.github.io/논문리뷰/3dgs-mcmc) densification

### 1. Enhancing In-the-Wild Captures
다음은 Nerfbusters와 DL3DV에서의 아티팩트 제거 결과이다.

<center><img src='{{"/assets/img/arti-fixer/arti-fixer-table1.webp" | relative_url}}' width="90%"></center>
<br>
다음은 Mip-NeRF 360 데이터셋에서의 sparse view reconstruction 결과이다.

<center><img src='{{"/assets/img/arti-fixer/arti-fixer-table2.webp" | relative_url}}' width="100%"></center>
<br>
다음은 각 ArtiFixer 버전의 생성 결과를 비교한 것이다.

<center><img src='{{"/assets/img/arti-fixer/arti-fixer-fig5.webp" | relative_url}}' width="75%"></center>

### 2. Novel Content Generation
다음은 완전히 관찰되지 않은 영역에 대한 생성 결과이다. 

<center><img src='{{"/assets/img/arti-fixer/arti-fixer-table3.webp" | relative_url}}' width="62%"></center>

### 3. Diagnostics
다음은 ablation study 결과이다.

<center><img src='{{"/assets/img/arti-fixer/arti-fixer-table4.webp" | relative_url}}' width="75%"></center>
<br>
다음은 초기 렌더링 조건 없이 레퍼런스 뷰들만 사용하여 모델이 예측한 결과이다.

<center><img src='{{"/assets/img/arti-fixer/arti-fixer-fig6.webp" | relative_url}}' width="75%"></center>
<br>
다음은 텍스트 프롬프트만 사용하여 동영상을 생성한 예시이다.

<center><img src='{{"/assets/img/arti-fixer/arti-fixer-fig7.webp" | relative_url}}' width="80%"></center>
<br>
다음은 inference 속도를 비교한 결과이다.

<center><img src='{{"/assets/img/arti-fixer/arti-fixer-table5.webp" | relative_url}}' width="36%"></center>