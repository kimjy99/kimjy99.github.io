---
title: "[논문리뷰] Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust"
last_modified_at: 2023-06-13
categories:
  - 논문리뷰
tags:
  - Diffusion
  - Computer Vision
  - AI
excerpt: "Tree-Ring Watermarks 논문 리뷰"
use_math: true
classes: wide
---

> arXiv 2023. [[Paper](https://arxiv.org/abs/2305.20030)] [[Github](https://github.com/YuxinWenRick/tree-ring-watermark)]  
> Yuxin Wen, John Kirchenbauer, Jonas Geiping, Tom Goldstein  
> University of Maryland  
> 31 May 2023  

## Introduction
Diffusion model의 개발로 인해 이미지 생성 품질이 급증했다. Stable Diffusion과 Midjourney와 같은 최신 text-to-image diffusion model은 무수히 많은 스타일로 다양하고 참신한 이미지를 생성할 수 있다. 이러한 시스템은 범용 이미지 생성 도구로, 악의적인 목적으로 가짜 사건을 사실적으로 묘사할 뿐만 아니라 새로운 예술을 생성할 수 있다.

Text-to-image model의 잠재적 남용은 출력물에 워터마크를 개발하도록 동기를 부여하였다. 워터마크가 있는 이미지는 사람에게는 보이지 않지만 기계 생성으로 표시되는 신호를 포함하는 생성된 이미지이다. 워터마크는 이미지 생성 시스템의 사용을 문서화하여 소셜 미디어, 뉴스, diffusion 플랫폼 자체가 이미지의 출처를 식별하여 피해를 완화하거나 법 기관과 협력할 수 있도록 한다.

디지털 콘텐츠에 대한 워터마킹의 연구 및 적용은 오랜 역사를 가지고 있으며 지난 10년 동안 많은 접근 방식이 고려되었다. 그러나 지금까지 연구에서는 항상 워터마크를 기존 이미지에 각인되는 최소한의 수정으로 개념화했다. 예를 들어 현재 Stable Diffusion에 배치된 워터마크는 생성된 이미지에서 특정 푸리에 주파수를 수정하여 작동한다.

본 논문에서 제안하는 워터마크 접근 방식은 개념적으로 다르다. 이것은 이미지에 사후 수정이 이루어지지 않기 때문에 진정으로 보이지 않는 최초의 워터마크이다. 대신, 생성된 이미지의 분포가 눈에 띄지 않게 수정되고 이 수정된 분포에서 이미지가 그려진다. 이러한 방식으로 실제 샘플에는 고전적인 의미의 워터마크가 없지만 이미지의 알고리즘 분석을 통해 높은 정확도로 워터마크를 감지할 수 있다. 보다 실용적인 관점에서 워터마크는 생성된 장면의 잠재적인 레이아웃에서 약간의 변화로 구체화되며, 이는 사람이 검사하는 다른 랜덤 샘플과 구별할 수 없다.

**Tree-Ring Watermarking**이라고 하는 워터마킹에 대한 이 새로운 접근 방식은 diffusion model의 noise 벡터의 푸리에 공간에 각인된 패턴을 기반으로 하며, 기존 diffusion model API에 쉽게 통합될 수 있고 샘플에 워터마크가 표시되지 않는다. 가장 중요한 것은 Tree-Ring Watermarking이 crop, color jitter, dilation, 뒤집기, 회전, noise와 같은 일반적인 이미지 변환에 대해 기존 방법보다 훨씬 더 강력하다는 것이다. Tree-Ring Watermarking은 구현을 위한 추가 학습이나 fine-tuning이 필요하지 않으며 워터마크는 이미지 생성 모델을 제어하는 당사자만 감지할 수 있다. 저자들은 이미지 품질 점수에 대한 무시할 수 있는 영향, 변환에 대한 높은 견고성, 감지 시 낮은 false-positive 비율, 텍스트 컨디셔닝 유무에 관계없이 임의의 diffusion model의 유용성을 측정하는 여러 테스트에서 워터마크를 검증한다.

## Method
### 1. Threat Model
먼저 본 논문에서 고려한 위협 모델에 대해 간략하게 설명하고 설정을 명확히 한다. 워터마킹의 목표는 품질 저하 없이 이미지를 생성할 수 있도록 하는 동시에 모델 소유자가 주어진 이미지가 자신의 모델에서 생성되었는지 여부를 식별할 수 있도록 하는 것이다. 

한편, 워터마크된 이미지는 많은 이미지 조작 및 수정의 대상이 된다. 일반적인 이미지 조작을 사용하여 감지를 피하기 위해 워터마크를 제거하려는 적으로 이를 형식화하지만, 일반적인 사용에서 워터마크 견고성에도 관심이 있다. 궁극적으로 이 설정은 순차적으로 작동하는 두 개의 에이전트가 있는 위협 모델로 이어진다.

- Model Owner (생성 단계): Diffusion model $\epsilon_\theta$를 소유하고 있으며 워터마킹 알고리즘 $\mathcal{T}$가 포함된 API를 통해 이미지 $x$를 생성할 수 있다. $\mathcal{T}$는 품질이 유지되고 워터마킹이 눈에 보이는 흔적을 남기지 않도록 생성된 분포에 무시할 수 있는 영향을 미쳐야 한다.
- Forger: API를 통해 이미지 $x$를 생성한 다음 $x$를 $x'$으로 변환하는 강력한 데이터 augmentation을 적용하여 $\mathcal{T}$의 감지를 피하려고 한다. 나중에 금지된 목적으로 $x'$를 사용하고 $x'$이 자신의 지적 재산이라고 주장한다.
- Model Owner (감지 단계): $\epsilon_\theta$와 $\mathcal{T}$에 대한 액세스가 주어지면 $x'$이 $\epsilon_\theta$에서 유래했는지 확인하려고 한다. 모델을 컨디셔닝하는 데 사용되는 텍스트, guidance scale, 생성 step 수와 같은 기타 hyperparameter에 대한 지식이 없다. 

### 2. Overview of Tree-Ring Watermarking
<center><img src='{{"/assets/img/tree-ring-watermark/tree-ring-watermark-fig1.PNG" | relative_url}}' width="95%"></center>
<br>
Diffusion model은 Gaussian noise 배열을 깨끗한 이미지로 변환한다. Tree-Ring Watermarking은 푸리에 변환이 중심 근처에 신중하게 구성된 패턴을 포함하도록 초기 noise 배열을 선택한다. 이 패턴을 "key"라고 한다. 이 초기 noise 벡터는 수정 없이 표준 diffusion 파이프라인을 사용하여 이미지로 변환된다. 이미지에서 워터마크를 감지하기 위해 생성에 사용된 원본 noise 배열을 검색하는 프로세스를 사용하여 diffusion model을 반전시킨다. 그런 다음 이 배열을 검사하여 key가 있는지 확인한다.

결과 이미지에서 눈에 띄는 패턴을 유발할 수 있는 Gaussian 배열에 key를 직접 각인하는 대신 시작 noise 벡터의 푸리에 변환에 key를 각인한다. Binary mask $M$을 선택하고 key $k^\ast \in \mathbb{C}^{\vert M \vert}$를 샘플링한다. 따라서 초기 noise 벡터 $x_T \in \mathbb{R}^L$은 푸리에 공간에서 다음과 같이 설명할 수 있다.

$$
\begin{equation}
\mathcal{F} (x_T)_i \sim \begin{cases}
k_i^\ast & \quad \textrm{if } i \in M \\
\mathcal{N} (0, 1) & \quad \textrm{otherwise}
\end{cases}
\end{equation}
$$

저주파수 mode를 중심으로 반지름 $r$이 있는 원형 마스크로 $M$을 선택한다. 이미지 $x'_0$가 주어지면 모델 소유자는 DDIM inversion 프로세스를 통해 대략적인 초기 noise 벡터 $x'_T$를 얻을 수 있다. 

$$
\begin{equation}
x'_T = D_\theta^\dagger (x'_0)
\end{equation}
$$

최종 metric은 반전된 noise 벡터와 워터마크 영역 $M$의 푸리에 공간의 key 사이의 L1 거리로 계산된다.

$$
\begin{equation}
d_\textrm{detection distance} = \frac{1}{\vert M \vert} \sum_{i \in M} \vert k_i^\ast - \mathcal{F} (x'_T)_i \vert
\end{equation}
$$

워터마크가 조정된 임계값 $\tau$ 아래로 떨어지면 워터마크가 감지된다.

위에서 설명한 프로세스는 간단하지만 그 성공은 "key" 패턴의 구성에 크게 좌우된다.

### 3. Constructing a Tree-Ring Key
원본 Gaussian noise 배열의 푸리에 공간에 "key" 패턴을 배치하여 이미지에 워터마크를 표시한다. 패턴은 주기적 신호에 대한 푸리에 변환의 몇 가지 고전적인 속성을 활용할 수 있다.

- 픽셀 공간의 회전(rotation)은 푸리에 공간의 회전에 해당한다.
- 픽셀 공간에서의 변환(translation)은 모든 푸리에 계수에 상수 복소수를 곱한다.
- 픽셀 공간의 확장/압축은 푸리에 공간의 압축/확대에 해당한다.
- 픽셀 공간의 color jitter(채널의 모든 픽셀에 상수 추가)는 zero-frequency Fourier mode의 크기 변경에 해당한다.

많은 고전적인 워터마킹 전략은 푸리에 공간의 워터마킹에 의존하며 유사한 불변성을 이용한다. 본 논문의 워터마크는 diffusion이 일어나기 전에 랜덤 noise 배열에 푸리에 워터마크를 적용함으로써 고전적인 방법에서 벗어난다. 흥미롭게도 $x_0$의 픽셀 공간에서 이미지 조작이 수행되는 경우에도 위의 불변 속성이 $x_T$에서 보존된다.

위의 불변성을 활용하는 것 외에도 선택한 key는 통계적으로 Gaussian noise와 유사해야 한다. Gaussian noise 배열의 푸리에 변환도 Gaussian noise로 분포된다. 이러한 이유로 Gaussian이 아닌 key를 선택하면 diffusion model에 영향을 미치는 분포 이동이 발생할 수 있다.

다음과 같은 세 가지 다른 유형의 key를 고려한다. 

- **Tree-Ring<sub>Zeros</sub>**: 이미지 공간의 회전에 대한 불변성을 유지하기 위해 마스크를 원형 영역으로 선택한다. Key는 0 배열로 선택되어 이동, 자르기 및 확장에 대한 불변성을 생성한다. 이 key는 조작에 대해 불변이지만 가우시안 분포에서 심각하게 벗어나는 대가를 치른다. 또한 여러 key가 모델을 구별하는 데 사용되는 것을 방지한다.
- **Tree-Ring<sub>Rand</sub>**: 가우시안 분포에서 고정 key $k^\ast$를 그린다. Key는 noise 배열의 원래 푸리에 모드와 동일한 가우시안 특성을 가지므로 이 전략이 생성 품질에 미치는 영향이 가장 적을 것으로 예상된다. 이 방법은 또한 모델 소유자가 여러 key를 소유할 수 있는 유연성을 제공한다. 그러나 이미지 조작은 불변하지 않다.
- **Tree-Ring<sub>Rings</sub>**: 여러 링으로 구성된 패턴과 각 링을 따라 일정한 값을 도입한다. 이렇게 하면 워터마크가 회전에 영향을 받지 않는다. 가우시안 분포에서 상수 링 값을 선택한다. 이는 여러 유형의 이미지 변환에 약간의 불변성을 제공하는 동시에 전체 분포가 가우시안 분포에서 최소한으로만 이동되도록 한다.

## Experiments
- Metric
  - **AUC**: ROC curve의 곡선 아래 영역
  - **TPR@1%FPR**: False Positive Rate가 1%일 때의 True Positive Rate
  - **FID**: 이미지 품질
  - **CLIP score**: 생성된 이미지와 프롬프트 사이의 관계를 측정

### 1. Benchmarking Watermark Accuracy and Image Quality

<center><img src='{{"/assets/img/tree-ring-watermark/tree-ring-watermark-table1.PNG" | relative_url}}' width="75%"></center>

### 2. Benchmarking Watermark Robustness

<center><img src='{{"/assets/img/tree-ring-watermark/tree-ring-watermark-table2.PNG" | relative_url}}' width="80%"></center>

### 3. Ablation Experiments
다음은 생성과 감지 step 수에 따른 AUC를 측정한 그래프이다.

<center><img src='{{"/assets/img/tree-ring-watermark/tree-ring-watermark-fig3.PNG" | relative_url}}' width="67%"></center>
<br>
다음은 Watermark 반지름과 guidance scale에 대한 ablation 결과이다.

<center><img src='{{"/assets/img/tree-ring-watermark/tree-ring-watermark-fig4.PNG" | relative_url}}' width="95%"></center>
<br>
다음은 다양한 공격 강도에 대한 ablation 결과이다. 

<center><img src='{{"/assets/img/tree-ring-watermark/tree-ring-watermark-fig5.PNG" | relative_url}}' width="100%"></center>

### 4. Qualitative comparison
다음은 동일한 random seed에서 워터마킹하여 생성한 이미지이다. 

<center><img src='{{"/assets/img/tree-ring-watermark/tree-ring-watermark-fig2.PNG" | relative_url}}' width="90%"></center>
<br>
Tree-ring은 다른 워터마킹 방법들과 다르게 워터마크가 보이지 않는 특성을 가진다. 