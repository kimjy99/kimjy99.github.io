---
title: "[논문리뷰] Flex-Forcing: Towards a Unified Autoregressive and Bidirectional Video Diffusion Model"
last_modified_at: 2026-09-07
categories:
  - 논문리뷰
tags:
  - Computer Vision
  - Diffusion
  - Video Generation
  - NVIDIA
  - ICML
excerpt: "Flex-Forcing 논문 리뷰 (ICML 2026 Spotlight)"
use_math: true
classes: wide
---

> ICML 2026 (Spotlight). [[Paper](https://arxiv.org/abs/2607.03509)] [[Page](https://research.nvidia.com/labs/genair/flex-forcing/)]  
> Xinyin Ma, Julius Berner, Chao Liu, Arash Vahdat, Weili Nie, Xinchao Wang  
> National University of Singapore | NVIDIA  
> 3 Jul 2026  

<center><img src='{{"/assets/img/flex-forcing/flex-forcing-fig1.webp" | relative_url}}' width="60%"></center>

## Introduction
기존 동영상 생성 방식에는 크게 두 가지 패러다임이 있다. 하나는 사전 학습된 video diffusion model에 널리 사용되는 양방향 diffusion 방식이고, 다른 하나는 최근 효율적인 대안으로 부상한 autoregressive 방식이다. 이 두 방식은 각각 상호 보완적인 장점을 제공한다.

양방향 diffusion 방식은 전체 컨텍스트 attention을 활용하여 모든 프레임을 동시에 모델링함으로써 강력한 시간적 일관성과 높은 시각적 충실도를 구현한다. 따라서 카메라 움직임, 장면 전환, 복잡한 상호작용과 같은 장거리 의존성을 포착하는 데 적합하다. 그러나 상당한 계산 오버헤드를 수반하여 inference 비용이 높아지므로 실시간 환경 및 긴 동영상 생성에 대한 확장성이 제한적이다.

반면, autoregressive 방식은 인과적 조건 하에서 작동하며, KV 캐싱을 통해 과거의 key–value를 재사용하면서 프레임을 순차적으로 생성한다. 이를 통해 실시간 추론이 가능하고, 이전 프레임을 재처리하지 않고도 임의 길이의 동영상 생성을 자연스럽게 지원한다. 그럼에도 불구하고, 생성 과정에서 글로벌 컨텍스트가 부족하기 때문에 시간이 지남에 따라 오차가 누적되기 쉬우며, 이로 인해 물체의 외형, 움직임, 장면 구조가 변할 뿐만 아니라 글로벌한 시간적 일관성이 약화되고 장기적인 계획 능력이 제한된다.

본 논문에서는 두 가지 패러다임을 단일 모델에 통합하여, exposure bias나 계산 비용을 증가시키지 않으면서 효율적인 긴 동영상 inference를 지원하는 동시에 일관된 동영상 생성을 가능하게 하고자 하였다. 이를 위해, test-time에 inference 패러다임을 제어할 수 있는 통합 프레임워크인 **Flex-Forcing**을 제안하였다. 하나의 모델로 3가지 모드에서 유연하게 작동할 수 있다.

1. 글로벌하게 일관된 생성을 위한 양방향 모드
2. 긴 동영상 또는 스트리밍 합성을 위한 autoregressive 모드
3. 품질과 효율성 간의 중간 균형을 제공하는 semi-autoregressive 하이브리드 모드

이러한 유연한 inference를 구현하기 위한 핵심 과제는 하나의 모델에 인과적(autoregressive) 및 비인과적(양방향) 생성 기능을 모두 갖추는 것이다. 이를 통해 모델은 엄격한 인과적 inference와 완전한 양방향 inference 사이의 중간 영역에서 작동할 수 있다.

본 논문에서는 두 개의 직교 축, 즉 시간 프레임과 denoising step을 따라 정의되는 Flexible Chunking을 도입하여 이 문제를 해결하였다. 이 축 하에서 autoregressive inference와 양방향 inference는 두 가지 극단적인 경우로 나타난다. 이 공식은 학습과 inference 모두에 일관되게 적용되어 모델이 혼합된 인과적 및 비인과적 조건 컨텍스트에 노출된다. 동일한 query 토큰은 서로 다른 noise level의 key-value 쌍에 attention될 수 있으며, 일반적으로 인과적인 과거 토큰은 noise가 더 많이 제거된 반면, 비인과적인 미래 토큰은 noise가 더 많다. 이러한 noise level의 차이를 해결하기 위해 인과적 및 비인과적 attention 컨텍스트 간의 표현 일관성을 명시적으로 강제하는 정렬 메커니즘을 도입하였다.

이러한 유연한 inference 모드는 뛰어난 품질-효율성 균형을 달성하여 Pareto frontier를 크게 향상시키고, 짧은 동영상 벤치마크와 긴 동영상 벤치마크 모두에서 [Self-Forcing](https://kimjy99.github.io/논문리뷰/self-forcing)보다 훨씬 우수한 성능을 보여주었다. 이러한 유연한 패러다임은 동영상 생성뿐 아니라, 순서나 step에 관계없이 autoregressive 편집을 지원하는 적응형 인과 제약 조건을 갖는 다운스트림 애플리케이션에도 적용될 수 있다. 이를 통해 원본 동영상과의 전체적인 일관성을 유지하면서 부분적인 시간적 편집도 가능하다.

## Flex-Forcing
### 1. Flexible Chunking over Video Frames
동영상을 $F$개의 프레임 시퀀스 $x = (x^{(1)}, x^{(2)}, \ldots, x^{(F)})$로 표현하자. Diffusion 샘플링은 $$\{x_t\}_{t=1}^T$$인 T번의 denoising step을 수행한다. 청크는 연속적인 프레임 시퀀스로 구성되며, 이러한 청크의 크기는 유연하게 조절할 수 있다. 이러한 방식으로, 청크 내 의존성은 양방향으로 모델링되는 반면 청크 간 생성은 autoregressive한 하이브리드 인수분해를 얻는다. 파티션은 denoising step에 따라 달라질 수 있으므로, 인수분해는 생성 과정의 불확실성에 적응할 수 있다.

구체적으로, 각 denoising step $t$에서 프레임 인덱스 $$\{1, \ldots, F\}$$의 연속적인 파티션을 정의한다. 청크 경계 인덱스 $$\textbf{a}_t = (a_{t,0}, a_{t,1}, \ldots, a_{t, K_t})$$에 따라 구분되며, $$1 = a_{t,0} < \cdots < a_{t, K_t} = F + 1$$이다. Timestep $t$에서의 $k$번째 청크는 프레임 인덱스 집합 $$\mathcal{F}_{t,k}$$를 포함한다.

$$
\begin{equation}
\mathcal{F}_{t,k} = \{f \mid a_{t, k-1} \le f < a_{t,k}\}
\end{equation}
$$

각 denoising step $t$에서, 주어진 파티션 $$\textbf{a}_t$$에 대해 chunk-wise causal sampling은 다음과 같이 정의된다.

$$
\begin{equation}
x_{t-1}^{\mathcal{F}_{t,k}} \sim q_\theta \left( x_{t-1}^{\mathcal{F}_{t,k}} \mid x_0^{\mathcal{F}_{t, <k}}, x_t^{\mathcal{F}_{t,k}}; \textbf{a}_t \right)
\end{equation}
$$

($$\mathcal{F}_{t, <k} = \bigcup_{u < k} \mathcal{F}_{t, u}$$는 현재 청크 이전의 모든 프레임 인덱스, $$x^{\mathcal{F}}$$는 청크 $\mathcal{F}$의 프레임)

[Self-Forcing](https://kimjy99.github.io/논문리뷰/self-forcing)을 따라 $$x_0^{\mathcal{F}_{t, <k}}$$는 DiT에서 이전에 예측된 프레임들의 KV 캐시로 표현된다.

위 식은 이전 청크를 조건으로 하는 denoising을 통해 이후 청크가 생성됨을 뜻하며, 동영상의 프레임 축을 따라 유연한 chunking 전략을 의미한다. 프레임 선택 파라미터 $$\textbf{a}_t$$를 설정함으로써, 본 프레임워크는 인과적 제약 조건 하에서 히스토리로 활용될 수 있는 가변적인 프레임 부분집합과 양방향 attention에서 상호작용하는 컨텍스트를 정의한다.

<center><img src='{{"/assets/img/flex-forcing/flex-forcing-table1.webp" | relative_url}}' width="60%"></center>

### 2. Flexible Chunking over Denoising Timesteps
$t$에 따라 달라지는 청크 분할은 denoising timestep에 따라 chunking 전략을 변경할 수 있도록 한다. 직관적으로, noise level이 높은 초기 denoising step에서는 글로벌 구조에 초점을 맞추고 더 큰 청크를 사용하는 것이 유리하다. 반대로, 후반 step에서는 로컬 디테일을 우선시하며, 이는 긴 컨텍스트가 덜 필요한 더 작은 청크로 효과적으로 모델링할 수 있다. 결과적으로 noise level이 낮아짐에 따라 청크 크기가 감소하는 계층적인 피라미드형 구조가 생성된다.

<center><img src='{{"/assets/img/flex-forcing/flex-forcing-fig3.webp" | relative_url}}' width="100%"></center>
<br>
$$\textbf{a}_t = (a_{t,0}, a_{t,1}, \ldots, a_{t, K_t})$$는 denoising timestep $t$에서의 프레임 분할을 나타낸다. Denoising이 진행됨에 따라 각 청크를 추가로 세분화할 수 있도록 중첩된 유연성을 적용한다. 구체적으로, timestep $t+1$에서 denoising이 완료된 후 ($x_{t+1} \rightarrow x_t$), 기존의 모든 경계를 유지하면서 추가 경계를 삽입하여 $t$에 대한 분할을 나눈다. Timestep $t$는 $t+1$과 동일한 구성을 상속받는다.

$$
\begin{equation}
\mathcal{F}_{t,k} = [a_{t,k-1}, a_{t,k}) = [a_{t+1,k-1}, a_{t+1,k})
\end{equation}
$$

그리고 $n_{t,k}$개의 새로운 분할점 시퀀스 $$\mathcal{S}_{t,k}$$를 도입한다. 원래 끝점과 분할점을 병합하면 새로운 chunking 패턴이 생성된다.

$$
\begin{equation}
\textbf{a}_{t,k}^\prime = a_{t,k-1} \cup \mathcal{S}_{t,k} \cup a_{t,k}
\end{equation}
$$

이러한 분리 가능성은 재귀적으로 적용될 수 있으며, 결과로 생성된 sub-chunk는 다음 denoising step에서 추가 경계를 삽입하여 더 세분화할 수 있다.

Inference 시에는 프레임은 시간 순서대로 처리된다. 그러나 피라미드형 chunking 전략에서는 step $t$에서 큰 청크가 step $t-1$에서 더 작은 sub-chunk로 분할될 때 동기화 문제가 발생한다. 이러한 경우, step $t$의 원래 양방향 청크 내의 모든 프레임에 대한 denoising 결과를 일시적으로 버퍼링한다. 그런 다음, 이전 sub-chunk에서 필요한 KV 캐시가 사용 가능해지면 step $t-1$에서 각 sub-chunk에 대한 denoising을 autoregressive하게 재개한다. 이러한 실행 순서는 다양한 세분성에서 인과 관계가 충족되도록 보장한다.

### 3. Flexible-chunk Training
학습은 두 부분으로 구성된다.

1. 미래 청크에 attention하는 비인과성을 유지하면서 양방향 diffusion model에 인과성 제약을 도입.
2. 모델이 다양한 noise level의 입력에서 파생된 key state에 attention할 수 있도록 함.

##### 비인과성을 유지하면서 인과성을 주입하는 방법
본 논문에서는 양방향 diffusion model을 인과적 모델로 변환하는 [CausVid](https://kimjy99.github.io/논문리뷰/causvid)와 [Self-Forcing](https://kimjy99.github.io/논문리뷰/self-forcing)에서 소개된 학습 패러다임을 주로 따른다. 이 학습 파이프라인은 두 단계로 구성된다.

1. ODE 초기화 단계에서는 causal attention mask를 적용하여 모델에 인과성을 주입한다.
2. DMD 학습을 통한 비대칭 distillation 단계에서는 self-rollout을 통해 인과성을 더욱 강화하고 VSD loss를 적용하여 multi-step 모델을 few-step 모델로 distillation한다.

비대칭 distillation 단계에서는 각 rollout 내에서 attention 패턴을 동적으로 변화시키는 stochastic chunking 전략을 도입한다. 프레임 인덱스 집합을 경계 설정을 통해 연속적으로 분할하고, 학습 중에 청크 분할 $$\textbf{a}_t$$를 무작위로 샘플링한 다음, 이러한 유연한 청크에 동적으로 rollout을 적용한다. 모든 프레임 $$\textbf{x}_0 = \{x0^i\}_{i=1}^F$$의 latent들이 각 rollout에 대해 모두 적용된 후, 다음과 같이 gradient를 취하여 generator $$G_\theta$$를 학습시킨다.

$$
\begin{equation}
\nabla_\theta D_\textrm{KL} = \mathbb{E}_{\textbf{z} \sim \mathcal{N}(0, \textbf{I})} \left[ - (s_\textrm{real} (\textbf{x}_t) - s_\textrm{fake} (\textbf{x}_t)) \frac{\partial G_\theta}{\partial \theta} \right] \\
\textrm{where} \quad \textbf{x}_t = I (G_\theta (\textbf{z}), t)
\end{equation}
$$

($$s_\textrm{real}$$과 $$s_\textrm{fake}$$는 각각 실제 score와 가짜 score, $$I(\textbf{x}_0, t)$$는 forward process)

청크 크기를 무작위로 샘플링함으로써, 모델은 엄격한 인과성부터 완전한 비인과성에 이르기까지 다양한 attention 구성에 노출된다. 결과적으로, 학습에는 인과적 의존성과 양방향 의존성이 암묵적으로 혼합되어 포함된다.

##### 인과적 및 비인과적 attention 전반에 걸친 noise level 정렬
본 모델은 인과적 inference와 양방향 inference를 모두 지원하므로, 동일한 query 토큰이 서로 다른 생성 방식에서 생성된 컨텍스트에 attention될 수 있다. 주어진 self-attention layer에서, 이는 query가 noise가 더 적은 표현에서 파생된 key-value state뿐만 아니라 (인과적 과거 토큰), noise가 더 많은 latent로부터 생성된 state에도 (비인과적 미래 토큰) attention될 수 있음을 의미한다. 표준 self-attention은 모든 key-value 쌍을 균일하게 처리하므로 컨텍스트 집계에서 noise level 불일치가 발생하고 유연한 inference 환경에서 성능이 저하된다.

이 문제를 해결하기 위해, 저자들은 key state에 대한 noise 정렬 projection인 **K-projection**을 제안하였다. 구체적으로, 이는 깨끗한 입력으로부터 생성된 key state 캐시를 현재 diffusion timestep에 해당하는 noisy latent space로 projection한다. Projection 후, 모든 key state는 noise 일관성을 갖는 표현 공간으로 표현되므로 표준 self-attention을 적용할 수 있다.

$$
\begin{equation}
\Pi_{t \leftarrow 0} : \mathbb{R}^d \rightarrow \mathbb{R}^d, \quad \tilde{K}_t = \Pi_{t \leftarrow 0} (K_0)
\end{equation}
$$

($$\Pi_{t \leftarrow 0}$$는 timestep에 따라 달라지는 linear projection이며, identity mapping으로 초기화)

각 denoising step $t$에 대해 self-attention 계산은 다음과 같이 수행된다.

$$
\begin{equation}
\textrm{Attn}(Q_t, \tilde{K}_t, V_t) = \textrm{softmax} \left( \frac{Q_t \tilde{K}_t^\top}{\sqrt{d}} \right) V_t \\
\textrm{where} \quad \tilde{K}_t = \textrm{concat} \left( \Pi_{t \leftarrow 0} \left( K_0^{\mathcal{F}_{t, <k}} \right), \tilde{K}_t^{\mathcal{F}_{t,k}} \right)
\end{equation}
$$

Projection은 attention 연산 중에 즉시 적용되며, 캐싱된 KV 텐서나 KV 캐시의 gradient propogation을 수정하지 않는다. Inference 시에는 깨끗한 KV state가 한 번 저장되고 timestep에 따라 동적으로 projection된다. 이러한 설계는 KV 캐싱의 효율성 이점을 유지하면서 인과적 및 비인과적 상황 모두에서 안정적이고 유연한 inference를 가능하게 한다.

## Applications of Flex-Forcing
### 1. Inference Flexibility: Better Speed, Better Quality
Flex-Forcing은 가변적인 청크 크기를 허용함으로써 Flex-Forcing은 다양한 컴퓨팅 예산과 다양한 길이의 동영상에 맞춰 조정할 수 있는 적응형 청크 구성을 지원하며, 품질과 효율성 간의 더욱 유리한 trade-off를 제공한다.

저자들은 5초 분량의 동영상에 대해 3개의 청크로 구성된 최적의 청크 구성을 찾기 위해 brute-force search를 수행했다. 구체적으로, 21개의 latent 프레임을 3개의 청크로 분할하는 모든 유효한 경우를 테스트했으며, 청크에 프레임이 하나만 포함된 경우는 제외했다.

<center><img src='{{"/assets/img/flex-forcing/flex-forcing-fig4.webp" | relative_url}}' width="60%"></center>
<br>
결과는 위 그림과 같으며, 이를 통해 다음과 같은 주요 관찰 결과를 도출했다.

- 프레임을 균등하게 분할하면 비대칭적인 chunking에 비해 성능이 현저히 떨어질 수 있다.
- 청크 레이아웃은 exposure bias 외에도 중요하다. 동일한 exposure 라운드 수에서 청크 구성에 따라 성능 차이가 크게 나타난다. 
- 초기 프레임에 큰 청크를, 후기 프레임에 작은 청크를 할당하면 가장 좋은 결과를 얻을 수 있으며, 때로는 양방향 inference보다 우수한 성능을 보이기도 한다.

### 2. Autoregressive Any-timestep, Any-order Editing
Flex-Forcing을 통해 기존의 autoregressive 패러다임으로는 불가능했던 두 가지 새로운 형태의 편집 방식을 구현할 수 있다.

##### Autoregressive any-timestep editing
저자들은 글로벌 일관성을 유지하기 위해, low-level timestep에만 편집을 제한하는 구조 보존 편집 전략을 채택하였다. 이러한 분리는 글로벌 구조 형성과 로컬 디테일 수정을 명시적으로 분리하여 기존의 autoregressive 편집 방법보다 일관성을 크게 향상시킨다. Self-Forcing에서는 글로벌하게 결합된 생성 역학으로 인해 작은 로컬 편집조차도 프레임 간에 전파되어 최종 출력에서 ​​큰 편차를 초래하는 경우가 많다.

<center><img src='{{"/assets/img/flex-forcing/flex-forcing-fig6.webp" | relative_url}}' width="92%"></center>

##### Autoregressive any-order editing
과거 및 미래 토큰 모두에 attention하는 능력을 활용하여, 본 모델은 autoregressive 생성 프레임워크 내에서 작동하면서도 순서에 상관없이 편집을 가능하게 한다. 따라서 대상 청크의 편집은 시간적 순서와 관계없이 수행될 수 있다.

Timestep $t$와 청크 $k$에서의 편집은 다음과 같이 정의된다.

$$
\begin{equation}
x_{t-1}^{\mathcal{F}_{t,k}} \sim q_\theta \left( x_{t-1}^{\mathcal{F}_{t,k}} \mid x_0^{\mathcal{F}_{t, <k}}, x_0^{\mathcal{F}_{t, >k}}, x_t^{\mathcal{F}_{t,k}} \right)
\end{equation}
$$

양쪽 컨디셔닝을 통해 전체 시퀀스를 다시 생성하지 않고도 전체 동영상 생성 후 임의의 시간 세그먼트를 재편집할 수 있으므로 표준 autoregressive 모델의 엄격한 인과 제약을 완화할 수 있다.

<center><img src='{{"/assets/img/flex-forcing/flex-forcing-fig5.webp" | relative_url}}' width="100%"></center>

## Experiments
- base model: Wan2.1-T2V-1.3B
- teacher model: Wan2.1-T2V-14B

### 1. Performance on 5s videos
다음은 VBench 벤치마크에서 5초 동영상 생성 성능을 비교한 결과이다.

<center><img src='{{"/assets/img/flex-forcing/flex-forcing-table2.webp" | relative_url}}' width="68%"></center>
<br>
다음은 few-step distillation된 모델들과의 비교 결과이다.

<center><img src='{{"/assets/img/flex-forcing/flex-forcing-table3.webp" | relative_url}}' width="54%"></center>
<br>
다음은 5초 동영상 생성 예시들이다.

<center><img src='{{"/assets/img/flex-forcing/flex-forcing-fig7.webp" | relative_url}}' width="100%"></center>

### 2. Performance of hybrid chunking over timesteps
다음은 하이브리드 chunking에 대한 성능 비교 결과이다.

<center><img src='{{"/assets/img/flex-forcing/flex-forcing-fig8.webp" | relative_url}}' width="80%"></center>

### 3. Performance on 30s videos
다음은 VBench-Long 벤치마크에서 30초 동영상 생성 성능을 비교한 결과이다.

<center><img src='{{"/assets/img/flex-forcing/flex-forcing-table4.webp" | relative_url}}' width="100%"></center>

### 4. User preference study
다음은 user study 결과이다.

<center><img src='{{"/assets/img/flex-forcing/flex-forcing-fig10a.webp" | relative_url}}' width="36%"></center>

### 5. Ablation study: impact of K-projection
다음은 K-projection에 대한 ablation study 결과이다.

<center><img src='{{"/assets/img/flex-forcing/flex-forcing-fig10b.webp" | relative_url}}' width="40%"></center>