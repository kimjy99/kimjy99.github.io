---
title: "[논문리뷰] A Machine Learning Approach That Beats Large Rubik’s Cubes"
last_modified_at: 2026-07-31
categories:
  - 논문리뷰
tags:
  - NeurIPS
  - Reinforcement Learning
excerpt: "A Machine Learning Approach That Beats Large Rubik’s Cubes 논문 리뷰 (NeurIPS 2025 Spotlight)"
use_math: true
classes: wide
---

> NeurIPS 2025 (Spotlight). [[Paper](https://arxiv.org/abs/2502.13266)]  
> Alexander Chervov, Kirill Khoruzhii, Nikita Bukhal, Jalal Naghiyev, Vladislav Zamkovoy, Ivan Koltsov, Lyudmila Cheldieva, Arsenii Sychev, Arsenii Lenin, Mark Obozov, Egor Urvanov, Alexey Romanov  
> Institut Curie | Technical University of Munich | Novosibirsk State University | RTU MIREA | Innopolis University  
> 18 Feb 2025  

## Introduction
루빅스 큐브를 푸는 것은 planning 문제의 구체적인 사례이다. 초기 state와 해결된 state 사이를 전환하기 위한 action들을 계획해야 한다. 이러한 문제에 대한 수학적 틀은 그래프 상의 경로 탐색이다. 모든 가능한 state는 노드로 표현되고, edge는 action에 기반한 state 간의 전환을 나타낸다. 따라서 planning task는 주어진 초기 노드에서 하나 이상의 원하는 노드까지의 경로를 찾는 것으로 귀결된다. 루빅스 큐브는 퍼즐의 대칭군(symmetry group)에 대한 케일리 그래프(Cayley graph)로 표현된다. 이러한 그래프는 대칭군이 어떤 노드든 다른 노드로 변환할 수 있는 고도로 대칭적인 state 전이 그래프이다.

일반적인 유한 케일리 그래프에서 최단 경로를 찾는 것은 NP-hard 문제이다. 기존의 최단 경로 탐색 알고리즘들은 막대한 계산 자원을 필요로 하며, 루빅스 큐브와 같은 훨씬 더 큰 규모의 그래프에서는 실용적이지 않다. 또한, 현재로서는 대규모 유한 케일리 그래프에서 최단 경로뿐 아니라 모든 경로를 찾을 수 있는 효과적인 도구가 없다. 

본 논문은 머신러닝을 활용하여 루빅스 큐브 및 이와 유사한 대규모 경로 탐색 문제를 높은 최적성으로 해결하는 데 있어 앞서 언급한 한계를 극복하는 것을 목표로 한다. 구체적으로, 유한군(finite group)으로 이루어진 케일리 그래프에서 경로를 찾는 새로운 멀티 에이전트 기반의 머신러닝 접근법을 제안하였다. 이는 최대 $10^{74}$개의 요소로 이루어진 대규모 군을 처리할 수 있는 최초의 머신러닝 접근법이다. 3×3×3 큐브로 구성된 DeepCubeA 데이터셋에서 98% 이상의 최적성을 달성하여 패턴 데이터베이스 기반의 솔버 수준에 도달했다. 또한, 4×4×4, 5×5×5 루빅스 큐브에서도 현재까지 알려진 모든 알고리즘보다 더 짧은 해 경로를 보여주었다.

본 논문에서는 학습에 사용되는 데이터셋의 크기를 늘려도 성능에 미치는 영향은 제한적임을 보여주었다. 동시에, beam 폭과 에이전트 수를 늘리면 평균 해의 길이와 최적성이 크게 향상된다. 이러한 발견은 각 에이전트에 적합한 학습 데이터 크기를 선택하는 데 도움을 주어, 추가 학습에 계산 자원을 낭비하지 않고도 최고 수준의 성능을 달성할 수 있게 해준다. 또한, 이전 최고의 머신러닝 솔루션과 유사한 길이를 제공하면서도 25.6배 빠르며, 학습 시간은 최대 8.5배 적게 소요하였다.

## Method
본 논문은 다양한 그래프에서 경로를 찾는 통합적인 접근 방식을 제시하고, 특히 루빅스 큐브 그래프에 대한 효율성을 입증하는 데 중점을 두었다. 이 접근 방식은 그래프에 대한 사전 지식이나 인간의 전문 지식에 의존하지 않는다. 주요 구성 요소는 신경망 모델과 그래프 탐색 알고리즘이다. 신경망 모델은 퍼즐의 최종 목표 지점에 도달하기 위해 어떤 움직임을 취해야 하는지 예측하도록 학습된다. 그래프 탐색 알고리즘은 주어진 노드에서 시작하여 신경망의 예측을 기반으로 목표 지점에 더 가까운 노드들을 탐색하며, 최종 목표 지점에 도달할 때까지 진행한다.

그래프에 대한 기본적인 가정은 각 노드에 feature 벡터가 연결되어 있다는 것이다. Feature 벡터는 신경망의 입력으로 사용된다. 제안된 방법의 성공적인 작동을 보장하는 feature 벡터의 요구 사항을 정확하게 정량화하는 것은 어려운 과제이다. 극단적인 경우로, 학습 데이터가 모든 노드를 포함하는 경우라면 무작위 벡터로도 충분할 수 있다.

그러나 본 논문의 초점은 다르다. 학습 데이터는 소수의 노드만을 포함하는 경우에 해당한다. 핵심은 신경망이 이러한 작은 부분 집합에서 전체 그래프로 일반화할 수 있는 능력이다. 이는 랜덤 feature로는 불가능하다. 또한, feature 벡터는 그래프에서 노드 간의 거리에 관련되어 있으므로 더 많은 학습 데이터가 필요하다. 신경망의 역할은 초기 feature 벡터를 latent 표현으로 변환하는 것이다. 이때 그래프 상에서 가까운 노드일수록 latent space에서도 가깝다. 퍼즐이나 순열 그룹의 경우, feature 벡터는 $l$개 원소의 순열 $p$를 나타내는 벡터, 즉 숫자 벡터 $(p(0), \ldots, p(l−1))$이다. 또한, 그래프 상의 특정 노드, 예를 들어 퍼즐의 경우 해결된 state가 선택되어 있다고 가정한다. 목표는 주어진 노드에서 선택된 노드까지의 경로를 찾는 것이다. 그래프 크기가 $10^{40}$을 초과할 수 있으므로 표준 경로 탐색 방법은 적용할 수 없다.

<center><img src='{{"/assets/img/rubiks-cube/rubiks-cube-fig1a.webp" | relative_url}}' width="50%"></center>
<br>
제안된 방법의 주요 단계는 다음과 같다.

##### Random walk를 이용한 학습 데이터셋 생성
선택된 노드에서 시작하여 $N$개의 random walk 궤적을 생성한다. Random walk는 현재 노드의 임의의 이웃 노드를 선택하고 이 과정을 여러 단계에 걸쳐 반복적으로 수행하여 간단하게 생성할 수 있다. 각 random walk 궤적은 최대 $$K_\textrm{max}$$ 단계로 구성된다. Random walk 중에 만나는 일부 노드에 대해 $(v, k)$ 쌍을 저장한다. 여기서 $v$는 해당 노드에 해당하는 벡터이고 $k$는 random walk를 통해 해당 노드에 도달하는 데 필요한 단계 수이다. 이 데이터셋은 학습 데이터로 사용된다.

루빅스 큐브의 경우, random walk는 랜덤 스크램블링에 해당한다. 해결된 state에서 시작하여 일련의 랜덤 스크램블링을 수행하고 결과 위치와 수행된 스크램블링 횟수를 기록한다. 개념적으로, $N \rightarrow \infty$에서 $k$의 평균값은 diffusion distance를 측정한다. Random walk 생성은 계산 비용이 매우 저렴하므로 학습 과정 중에 직접 생성할 수 있다.

##### 신경망 학습
생성된 쌍 $(v, k)$ 집합은 신경망의 학습 세트로 사용된다. 구체적으로, $v$는 신경망의 입력 역할을 하고, k는 신경망이 예측해야 하는 출력을 나타낸다. 따라서, 주어진 노드 $v$에 대한 신경망의 예측은 $v$에서 선택된 목적지 노드까지의 거리를 추정한다. 여러 개의 residual block과 batch normalization을 사용하는 MLP 아키텍처를 사용하며, 이를 ResMLP라고 부른다. 모든 모델은 퍼즐 해결 단계 이전에 미리 학습된다.

<center><img src='{{"/assets/img/rubiks-cube/rubiks-cube-fig1b.webp" | relative_url}}' width="50%"></center>

##### 신경망 기반 그래프 탐색 (Beam search)
이 단계에서는 주어진 노드에서 목적지 노드까지의 경로를 찾는다. 신경망은 다음 단계를 어디에서 수행해야 하는지에 대한 휴리스틱을 제공하고, 그래프 경로 탐색 기법은 신경망 예측의 오차를 보정한다. Beam search 경로 탐색 방법은 매우 간단하지만 가장 효과적인 것으로 입증되었으며 다음과 같이 작동한다.

<center><img src='{{"/assets/img/rubiks-cube/rubiks-cube-fig1c.webp" | relative_url}}' width="37%"></center>
<br>
Beam 크기 $W$를 설정한다. 주어진 노드에서 시작하여 모든 이웃 노드를 선택하고 각 이웃 노드에 대한 신경망 예측을 계산한다. 그런 다음 신경망 예측에 따라 목적지에 가장 가까운 $W$개의 노드, 즉 예측값이 작은 노드를 선택한다. 선택된 $W$개 노드의 이웃 노드를 선택하고 중복을 제거한 후 다시 신경망 예측을 계산하여 최소 예측값을 가진 상위 $W$개 노드를 선택한다. 목적지 노드를 찾을 때까지 또는 최대 step을 초과할 때까지 탐색을 반복한다.

##### 멀티 에이전트
학습 세트 생성에 random walk를 사용하므로, 무작위성 때문에 매번 실행할 때마다 새로운 학습 세트가 생성되고, 결과적으로 각 신경망은 diffusion distance를 다르게 근사한다. 이러한 다양성은 일반적으로 실행할 때마다 새로운 해 경로를 생성할 만큼 충분히 크다. 따라서 일반적으로 여러 번 반복 실행하면 한 번 실행했을 때보다 더 짧은 경로를 발견할 수 있다. 각 학습된 신경망을 에이전트라고 부른다. 주어진 state를 해결하기 위해 모든 에이전트를 사용하여 문제를 해결한 다음, 가장 짧은 해 경로를 선택한다.

## Experiments
### 1. Parameters
다음은 학습 세트 크기, 모델 크기, 모델 깊이에 대한 평균 해법 길이를 비교한 결과이다.

<center><img src='{{"/assets/img/rubiks-cube/rubiks-cube-fig2a.webp" | relative_url}}' width="60%"></center>
<br>
다음은 beam 크기와 학습 세트 크기에 대한 평균 해법 길이를 비교한 결과이다.

<center><img src='{{"/assets/img/rubiks-cube/rubiks-cube-fig2b.webp" | relative_url}}' width="61%"></center>
<br>
다음은 에이전트 수에 따른 평균 해법 길이를 비교한 결과이다.

<center><img src='{{"/assets/img/rubiks-cube/rubiks-cube-fig3a.webp" | relative_url}}' width="100%"></center>
<br>
다음은 해법 길이의 분포를 나타낸 그래프이다.

<center><img src='{{"/assets/img/rubiks-cube/rubiks-cube-fig3b.webp" | relative_url}}' width="100%"></center>

### 2. Comparison
다음은 다른 루빅스 큐브 솔버와 비교한 결과이다. ($A$는 에이전트 수, $W$는 beam 크기, $P$는 모델 파라미터 수, $T$는 학습 세트 크기)

<center><img src='{{"/assets/img/rubiks-cube/rubiks-cube-table1.webp" | relative_url}}' width="70%"></center>

## Limitations
1. 루빅스 큐브 외에 다른 task에 직접 적용하려면 추가적인 이론적 분석이 필요하다.
2. 4×4×4, 5×5×5 루빅스 큐브에 대한 최적의 솔버는 아직 존재하지 않다. 따라서, 저자들은 사용 가능한 참조 데이터셋에 한정했다.