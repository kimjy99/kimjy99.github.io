---
title: "[논문리뷰] A Unified Differentiable Boolean Operator with Fuzzy Logic"
last_modified_at: 2026-04-16
categories:
  - 논문리뷰
tags:
  - 3D Vision
  - SIGGRAPH
excerpt: "Fuzzy Boolean 논문 리뷰 (SIGGRAPH 2024)"
use_math: true
classes: wide
---

> SIGGRAPH 2024. [[Paper](https://arxiv.org/abs/2407.10954)] [[Github](https://github.com/HTDerekLiu/fuzzy-boolean)]  
> Hsueh-Ti Derek Liu, Maneesh Agrawala, Cem Yuksel, Tim Omernick, Vinith Misra, Stefano Corazza, Morgan McGuire, Victor Zordan  
> Roblox | Stanford University | University of Utah | McGill University  
> 15 Jul 2024  

<center><img src='{{"/assets/img/fuzzy-boolean/fuzzy-boolean-fig1.webp" | relative_url}}' width="100%"></center>

## Introduction
본 연구에서는 통합된 미분 가능한 boolean 연산자를 개발하고, 이 연산자를 이용하여 각 내부 CSG 노드에 대한 discrete한 boolean 연산 선택을 continuous한 최적화 변수로 변환함으로써 CSG 최적화를 더욱 완화하는 방법을 제시하였다. 기존의 boolean 연산자(최소/최대 연산자)는 SDF에 대해 연산하는 반면, fuzzy boolean 연산자는 결과가 soft occupancy function으로 유지되도록 보장한다. 이러한 soft occupancy function에 대한 fuzzy boolean 연산은 CSG를 ​다양한 shape 모델링에 자연스럽게 일반화한다.

저자들은 사면체 barycentric interpolation을 사용하여 개별 fuzzy boolean 연산들을 결합하는 통합된 fuzzy boolean 연산자를 구성하였다. 본 논문에서는 제안하는 통합 연산자가 미분 가능하고, gradient vanishing 문제를 방지하며, 단조성을 만족하여 gradient 기반 최적화에 특히 적합하다. 제안된 연산자를 CSG 최적화에 적용한 결과, 기존 방법들에 비해 생성된 트리의 정확도가 크게 향상되었다.

<center><img src='{{"/assets/img/fuzzy-boolean/fuzzy-boolean-fig2.webp" | relative_url}}' width="55%"></center>

## Method
### 1. Product Fuzzy Logic
Continuous한 최적화라는 목표에 따라, 저자들은 각각의 fuzzy boolean 연산이 미분 가능하고 입력값에 대해 0이 아닌 gradient를 갖도록 하고자 하였다. Gradient가 0이 되면 energy landscape에 평탄한 구간이 생겨 gradient 기반 최적화가 어려워질 수 있다. 구체적으로 다음과 같이 정의된다.

$$
\begin{aligned}
f_{X \cap Y} = xy, \quad f_{X \cup Y} = x + y - xy, \quad f_{\neg X} = 1 - x
\end{aligned}
$$

($X$와 $Y$는 두 개의 shape, $$x = f_X (p), y = f_Y (p) \in [0, 1]$$는 임의의 점 $p$에서의 occupancy 값)

또한 드 모르간의 법칙을 만족하므로 차집합을 다음과 같이 계산할 수 있다.

$$
\begin{equation}
f_{X \setminus Y} = x - xy, \quad f_{Y \setminus X} = y - xy
\end{equation}
$$

Product fuzzy boolean 함수는 입력 $x$와 $y$에 대해 미분 가능하다. 또한, 다른 fuzzy logic 함수 정의에 비해 gradient vanishing 현상이 훨씬 덜 발생한다. Gradient vanishing은 편미분 $\frac{\partial}{\partial x}$, $\frac{\partial}{\partial y}$가 0이거나 매우 작아질 때 발생한다.

<center><img src='{{"/assets/img/fuzzy-boolean/fuzzy-boolean-fig.webp" | relative_url}}' width="32%"></center>
<br>
위 그림에서 볼 수 있듯이, 괴델의 max 연산자으로 합집합을 정의하면 $\frac{\partial}{\partial y} = 0$이 된다. 반면, product fuzzy boolean으로 합집합을 정의하면 여전히 $x$​와 $y$ 모두에 대해 0이 아닌 기울기를 가지고 있다.

<center><img src='{{"/assets/img/fuzzy-boolean/fuzzy-boolean-fig4.webp" | relative_url}}' width="55%"></center>

### 2. Unifying Boolean Operations
Boolean 연산 유형(교집합, 합집합, 차집합)에 대해 미분 가능한 통합 fuzzy boolean 연산자를 만들기 위해, 본 논문에서는 interpolation 제어 파라미터 $\textbf{c}$를 사용하여 각 함수를 interpolation하는 방식을 제안하였다. 본 논문의 목표는 interpolation 함수가 불필요한 local minima에 도달하지 않도록 파라미터 $\textbf{c}$에 대해 continuous하고 monotonic한 interpolation 체계를 설계하는 것이다.

<center><img src='{{"/assets/img/fuzzy-boolean/fuzzy-boolean-fig5.webp" | relative_url}}' width="50%"></center>
<br>
단순한 해결책은 네 가지 boolean 연산 $$f_{X \cap Y}$$, $$f_{X \cup Y}$$, $$f_{X \setminus Y}$$, $$f_{Y \setminus X}$$ 사이에 bilinear interpolation을 사용하는 것이다. Bilinear interpolation은 매끄럽게 보일 수 있지만, monotonic하지 않은 변화를 보이며 interpolation된 occupancy에 local minima를 생성한다. 이는 bilinear interpolation이 $$f_{X \cup Y}$$, $$f_{Y \setminus X}$$의 평균과 $$f_{X \cap Y}$$, $$f_{X \setminus Y}$$의 평균을 동일하게 만들기 때문이다. 많은 경우 이러한 평균값은 서로 같지 않으므로, 이러한 제약 조건으로 인해 interpolation이 monotonic하지 않게 된다.

대신, 저자들은 사면체 barycentric interpolation을 사용한다. 더 구체적으로 말하자면, 개별 boolean 연산을 사면체의 꼭짓점으로 취급하고, 통합된 boolean 연산자 함수 $$\mathcal{B}_\textbf{c}$$를 그 안에서의 barycentric interpolation으로 정의한다.

$$
\begin{equation}
\mathcal{B}_\textbf{c} (x, y) = (c_1 + c_2) x + (c_1 + c_3) y + (c_0 - c_1 - c_2 - c_3) xy
\end{equation}
$$

여기서 $$\textbf{c} = \{c_0, c_1, c_2, c_3\}$$는 boolean 연산의 유형을 제어하는 파라미터이며, 다음과 같은 barycentric coordinate의 속성을 만족한다.

$$
\begin{equation}
0 \le c_i \le 1 \quad \textrm{and} \quad c_0 + c_1 + c_2 + c_3 = 1
\end{equation}
$$

파라미터 $\textbf{c}$가 one-hot 벡터, 즉 사면체의 꼭짓점에 대한 barycentric coordinate일 경우, product fuzzy boolean 연산자를 정확하게 재현한다.

$$
\begin{aligned}
\mathcal{B}_{1,0,0,0} (x, y) &= xy &= f_{X \cap Y} \\
\mathcal{B}_{0,1,0,0} (x, y) &= x + y - xy &= f_{X \cup Y} \\
\mathcal{B}_{0,0,1,0} (x, y) &= x - xy &= f_{X \setminus Y} \\
\mathcal{B}_{0,0,0,1} (x, y) &= y - xy &= f_{X \setminus Y}
\end{aligned}
$$

통합 연산자는 입력 $x$, $y$와 제어 파라미터 $c_i$ 모두에 대해 미분 가능하다. 또한 사면체의 꼭짓점을 따라 interpolation하기 때문에 꼭짓점에서의 개별 boolean 연산 간에 monotonic interpolation을 제공한다.

<center><img src='{{"/assets/img/fuzzy-boolean/fuzzy-boolean-fig6.webp" | relative_url}}' width="50%"></center>

## Experiments
- 구현 디테일
  - 기본적으로 primitive는 quadric surface를 사용 (파라미터 10개)
  - sigmoid로 quadric function을 soft occupancy function으로 변환 (학습 가능한 sharpness 파라미터 $s$ 사용)

$$
\begin{equation}
q(x, y, z) = q_0 x^2 + q_1 y^2 + q_2 z^2 + q_3 xy + q_4 yz + q_5 zx + q_6 x + q_7 y + q_8 z + q_9 \\
o(x, y, z) = \textrm{sigmoid} (s \times q (x, y, z))
\end{equation}
$$

  - $\textbf{c}$는 softmax로 구현

$$
\begin{equation}
\textbf{c} = \textrm{softmax} (\sin (\omega \tilde{\textbf{c}}) \cdot t), \quad $\tilde{\textbf{c}} \in \mathbb{R}^4$, \; \omega = 10, \; t = 10^3
\end{equation}
$$

### 1. Fuzzy CSG System
다음은 SDF에 대한 smoothed min/max 연산자와의 비교 예시이다.

<center><img src='{{"/assets/img/fuzzy-boolean/fuzzy-boolean-fig7.webp" | relative_url}}' width="50%"></center>
<span style="display: block; margin: 1px 0;"></span>
<center><img src='{{"/assets/img/fuzzy-boolean/fuzzy-boolean-fig8.webp" | relative_url}}' width="50%"></center>
<br>
다음은 각 primitive의 smoothness를 제어한 예시이다.

<center><img src='{{"/assets/img/fuzzy-boolean/fuzzy-boolean-fig9.webp" | relative_url}}' width="50%"></center>

### 2. Single Shape Inverse CSG with Gradient Descent
다음은 2D 예시에 대한 fitting 결과이다. 128개의 primitive에서 시작하여 CSG tree pruning을 통해 13개로 줄였다.

<center><img src='{{"/assets/img/fuzzy-boolean/fuzzy-boolean-fig10.webp" | relative_url}}' width="50%"></center>
<br>
다음은 다양한 종류의 primitive에 대한 fitting 결과이다.

<center><img src='{{"/assets/img/fuzzy-boolean/fuzzy-boolean-fig11.webp" | relative_url}}' width="58%"></center>
<br>
다음은 boolean node를 고정하고 primitive만 최적화한 경우와 비교한 예시이다.

<center><img src='{{"/assets/img/fuzzy-boolean/fuzzy-boolean-fig12.webp" | relative_url}}' width="63%"></center>

### 3. CSG Generative Models
다음은 [CSG-Stump](https://arxiv.org/abs/2108.11305)의 디코더를 통합 boolean 연산자 기반의 디코더로 바꾼 결과이다.

<center><img src='{{"/assets/img/fuzzy-boolean/fuzzy-boolean-fig13.webp" | relative_url}}' width="50%"></center>
<span style="display: block; margin: 1px 0;"></span>
<center><img src='{{"/assets/img/fuzzy-boolean/fuzzy-boolean-table1.webp" | relative_url}}' width="48%"></center>