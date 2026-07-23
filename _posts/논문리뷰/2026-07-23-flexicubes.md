---
title: "[논문리뷰] Flexible Isosurface Extraction for Gradient-Based Mesh Optimization"
last_modified_at: 2026-07-23
categories:
  - 논문리뷰
tags:
  - Mesh Optimization
  - 3D Vision
  - SIGGRAPH
  - NVIDIA
excerpt: "FlexiCubes 논문 리뷰 (SIGGRAPH 2023)"
use_math: true
classes: wide
---

> SIGGRAPH 2023. [[Paper](https://arxiv.org/abs/2308.05371)] [[Page](https://research.nvidia.com/labs/toronto-ai/flexicubes/)] [[Github](https://github.com/nv-tlabs/FlexiCubes)]  
> Tianchang Shen, Jacob Munkberg, Jon Hasselgren, Kangxue Yin, Zian Wang, Wenzheng Chen, Zan Gojcic, Sanja Fidler, Nicholas Sharp, Jun Gao  
> NVIDIA | University of Toronto | Vector Institute  
> 10 Aug 2023  

<center><img src='{{"/assets/img/flexicubes/flexicubes-fig1.webp" | relative_url}}' width="100%"></center>

## Introduction
일반적인 메쉬 최적화 방법은 공간에서 occupancy 또는 SDF를 정의하고 최적화한 다음, 해당 함수의 level set을 근사하는 삼각형 메쉬를 추출하는 것이다. 메쉬를 추출할 때 미묘하지만 중요한 문제는 생성 가능한 메쉬의 공간이 제한될 수 있다는 점이다. 삼각형 메쉬를 추출하는 데 사용되는 특정 알고리즘의 선택은 생성된 shape의 속성을 직접적으로 결정한다.

따라서 메쉬 생성 절차는 쉽고 효율적이며 고품질의 최적화를 위해 다음과 같은 두 가지 핵심 속성을 제공해야 한다.

1. **미분 가능성**. 메쉬에 대한 미분은 잘 정의되어 있으며, 실제로 gradient 기반 최적화는 효과적으로 수렴해야 한다.
2. **유연성**. 메쉬 vertex는 표면 특징에 맞게 개별적으로 로컬 조정이 가능하며 적은 수의 요소로 고품질 메쉬를 찾을 수 있어야 한다.

그러나 이 두 가지 속성은 본질적으로 상충된다. 유연성이 증가하면 degenerate geometry와 self-intersection을 표현할 수 있는 능력이 향상되지만, 이는 gradient 기반 최적화에서 수렴을 저해한다. 결과적으로 기존 기법들은 일반적으로 이 두 가지 속성 중 하나를 간과한다.

<center><img src='{{"/assets/img/flexicubes/flexicubes-table1.webp" | relative_url}}' width="75%"></center>
<br>
예를 들어, Marching Cubes는 vertex가 항상 고정된 격자를 따라 위치하기 때문에 유연성이 부족하다. 반면, Dual Contouring은 날카로운 형상을 포착하지만, vertex 위치 지정에 사용되는 선형 시스템은 불안정하고 비효율적인 최적화로 이어진다.

본 논문에서는 두 가지 속성을 모두 만족하는 **FlexiCubes**라는 새로운 기법을 제시하였다. 핵심 아이디어는 특정 Dual Marching Cubes 공식에 추가적인 자유도를 도입하여 추출된 각 vertex들을 dual cell 내에 유연하게 배치하는 것이다. 저자들은 대부분의 경우 self-intersection이 없는 매니폴드 메쉬를 생성하고, 메쉬에 대한 미분이 원활하게 이루어지도록 신중하게 제약 조건을 설정했다.

## Method
FlexiCubes의 핵심은 그리드 상의 스칼라 함수 $s$이며, 이 함수로부터 Dual Marching Cubes 알고리즘을 통해 삼각형 메쉬를 추출한다. 본 논문에서는 메쉬 표현의 유연성을 높이면서도 robustness와 최적화 용이성을 유지하기 위해 신중하게 선택된 세 가지 추가 파라미터 세트를 도입하였다.

- **Interpolation weight** $$\alpha \in \mathbb{R}_{>0}^8$$, $$\beta \in \mathbb{R}_{>0}^{12}$$: Dual vertex를 공간상에 배치하는 데 사용
- **Splitting weight** $$\gamma \in \mathbb{R}_{> 0}$$: 사각형을 삼각형으로 분할하는 방식을 제어
- **Deformation vector** $$\delta \in \mathbb{R}^3$$: 공간 정렬을 위해 사용

이러한 파라미터들은 $s$와 함께 자동 미분을 통해 최적화되어 원하는 목표에 맞는 메쉬를 생성한다.

### 1. Dual Marching Cubes Mesh Extraction
각 grid vertex $x$에서의 스칼라 함수 $s(x)$ 값을 기반으로 Dual Marching Cubes 메쉬의 연결성을 추출하는 것으로 시작한다. 그리드의 모서리를 따라 vertex를 추출하는 일반적인 Marching Cubes와 달리, Dual Marching Cubes는 cell 내의 각 primal face에 대해 하나의 vertex를 추출한다. 인접한 cell에서 추출된 vertex들은 edge로 연결되어 사각형 면으로 구성된 dual mesh를 형성한다. 결과적으로 생성되는 메쉬는 매니폴드임이 보장되지만, 추후 추가될 자유도로 인해 드물게 self-intersection이 발생할 수 있다.

<center><img src='{{"/assets/img/flexicubes/flexicubes-fig5.webp" | relative_url}}' width="60%"></center>

### 2. Flexible Dual Vertex Positioning
본 논문에서는 추출된 메쉬 vertex 위치를 계산하는 방식에 대해 일반적인 Dual Marching Cubes 알고리즘을 일반화하였다. Marching Cubes의 primal vertex는 cell 모서리를 따라 스칼라 값이 0이 되는 지점에 위치한다.

$$
\begin{equation}
u_e = \frac{s(x_b) x_a - s (x_a) x_b}{s(x_b) - s(x_a)}
\end{equation}
$$

일반적인 Dual Marching Cubes는 추출된 각 vertex의 위치를 ​​해당 primal face의 중심점으로 정의한다.

$$
\begin{equation}
v_d = \frac{1}{\vert V_E \vert} \sum_{u_e \in V_E} u_e
\end{equation}
$$

($V_E$는 primal face의 vertex인 교차점들의 집합)

<center><img src='{{"/assets/img/flexicubes/flexicubes-fig6.webp" | relative_url}}' width="62%"></center>
<br>
이 표현에 추가적인 유연성을 도입하기 위해, 먼저 각 cell마다 큐브 꼭짓점에 연결되는 가중치 집합 $$\alpha \in \mathbb{R}_{>0}^8$$를 정의한다. 이 가중치는 각 모서리를 따라 교차점의 위치를 다음과 같이 ​​조정한다.

$$
\begin{equation}
u_e = \frac{s(x_i) \alpha_i x_j - s(x_j) \alpha_j x_i}{s(x_i) \alpha_i - s(x_j) \alpha_j}
\end{equation}
$$

저자들은 $\alpha$의 범위를 $[0, 2]$로 제한하기 위해 $\textrm{tanh}(\cdot) + 1$을 적용하였다.

마찬가지로, dual vertex를 primal face의 중심에 단순히 배치하는 대신, 각 cell마다 큐브 모서리에 연결되는 가중치 집합 $$\beta \in \mathbb{R}_{>0}^{12}$$를 정의한다. 이 가중치는 각 면 내부에서 dual vertex의 위치를 ​​다음과 같이 조정한다.

$$
\begin{equation}
v_d = \frac{1}{\sum_{u_e \in V_E} \beta_e} \sum_{u_e \in V_E} \beta_e u_e
\end{equation}
$$

실제로는 $\alpha$와 유사하게 $\beta$의 범위를 제한하기 위해 $\textrm{tanh}(\cdot) + 1$을 적용한다.

이러한 가중치를 모두 합하면 cell당 20개의 스칼라가 된다. 가중치는 cell별로 독립적으로 정의되며, 인접한 꼭짓점이나 모서리에서 공유되지 않는다. 독립적인 가중치는 더 큰 유연성을 제공하며, 인접한 요소 간에 유지해야 할 연속성 조건이 없다.

두 식은 의도적으로 convex combination으로 설계되었으므로, 결과적으로 추출된 vertex 위치는 해당 cell vertex의 convex hull 내에 반드시 위치하게 된다. 또한, convex cell이 여러 개의 dual vertex를 생성할 때, dual vertex들이 위치하는 해당 primal face들은 서로 교차하지 않으므로, 결과 메쉬에서 거의 모든 self-intersection을 방지할 수 있다.

### 3. Flexible Quad Splitting
Dual Marching Cubes와 FlexiCubes는 평면이 아닌 면을 가진 순수 사각형 메쉬를 추출하는데, 이러한 메쉬는 응용 분야에서의 원할한 사용을 위해 일반적으로 삼각형으로 분할된다. 임의의 대각선을 따라 단순히 분할하면 곡선 영역에서 상당한 아티팩트가 발생할 수 있으며, 일반적으로 모든 geometry를 표현하기 위해 평면이 아닌 사각형을 분할하는 데 이상적인 방법은 없다.

<center><img src='{{"/assets/img/flexicubes/flexicubes-fig8.webp" | relative_url}}' width="87%"></center>
<br>
따라서 저자들은 분할 선택을 유연하게 만들고 연속적인 자유도로 최적화하기 위해 추가 파라미터를 도입하였다. 구체적으로, 각 cell에 가중치 $$\gamma \in \mathbb{R}_{> 0}$$을 정의하고, 이 가중치는 추출된 메쉬의 vertex로 전파된다. 최적화 시점에만 각 사각형 메쉬 면은 중간점 vertex $v_d$를 삽입하여 4개의 삼각형으로 분할된다. 이 중간점의 위치는 다음과 같이 계산된다.

$$
\begin{equation}
\bar{v}_d = \frac{\gamma_{c_1} \gamma_{c_3} (v_d^{c_1} + v_d^{c_3}) / 2 + \gamma_{c_2} \gamma_{c_4} (v_d^{c_2} + v_d^{c_4}) / 2}{\gamma_{c_1} \gamma_{c_3} + \gamma_{c_2} \gamma_{c_4}}
\end{equation}
$$

이는 면의 두 대각선의 중간점들의 가중 조합이며, 가중치는 해당 vertex의 $\gamma$ 파라미터에서 가져온다. 직관적으로, $\gamma$ 가중치를 조정하면 두 가지 가능한 분할에서 생성되는 geometry 사이를 부드럽게 interpolation할 수 있다. $\gamma$를 최적화하면 알맞는 분할을 선택할 수 있다. 최적화가 완료된 후 최종 추출 단계에서는 중간점 꼭짓점 $$\bar{v}_d$$를 삽입하지 않고, $\gamma$ 값의 곱이 더 큰 대각선을 따라 각 사각형을 분할한다.

### 4. Flexible Grid Deformation
[DefTet](https://arxiv.org/abs/2011.01437)와 [DMTet](https://kimjy99.github.io/논문리뷰/dmtet)에서 영감을 받아, 저자들은 기본 그리드의 vertex들이 각 vertex에서 변위 $\delta \in \mathbb{R}^3$에 따라 변형될 수 있도록 했다. 이러한 변형을 통해 그리드는 얇은 부분에 로컬하게 정렬될 수 있으며, vertex 위치 지정에 추가적인 유연성을 제공한다. Cell이 반전되지 않도록 변형의 최댓값은 그리드 간격의 절반으로 제한된다.

### 5. Tetrahedral Mesh Extraction
물리 시뮬레이션이나 캐릭터 애니메이션과 같은 많은 응용 분야에서는 shape 볼륨의 tetrahedralization이 필요하다. 본 논문에서는 FlexiCubes를 확장하여 필요에 따라 추출된 표면의 경계에 정확히 부합하고 표면 추출과 동일한 방식으로 자동 미분 기능을 지원하는 사면체 메쉬를 추가로 출력할 수 있도록 했다.

저자들은 [Octree-based Dual Contouring](https://link.springer.com/article/10.1007/s00366-013-0328-8)의 전략을 적용하였다. 사면체 메쉬의 vertex 집합은 grid vertex, cell에서 추출한 메쉬 vertex, 그리고 표면 vertex가 추출되지 않은 모든 cell의 중간점을 합집합한 것이다.

<center><img src='{{"/assets/img/flexicubes/flexicubes-fig10.webp" | relative_url}}' width="55%"></center>
<br>
사면체는 grid edge를 구성하는 두 grid vertex의 부호에 따라 다음과 같이 총 4개의 사면체를 생성한다.

- **같은 부호**: 두 grid vertex + 인접한 두 cell의 vertex 1개씩 = 사면체 1개.
- **다른 부호**: 하나의 grid vertex + 인접한 네 cell의 vertex 1개씩 = 피라미드 2개. 각 피라미드는 밑면을 기준으로 분할되어 각각 두 개의 사면체를 생성.

Dual Marching Cubes 연결성을 사용할 경우, 하나의 cell에 여러 개의 추출된 메쉬 vertex가 포함될 수 있다는 추가적인 복잡성이 발생하며, 사면체를 생성할 때 올바른 vertex를 선택해야 한다. 대부분의 경우, 이 선택은 명확하게 확인할 수 있다. 드물게 복잡한 변형으로 인해 작은 메쉬 결함이 발생할 수 있지만, 후속 응용에 지장을 주지 않는다.

### 6. Adaptive Mesh Resolution
또한, FlexiCubes가 적응형 계층적 그리드를 활용하도록 확장하여, geometry 디테일이 높은 영역에서 공간 해상도를 가변적으로 증가시키는 메쉬를 표현할 수 있다. 본 논문에서는 배경 그리드를 다양한 해상도를 가진 계층적 octree로 세분화하는 방식을 사용하였다. 대부분의 알고리즘은 octree에서 변경 없이 적용되지만, octree의 서로 다른 레벨에 걸쳐 있는 인접한 dual vertex를 연결하여 사각형 메쉬 면을 형성하는 문제는 예외이다.

<center><img src='{{"/assets/img/flexicubes/flexicubes-fig11.webp" | relative_url}}' width="43%"></center>
<br>
서로 다른 octree 큐브에서 공유하는 면에서는 SDF의 부호가 일관되지 않을 수 있다. 따라서 위 그림과 같이 coarse face vertex들의 값을 bilinear interpolation해서 fine face vertex의 값으로 사용한다. 이러한 방식은 거의 완벽한 적응형 메쉬를 생성한다. 추출된 메쉬에 구멍이 있는 경우가 소수 발생하지만, 적응적으로 세분화된 메쉬는 상당한 개선을 가져온다.

### 7. Regularizers
각 vertex 위치에 대한 over-parameterization은 의도적인 것이며, 최적화를 용이하게 하는 데 유익하다. 따라서 내부 표현을 정규화하기 위해 두 개의 항을 도입하였다.

첫 번째 항은 각 dual vertex $v$와 해당 vertex가 속한 면을 구성하는 edge 교차점 $u_e$ 사이의 거리 편차에 대한 페널티를 부여한다.

$$
\begin{equation}
\mathcal{L}_\textrm{dev} := \sum_{v \in V} \textrm{MAD} \left[ \{ \vert v - u_e \vert_2 \; : \; u_e \in \mathcal{N}_v \} \right] \\
\textrm{where} \quad \textrm{MAD}(Y) = \frac{1}{\vert Y \vert} \sum_{y \in Y} \vert y - \textrm{mean}(Y) \vert
\end{equation}
$$

이 항은 추출된 연결성을 정규화하고, vertex들이 cell의 중심 근처에 위치하도록 하여 유연하게 변화하고 적응할 수 있는 여유 공간을 확보한다.

두 번째 항은 supervision을 받지 않는 영역에서 잘못된 geometry가 발생하는 것을 방지한다. 저자들은 [NVDiffRec](https://arxiv.org/abs/2111.12503)을 따라, 모든 grid edge에서 부호 변화에 페널티를 부여하였다. Grid edge의 vertex $(a, b)$의 $s$ 값이 $(s_a, s_a)$라고 할 때, $$\mathcal{E}_g$$를 $$\textrm{sign}(s_a) \ne \textrm{sign}(s_b)$$를 만족하는 $(s_a, s_b)$ 쌍들의 집합이라고 정의하자. 그러면 loss는 다음과 같이 주어진다.

$$
\begin{equation}
\mathcal{L}_\textrm{sign} := \sum_{(s_a, s_b) \in \mathcal{E}_g} H \left( \sigma (s_a), \textrm{sign} (s_b) \right)
\end{equation}
$$

($H$는 cross-entropy, $\sigma$는 sigmoid function)

## Experiments
### 1. Mesh Reconstruction
다음은 다른 메쉬 추출 방식들과의 비교 결과이다.

<center><img src='{{"/assets/img/flexicubes/flexicubes-fig13.webp" | relative_url}}' width="100%"></center>
<span style="display: block; margin: 1px 0;"></span>
<center><img src='{{"/assets/img/flexicubes/flexicubes-table2.webp" | relative_url}}' width="68%"></center>
<br>
다음은 적응형 메쉬 생성 예시이다.

<center><img src='{{"/assets/img/flexicubes/flexicubes-fig14.webp" | relative_url}}' width="67%"></center>
<br>
다음은 추출된 메쉬의 품질을 비교한 그래프이다.

<center><img src='{{"/assets/img/flexicubes/flexicubes-fig15.webp" | relative_url}}' width="100%"></center>
<br>
다음은 dual vertex 구성에 대한 ablation 결과이다.

<center><img src='{{"/assets/img/flexicubes/flexicubes-table3.webp" | relative_url}}' width="85%"></center>

### 2. Mesh Optimization with Regularizations
다음은 모든 edge의 길이가 비슷해지도록 정규화를 적용하였을 때의 결과이다. (각 edge 길이와 평균 edge 길이 사이의 L2 loss)

<center><img src='{{"/assets/img/flexicubes/flexicubes-fig16.webp" | relative_url}}' width="88%"></center>
<span style="display: block; margin: 1px 0;"></span>
<center><img src='{{"/assets/img/flexicubes/flexicubes-table4.webp" | relative_url}}' width="95%"></center>

## Limitations
1. Self-intersection이 존재한다.
2. 글로벌하게 연속적이지 않다.