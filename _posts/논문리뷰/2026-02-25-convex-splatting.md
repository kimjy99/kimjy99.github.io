---
title: "[논문리뷰] 3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes"
last_modified_at: 2026-02-25
categories:
  - 논문리뷰
tags:
  - Novel View Synthesis
  - Gaussian Splatting
  - 3D Vision
  - CVPR
excerpt: "3D Convex Splatting 논문 리뷰 (CVPR 2025 Highlight)"
use_math: true
classes: wide
---

> CVPR 2025 (Highlight). [[Paper](https://arxiv.org/abs/2411.14974)] [[Page](https://convexsplatting.github.io/)] [[Github](https://github.com/convexsplatting/convex-splatting)]  
> Jan Held, Renaud Vandeghen, Abdullah Hamdi, Adrien Deliege, Anthony Cioppa, Silvio Giancola, Andrea Vedaldi, Bernard Ghanem, Marc Van Droogenbroeck  
> University of Liege | KAUST | University of Oxford  
> 22 Nov 2024  

<center><img src='{{"/assets/img/convex-splatting/convex-splatting-fig1.webp" | relative_url}}' width="47%"></center>

## Introduction
[3DGS](https://kimjy99.github.io/논문리뷰/3d-gaussian-splatting)의 Gaussian primitive에는 두 가지 주요 한계가 있다.

1. 물리적 경계가 명확하게 정의되어 있지 않아 평면을 정확하게 표현하거나 물리적으로 의미 있는 장면 분해를 가능하게 하는 데 적합하지 않다.
2. Gaussian은 날카로운 모서리나 기하학적 구조를 포착하는 데 부적합하다. 각 Gaussian은 대칭 분포를 갖는 타원체와 유사하게 동작하며, 각진 경계나 평면에 정확하게 밀착하기 어렵다.

이러한 내재적 한계는 sphere packing 문제에서 나타나는데, 구형이나 타원형을 조밀하게 채우면 틈이 생겨 특히 평평하거나 날카로운 모서리 부분에서 비효율적인 커버리지가 발생한다. 구체나 타원체와 마찬가지로 틈 없이 공간을 채우려면 비현실적으로 많은 수의 Gaussian이 필요하게 되어 메모리 사용량과 계산 오버헤드가 증가하다.

이러한 한계를 극복하기 위해, 본 논문에서는 멀티뷰 이미지로부터 기하학적으로 정확한 radiance field를 모델링하고 재구성하기 위한 primitive로 **3D smooth convex**를 활용하는 새로운 방법인 **3D Convex Splatting (3DCS)**을 제안하였다. 3D smooth convex는 Gaussian보다 유연성이 뛰어나며, 더 적은 primitive를 사용하여 날카로운 모서리와 세밀한 표면을 정확하게 형성할 수 있다. 또한, smoothness와 sharpness 파라미터를 도입하여 smooth convex의 곡률과 확산을 각각 제어할 수 있다. 

본 논문에서는 novel view synthesis를 위해 Gaussian의 빠른 렌더링 속도와 3D smooth convex의 유연성을 결합하였다. 효율적인 CUDA 기반 rasterizer를 사용하여 3D smooth convex를 렌더링함으로써 실시간 렌더링을 가능하게 하고 최적화 프로세스를 가속화하였다. 3D Convex Splatting은 미분 가능한 smooth convex를 활용하여 사실적인 장면에서 novel view synthesis를 구현하는 최초의 방법이며, 다른 primitive를 사용하는 기존 방법보다 우수한 성능을 보여준다.

## Method
### 1. Preliminaries on 3D Smooth Convexes
[CvxNet](https://arxiv.org/abs/1909.05736)을 따라, $J$개의 평면을 가진 볼록 다면체를 정의한다. 점 $$\textbf{p} \in \mathbb{R}^3$$과 평면 $$\mathcal{H}_j$$ 사이의 signed distance $L_j (\textbf{p})$는 다음과 같이 정의된다.

$$
\begin{equation}
L_j (\textbf{p}) = \textbf{n}_j \cdot \textbf{p} + d_j
\end{equation}
$$

($$\textbf{n}_j$$는 바깥쪽을 향하는 평면 normal 벡터, $$\textbf{d}_j$$는 offset)

점 $\textbf{p}$에서 convex까지의 signed distance는 모든 $J$개의 signed distance 중 최댓값으로 계산된다.

$$
\begin{equation}
\tilde{\phi} (\textbf{p}) = \max_{j = 1, \ldots, J} L_j (\textbf{p})
\end{equation}
$$

그러나 convex를 부드럽게 표현하기 위해 CvxNet의 LogSumExp 함수를 사용하여 $$\tilde{\phi} (\textbf{p})$$를 $$\phi (\textbf{p})$$로 근사한다.

$$
\begin{equation}
\phi (\textbf{p}) = \log \left( \sum_{j=1}^J \exp (\delta L_j (\textbf{p})) \right)
\end{equation}
$$

여기서 smoothness 파라미터 $\delta > 0$은 convex 근사의 곡률을 제어한다. $\delta$ 값이 클수록 $$\phi (\textbf{p})$$가 $$\tilde{\phi} (\textbf{p})$$에 더 가깝게 근사화되어 모서리가 더 날카로워지고, 값이 작을수록 꼭짓점이 부드러워진다.

Smooth convex의 indicator function $$I(\textbf{p})$$는 SDF에 sigmoid 함수를 적용하여 정의된다.

$$
\begin{equation}
I (\textbf{p}) = \textrm{Sigmoid} (- \sigma \phi (\textbf{p}))
\end{equation}
$$

여기서 sharpness 파라미터 $\sigma > 0$는 convex의 경계에서 indicator function이 얼마나 빠르게 변화하는지를 제어한다. $\sigma$ 값이 클수록 변화율이 가파르게 되어 모양의 경계가 더 명확하게 나타나고, 값이 작을수록 모양이 더 흐릿해진다. Convex의 경계, 즉 $$\phi (\textbf{p}) = 0$$에서 smooth convex의 indicator function은 $I(\textbf{p}) = 0.5$를 만족한다. 

<center><img src='{{"/assets/img/convex-splatting/convex-splatting-fig4.webp" | relative_url}}' width="45%"></center>

### 2. 3DCS: Splatting 3D Smooth Convexes
<center><img src='{{"/assets/img/convex-splatting/convex-splatting-fig3.webp" | relative_url}}' width="100%"></center>

##### 포인트 기반 3D convex shape 표현
카메라 평면 projection에서는 3D convex의 평면 기반 표현이 비실용적이다. CvxNet과 달리, 본 논문에서는 3D convex를 3D 점 집합 $$\mathbb{S} = \{p_1, p_2, \ldots, p_K\}$$의 convex hull로 정의한다. 최적화 과정에서 3D 점들은 자유롭게 움직일 수 있으므로, convex의 위치를 ​​유연하게 조정하고 형태를 변형할 수 있다. 여기서 $K$개의 점 집합은 볼록 다면체의 특정 꼭짓점에 해당하는 것이 아니라, 3D convex의 convex hull을 나타낸다.

##### 2D 이미지 평면으로의 미분 가능한 projection
효율성을 위해 3D convex hull을 명시적으로 구성하고 2D로 projection하는 대신, 3D 점들을 2D로 projection한 다음 2D convex hull을 구성한다. 구체적으로, pinhole 카메라 모델을 사용하여 각 3D 점 $$\textbf{p}_k \in \mathbb{S}$$를 2D 이미지 평면에 projection한다. 이 projection에는 intrinsic $\textbf{K}$와 extrinsic $[\textbf{R} \vert \textbf{t}]$가 사용된다.

$$
\begin{equation}
\textbf{q}_k = \textbf{K} (\textbf{R} \textbf{p}_k + \textbf{t}), \quad \forall k = 1, 2, \ldots, K
\end{equation}
$$

이 projection은 미분 가능하므로 최적화 과정에서 gradient가 3D 포인트들로 다시 흐를 수 있다.

##### 2D convex hull 계산
2D에서 convex shape을 구성하기 위해, projection된 shape의 외곽 경계를 정의하는 점들만 유지함으로써 convex hull을 효율적으로 계산하는  Graham Scan 알고리즘을 적용한다. 이를 통해 2D projection이 렌더링에 필요한 convex 윤곽선을 정확하게 표현할 수 있다.

Graham Scan은 기준점에 대한 극좌표 각도를 기준으로 점들을 정렬하는 것으로 시작한다. 정렬 후, convexity를 유지하면서 convex hull에 점들을 반복적으로 추가하여 convex hull을 구성한다. Convexity는 convex hull 상의 마지막 두 점 $$\textbf{q}_i$$와 $$\textbf{q}_j$$, 그리고 현재 점 $$\textbf{q}_k$$의 외적을 검사하여, 오른쪽으로 꺾이는 점(음의 외적)을 제거함으로써 보장된다.

##### 미분 가능한 2D convex indicator function
본 논문에서는 3D에서 2D로 smooth convex 표현을 확장함으로써 convex hull의 2D convex indicator function을 정의하였다. $$\phi (\textbf{q})$$와 $\textbf{I}(\textbf{q})$는 3D smooth convex와 같이 정의하지만, 3D 점 $\textbf{p}$를 2D 점 $\textbf{q}$로, 3D convex hull을 구분하는 평면을 결과적인 2D convex hull을 구분하는 선으로 대체한다.

파라미터 $\sigma$와 $\delta$는 3D smooth convex에서 계승되며, projection된 2D shape 경계의 sharpness와 smoothness를 제어한다. 2D projection에서의 원근 효과를 고려하기 위해 $\delta$와 $\sigma$를 거리 $d$로 scaling하여 convex shape의 외형이 카메라와의 거리에 따라 일관되게 유지되도록 한다.

##### 효율적인 미분 가능한 rasterizer
저자들은 실시간 렌더링을 구현하기 위해, 임의의 개수의 primitive에 걸쳐 효율적인 역전파를 가능하게 하는 3DGS 타일 기반 rasterizer를 기반으로 rasterizer를 구축했다. 3D-2D projection, convex hull 계산, 선분 정의, indicator function 구현을 포함한 모든 계산은 완전 미분 가능하며, 효율성과 렌더링 속도를 극대화하기 위해 자체 개발한 CUDA 커널 내에서 실행된다. 렌더링 과정에서 알파 블렌딩을 사용하여 3D shape을 타겟 뷰로 rasterization한다.

주어진 카메라 포즈 $\theta$와 각 픽셀 $\textbf{q}$를 렌더링하는 데 필요한 $N$개의 부드러운 convex hull에 대해, 카메라에서 중심까지의 거리가 증가하는 순서대로 $N$개의 convex hull을 정렬하고 각 픽셀 $\textbf{q}$의 색상 값을 계산한다.

$$
\begin{equation}
C (\textbf{q}) = \sum_{n=1}^N \textbf{c}_n o_n I (\textbf{q}) \left( \prod_{i=1}^{n-1} (1 - o_i I (\textbf{q})) \right)
\end{equation}
$$

($$\textbf{c}_n$$은 spherical harmonics (SH)로 정의되는 $n$번째 smooth convex의 색상, $o_n$은 불투명도)

### 3. Optimization
##### 초기화 및 loss
본 논문에서는 3D 공간에서 각 점 집합의 위치, $\delta$ 및 $\sigma$ 파라미터, 불투명도 $o$, 그리고 SH 함수 색상 계수 $\textbf{c}$를 최적화한다. 불투명도를 특정 범위 내로 제한하기 위해 sigmoid 함수를 적용한다. $\delta$와 $\sigma$에는 양수 값을 유지하기 위해 exponential activation을 사용한다.

각 convex shape은 Fibonacci sphere 알고리즘을 이용하여 포인트 클라우드의 점들을 중심으로 하는 구 주위에 균일하게 분포된 점 집합으로 초기화된다. 각 convex shape의 크기는 가장 가까운 세 개의 smooth convex까지의 평균 거리를 기준으로 정의한다. 이를 통해 점들이 밀집된 영역에서는 더 작은 smooth convex가, 점들이 드문 영역에서는 더 큰 smooth convex가 생성된다.

3DGS에서 사용된 접근 방식을 따라, learning rate에 exponential decay 스케줄링 기법을 적용하지만, 이는 3D 점들의 위치 최적화에만 적용된다. 이 기법을 $\delta$와 $\sigma$에도 적용해도 성능 향상은 관찰되지 않았다. 학습 과정에서는 convex의 개수를 줄이기 위해 [Compact 3DGS](https://arxiv.org/abs/2311.13681)와 동일한 regularization loss $$\mathcal{L}_m$$을 사용한다. 최종 loss는 다음과 같다.

$$
\begin{equation}
\mathcal{L} = (1 - \lambda) \mathcal{L}_1 + \lambda \mathcal{L}_\textrm{D-SSIM} + \beta \mathcal{L}_m
\end{equation}
$$

($\lambda = 0.2$, $\beta = 0.0005$)

##### Adaptive convex shape refinement
초기 smooth convex들은 SfM을 통해 얻은 sparse 포인트 클라우드부터 생성돤다. 이 초기 smooth convex들의 개수는 복잡한 장면을 정확하게 표현하기에 불충분하므로, 적응형 제어 메커니즘을 사용하여 smooth convex를 동적으로 추가한다. 3DGS에서는 view-space positional gradient가 큰 Gaussian을 분할하거나 복제하여 추가 Gaussian을 도입하였다. 그러나 3DCS에서는 positional gradient가 under-reconstruction된 영역이나 over-reconstruction된 영역과 일관되게 일치하지 않는다. 오히려 3DCS는 under-reconstruction된 영역과 over-reconstruction된 영역 모두에서 큰 sharpness $\sigma$를 나타낸다.

<center><img src='{{"/assets/img/convex-splatting/convex-splatting-fig5.webp" | relative_url}}' width="45%"></center>
<br>
작은 shape과 큰 shape을 복제하고 분할하는 대신, smooth convex를 일관되게 분할한다. Smooth convex를 두 개의 새로운 convex로 분할하는 대신, $K$개의 새로운 convex로 직접 분할한다. 각각의 새로운 convex shape은 축소되고, 이 새로운 convex shape들의 중심은 초기 convex shape을 정의하는 $K$개의 점에 대응된다. 새로운 convex shape들의 중심을 초기 convex shape의 3D 점에 배치함으로써, 새로운 shape들이 원래 convex shape의 전체 볼륨을 덮어 3D 표현의 완전성을 유지하도록 한다.

최적화 과정에서 더 dense한 shape이 형성되도록 하기 위해, 분할 과정 전반에 걸쳐 sharpness $\sigma$를 증가시키면서 smoothness $\delta$는 동일하게 유지한다. 또한, 미리 정의된 threshold보다 불투명도가 낮은 투명한 convex shape과 너무 큰 convex shape은 제거한다. 

## Experiments
### 1. Experiments on Synthetic Data
다음은 간단한 합성 데이터에 대한 재구성 결과를 비교한 것이다.

<center><img src='{{"/assets/img/convex-splatting/convex-splatting-fig6.webp" | relative_url}}' width="55%"></center>

### 2. Real-world Novel View Synthesis
다음은 기존 방법들과의 비교 결과이다.

<center><img src='{{"/assets/img/convex-splatting/convex-splatting-fig7.webp" | relative_url}}' width="100%"></center>
<span style="display: block; margin: 1px 0;"></span>
<center><img src='{{"/assets/img/convex-splatting/convex-splatting-table1.webp" | relative_url}}' width="100%"></center>
<br>
다음은 densification threshold를 높여서 학습한 가벼운 3DCS 모델과 3DGS를 비교한 결과이다. 

<center><img src='{{"/assets/img/convex-splatting/convex-splatting-fig8.webp" | relative_url}}' width="65%"></center>
<br>
다음은 실내 장면과 야외 장면에 대해 비교한 결과이다.

<center><img src='{{"/assets/img/convex-splatting/convex-splatting-table2.webp" | relative_url}}' width="56%"></center>

### 3. Ablation Study and Discussion
다음은 densification 전략에 대한 ablation 결과이다.

<center><img src='{{"/assets/img/convex-splatting/convex-splatting-fig9.webp" | relative_url}}' width="69%"></center>
<br>
다음은 convex당 포인트 수에 대한 ablation 결과이다.

<center><img src='{{"/assets/img/convex-splatting/convex-splatting-table3.webp" | relative_url}}' width="38%"></center>
<br>
다음은 파라미터 개수에 따른 LPIPS를 3DGS와 비교한 결과이다. (장면: Truck)

<center><img src='{{"/assets/img/convex-splatting/convex-splatting-fig10.webp" | relative_url}}' width="57%"></center>
<br>
다음은 적은 primitive로 장면을 표현할 때의 성능을 비교한 예시이다.

<center><img src='{{"/assets/img/convex-splatting/convex-splatting-fig11.webp" | relative_url}}' width="65%"></center>