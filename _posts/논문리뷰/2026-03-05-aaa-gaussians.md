---
title: "[논문리뷰] AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering"
last_modified_at: 2026-03-05
categories:
  - 논문리뷰
tags:
  - Gaussian Splatting
  - Novel View Synthesis
  - 3D Vision
  - ICCV
excerpt: "AAA-Gaussians 논문 리뷰 (ICCV 2025 Highlight)"
use_math: true
classes: wide
---

> ICCV 2025 (Highlight). [[Paper](https://arxiv.org/abs/2504.12811)] [[Page](https://derthomy.github.io/AAA-Gaussians/)] [[Github](https://github.com/DerThomy/AAA-Gaussians)]  
> Michael Steiner, Thomas Köhler, Lukas Radl, Felix Windisch, Dieter Schmalstieg, Markus Steinberger  
> ETH Zurich | Stanford University | Microsoft  
> 17 Apr 2025  

<center><img src='{{"/assets/img/aaa-gaussians/aaa-gaussians-fig1.webp" | relative_url}}' width="100%"></center>

## Introduction
[3D Gaussian Splatting (3DGS)](https://kimjy99.github.io/논문리뷰/3d-gaussian-splatting)의 2D splat 평가는 매우 효율적이지만, 3D에서 2D Gaussian으로의 projection은 여전히 ​​근사치에 불과하여 넓은 시야각과 같은 비표준 카메라 설정에서 특히 두드러지는 아티팩트가 발생한다. 또한, 3DGS는 2D splat이 현재 뷰 평면에 평행하다고 가정하여 렌더링을 더욱 근사화하므로 간단한 카메라 회전에서도 블렌딩 순서 불일치 및 popping 아티팩트가 발생한다.

몇몇 연구에서는 먼저 screen space에서 3D Gaussian의 경계를 설정한 다음 3D에서 각 광선에 대한 기여도를 계산함으로써 2D splatting과 3D ray tracing 사이의 간극을 메우려고 시도했다. 이러한 접근 방식은 3DGS의 특정 한계를 해결하지만 screen space 계산에 의존하기 때문에 특히 학습 데이터와 크게 다른 시점이나 파라미터를 렌더링할 때 아티팩트가 발생하기 쉽다.

1. 3D에서 가장 높은 기여 지점을 고려하거나 Gaussian별로 변환을 조정하더라도 왜곡이 발생할 수 있다.
2. 일반적인 학습 뷰를 넘어 확대 또는 축소하면 특히 3D에서 Gaussian을 평가할 때 2D 안티앨리어싱 기법을 더 이상 적용할 수 없으므로 아티팩트가 발생한다.
3. 3D bounding plane을 사용해도 Gaussian이 이미지 평면 뒤로 확장될 경우 screen space 계산이 불안정해져 이미지 경계에서 아티팩트가 발생한다.
4. 불투명도가 높은 Gaussian에 초점을 맞춘 단순화된 픽셀 단위 정렬 전략은 특히 학습 분포에서 멀리 떨어진 시점에서 부정확한 렌더링을 초래할 수 있다.

본 논문에서는 3DGS 렌더링 파이프라인의 모든 단계에서 Gaussian의 3D 특성을 고려하며 다음과 같이 3D Gaussian rasterizer의 단점들을 해결하였다.

- 저자들은 기존의 안티앨리어싱 방식을 분석하고, 3D Gaussian을 정확하게 팽창시켜 앨리어싱을 제거하는 적응형 3D smoothing 필터를 도입하였다.
- 3D Gaussian의 경계를 screen space에서 view space로 이동시켜 view frustum 외부에 도달하는 Gaussian의 경계를 안정적으로 유지함으로써 이미지 경계에서 발생하는 popping 아티팩트를 제거하였다.
- 기존의 2D 타일 기반 culling 알고리즘을 screen space를 이용한 frustum 기반 culling으로 확장하여 렌더링 속도를 향상시키고 rasterization에 필요한 정렬 비용을 절감하였다.

## Method
### 1. 3D Gaussian Anti-Aliasing
3D Gaussian 평가에서 가장 시급한 문제 중 하나는 앨리어싱이며, 특히 다양한 거리에서 장면을 렌더링할 때 더욱 그렇다. 3D smoothing 필터는 3D 평가와 본질적으로 호환되지만, 2D screen space Mip 필터는 직접 적용할 수 없다. 결과적으로, 3D 평가에 의존하는 최근 접근 방식들은 screen space 필터를 생략하고 3D smoothing 필터에만 의존한다. 이는 카메라를 가까이 이동시킬 때 아티팩트를 효과적으로 완화하지만, 시야 거리가 멀어질수록 미세한 디테일이 제대로 필터링되지 않아 깜빡임과 시각적 안정성 저하를 초래하는 앨리어싱 문제에 여전히 취약하다.

<center><img src='{{"/assets/img/aaa-gaussians/aaa-gaussians-fig3.webp" | relative_url}}' width="50%"></center>
<br>
이러한 한계를 해결하기 위해 2D screen scape Mip 필터를 3D 평가 방법과 원활하게 통합되는 완전한 3D 필터링 접근 방식으로 대체하여 저주파 앨리어싱을 효과적으로 방지한다. 단순한 접근 방식은 각 렌더링 뷰에 대해 3D smoothing 필터를 다시 계산하는 것이지만, 이는 Gaussian이 지나치게 투명해지는 문제를 야기한다. 이 문제는 진폭이 부피 변화에 따라 감소하는 반면, Gaussian은 광선 $\textbf{d}$를 따라 완전히 적분되는 것이 아니라 최대 기여 지점에서만 평가되기 때문에 발생한다.

$$
\begin{equation}
\hat{\mathcal{G}} (\textbf{x}) = \sqrt{\frac{\vert \Sigma \vert}{\vert \hat{\Sigma} \vert}} \exp \left( -\frac{1}{2} (\textbf{x} - \boldsymbol{\mu})^\top \hat{\Sigma}^{-1} (\textbf{x} - \boldsymbol{\mu}) \right)
\end{equation}
$$

결과적으로 광선을 따라 scaling 변화를 통합하면 이방성이 높은 Gaussian들의 진폭 scaling이 과대평가되어 정규화 계수가 지나치게 작아진다.

$$
\begin{equation}
\sqrt{\frac{\vert \Sigma \vert}{\vert \hat{\Sigma} \vert}} = \sqrt{\frac{\prod_{i=1}^3 \textbf{s}_i^2}{\prod_{i=1}^3 (\textbf{s}_i^2 + \frac{k}{\hat{v}^2})}}, \quad \hat{v} = \frac{f}{d}
\end{equation}
$$

($f$는 초점 거리, $d$는 Gaussian mean $$\boldsymbol{\mu}$$의 view space에서의 거리)

특히, 이러한 감소는 $\textbf{d}$에 수직인 방향의 scale 변화가 최소화된 경우에도 발생하므로, 보다 강력한 필터링 접근 방식이 필요함을 의미한다. 따라서, $\textbf{d}$에 수직인 방향의 scale 변화만을 고려하도록 정규화식을 재구성한다.

$$
\begin{equation}
\hat{\mathcal{G}}_\bot (\textbf{x}) = \sqrt{\frac{\vert \Sigma_\bot \vert}{\vert \hat{\Sigma}_\bot \vert}} \exp \left( -\frac{1}{2} (\textbf{x} - \boldsymbol{\mu})^\top \hat{\Sigma}^{-1} (\textbf{x} - \boldsymbol{\mu}) \right)
\end{equation}
$$

($\textbf{d}$는 $$\boldsymbol{\mu}$$와 카메라 원점 $\textbf{o}$ 사이의 정규화된 벡터, $$\Sigma_\bot$$은 $\textbf{d}$에 수직인 subspace에 projection된 2$\times$2 공분산 행렬)

수직 scaling factor는 다음과 같이 표현할 수 있다.

$$
\begin{equation}
\sqrt{\frac{\vert \Sigma_\bot \vert}{\vert \hat{\Sigma}_\bot \vert}} = \sqrt{\frac{\vert \Sigma \vert \textbf{d}^\top \Sigma^{-1} \textbf{d}}{\vert \hat{\Sigma} \vert \textbf{d}^\top \hat{\Sigma}^{-1} \textbf{d}}}
\end{equation}
$$

공분산 행렬의 역행렬은 $$\Sigma^{-1} = \textbf{R}\textbf{S}^{-2} \textbf{R}^\top$$이므로, $$\textbf{d}^\top \Sigma^{-1} \textbf{d}$$를 다음과 같이 표현할 수 있다.

$$
\begin{equation}
\textbf{d}^\top \Sigma^{-1} \textbf{d} = \textbf{d}^\top \textbf{R}\textbf{S}^{-2} \textbf{R}^\top \textbf{d} = \sum_{i=1}^3 \frac{\textbf{d}_i^{\prime 2}}{\textbf{s}_i^2}, \quad \textrm{where} \; \textbf{d}^\prime = \textbf{R}^\top \textbf{d}
\end{equation}
$$

따라서, 수직 scaling factor는 다음과 같이 단순화할 수 있다.

$$
\begin{equation}
\sqrt{\frac{\vert \Sigma_\bot \vert}{\vert \hat{\Sigma}_\bot \vert}} = \sqrt{\frac{\textbf{d}_1^{\prime 2} \textbf{s}_2^2 \textbf{s}_3^2 + \textbf{d}_2^{\prime 2} \textbf{s}_1^2 \textbf{s}_3^2 + \textbf{d}_3^{\prime 2} \textbf{s}_1^2 \textbf{s}_2^2}{\textbf{d}_1^{\prime 2} \hat{\textbf{s}}_2 \hat{\textbf{s}}_3 + \textbf{d}_2^{\prime 2} \hat{\textbf{s}}_1 \hat{\textbf{s}}_3 + \textbf{d}_3^{\prime 2} \hat{\textbf{s}}_1 \hat{\textbf{s}}_2}}, \quad \textrm{where} \; \hat{\textbf{s}}_i = \textbf{s}_i^2 + \frac{k}{\hat{v}^2}
\end{equation}
$$

이 공식은 명시적인 역행렬 계산을 피하면서 효율적인 계산을 가능하게 하여 수치적 안정성을 보장한다. 카메라를 Gaussian에 가깝게 이동할 때 발생하는 아티팩트를 해결하기 위해 3D smoothing 필터를 제안된 3D 커널과 통합한다. 모든 학습 카메라에서 관찰된 최대 샘플링 주파수 $$\hat{v}_\textrm{train}$$을 저장한다. 렌더링 시 유효 샘플링 주파수는 다음과 같이 정의된다.

$$
\begin{equation}
\hat{v}^\prime = \min (\hat{v}_\textrm{min}, \hat{v})
\end{equation}
$$

($$\hat{v}^\prime$$는 필터에 사용되는 샘플링 주파수)

이 공식은 카메라가 가까워질 때 Gaussian이 과도하게 축소되지 않도록 하면서도 카메라가 멀어질 때 효과적인 앨리어싱 방지 기능을 제공한다.

### 2. Perspective Correct Bounding
Rasterizer에서 3D Gaussian을 효율적으로 평가하려면 불필요한 평가를 피하기 위해 screen space에서 정확한 경계 설정이 필요하다. Projective space에서 $$\tau_\rho$$ level set으로 정의된 타원체에 정확한 plane fitting을 수행하면 Gaussian의 범위가 이미지 평면 뒤에 도달할 경우 실패한다. 이를 완화하기 위해 해당 Gaussian을 제거하면 눈에 띄는 popping 현상이 발생한다.

<center><img src='{{"/assets/img/aaa-gaussians/aaa-gaussians-fig5.webp" | relative_url}}' width="47%"></center>
<br>
본 논문에서는 view space에서 평면 $$\boldsymbol{\pi}_\theta = (\cos(\theta), 0, -\sin(\theta), 0)^\top$$와$$\boldsymbol{\pi}_\phi = (0, \cos(\phi), −\sin(\phi), 0)^\top$$을 사용하여 plane fitting을 수행하고 $\theta$와 $\phi$를 구했다.

$$
\begin{aligned}
\theta_{1,2} &= \textrm{tan}^{-1} \left( \frac{s_{1,3} \pm \sqrt{s_{1,3}^2 - s_{1,1} s_{3,3}}}{s_{3,3}} \right) \\
\phi_{1,2} &= \textrm{tan}^{-1} \left( \frac{s_{2,3} \pm \sqrt{s_{2,3}^2 - s_{2,2} s_{3,3}}}{s_{3,3}} \right) \\
\textrm{with} \; s_{i,j} &= \langle \textbf{t}, \textbf{T}_{(\textrm{view}, i)} \odot \textbf{T}_{(\textrm{view}, j)} \rangle, \; \textbf{t} = (\tau_\rho, \tau_\rho, \tau_\rho, -1)^\top
\end{aligned}
$$

($$\textbf{T}_{(\textrm{view}, i)}$$는 Gaussian space에서 view space로의 변환 행렬 $$\textbf{T}_{\textrm{view}} = \textbf{V} \textbf{T}$$의 $i$번째 행)

또한, view space Gaussian mean의 x/y 방향 각도 $$\theta_\mu$$, $$\phi_\mu$$를 계산한다. 그 후, 다음 조건을 만족하도록 rotation step을 수행한다.

$$
\begin{equation}
\theta_\mu - \pi < \theta_1 < \theta_\mu < \theta_2 < \theta_\mu + \pi
\end{equation}
$$

그런 다음 bounding step을 수행하여 $$[-\frac{\pi}{2} + \epsilon, \frac{\pi}{2} - \epsilon]$$ 범위로 해당 경계를 screen space로 변환한다.

$$
\begin{equation}
\theta_1 = \max \left( - \frac{\pi}{2} + \epsilon, \theta_1 \right), \quad \theta_2 = \min \left(\frac{\pi}{2} - \epsilon, \theta_2 \right)
\end{equation}
$$

카메라 중심이 Gaussian의 타원체 내부에 있으면 유효한 해가 존재하지 않으므로 해당 Gaussian을 폐기한다. 또한, 타원체가 x축과 교차하면 화면 공간의 y축으로 경계를 설정할 수 없으며, 그 반대의 경우도 마찬가지이다. 이러한 경우 제곱근 안의 항이 음수가 되므로, 해당 축에 대해 화면 전체로 경계를 설정하는 것을 보수적으로 적용한다.

다음 단계에서는 타일 기반 culling을 통해 Gaussian의 영향이 미미하거나 Gaussian의 깊이가 타일 전체에 걸쳐 near plane보다 가까운 모든 타일을 제거한다.

### 3. Frustum-Based Culling
Axis-aligned bounding box (AABB)를 사용한 정확한 screen space 경계 설정은 픽셀 단위 작업량을 줄이는 데 도움이 되지만, 축 정렬이 매우 불량한 타원체에 대해서는 부정확한 경계를 제공한다. 또한, Gaussian은 유효한 화면 경계를 얻을 수 있지만 어떤 픽셀에도 기여하지 못할 수 있다. 이러한 문제를 해결하기 위해, [StopThePop](https://arxiv.org/abs/2402.00525)은 projection된 2D 타원에 대해 타일 단위 culling을 수행하여 Gaussian/타일 조합의 수를 크게 줄이고 계층적 정렬에서 불필요한 정렬 작업을 방지했다.

<center><img src='{{"/assets/img/aaa-gaussians/aaa-gaussians-fig6.webp" | relative_url}}' width="47%"></center>
<br>
본 논문에서는 이 타일 기반 culling 접근 방식을 4개의 평면에서 타일 단위 frustum $\mathcal{F}$를 구성하여 3D로 확장한다.

$$
\begin{aligned}
& \boldsymbol{\pi}_{x_1} = (1, 0, 0, -x_\textrm{min}), \; \boldsymbol{\pi}_{x_2} = (1, 0, 0, -x_\textrm{max}), \\
& \boldsymbol{\pi}_{y_1} = (0, 1, 0, -y_\textrm{min}), \; \boldsymbol{\pi}_{y_2} = (0, 1, 0, -y_\textrm{max})
\end{aligned}
$$

($$x_\textrm{min}$$, $$x_\textrm{max}$$, $$y_\textrm{min}$$, $$y_\textrm{max}$$는 픽셀 좌표에서 타일 경계)

다음으로, 이 3D frustum 내부에서 Gaussian의 최대 기여 지점을 계산하고 $$\rho (\textbf{x})^2$$가 threshold $$\tau_\rho$$보다 큰 타일을 제거한다.

$$
\begin{equation}
\min_{\textbf{x} \in \mathcal{F}} \rho (\textbf{x})^2 < \tau_\rho
\end{equation}
$$

Gaussian mean이 이미 이 frustum 내부에 있는 경우, 그 지점이 최대 기여 지점이 된다. 그렇지 않은 경우, 최대 기여 지점은 frustum의 평면과 모서리 위에 있어야 한다. 평면에서 최대 기여 지점을 찾기 위해 각 평면을 정규화된 Gaussian space로 변환하고 원점에 가장 가까운 점을 찾는다. 단순한 방법은 4개의 평면과 4개의 모서리에 대해 이 과정을 반복하고 projection된 점이 frustum의 경계 내에 있고 카메라 앞에 있는지 확인하는 것이다. 하지만 본 논문에서는 screen space에서 Gaussian mean에 가장 가까운 x/y 평면 및 해당 모서리에만 projection하여 평가 횟수를 2개의 평면과 3개의 모서리로 제한하였다. 이 방법은 효율적으로 구현할 수 있으며, 이는 각 Gaussian에 대해 많은 타일에 대해 이 루틴을 실행해야 하므로 매우 중요하다.

또한, 전처리 과정에서 전체 view frustum을 기준으로 Gaussian을 제거하여 기여도가 낮은 Gaussian을 모두 제거한다.

## Evaluation
### 1. Image Metrics
다음은 in-distribution 뷰들에 대한 렌더링 품질을 비교한 결과이다.

<center><img src='{{"/assets/img/aaa-gaussians/aaa-gaussians-table1.webp" | relative_url}}' width="85%"></center>
<br>
다음은 in-distribution 뷰들에 대한 ablation study 결과이다.

<center><img src='{{"/assets/img/aaa-gaussians/aaa-gaussians-table2.webp" | relative_url}}' width="80%"></center>

### 2. View-Consistent Rendering
다음은 더 넓은 FOV에 대한 렌더링 품질을 비교한 결과이다.

<center><img src='{{"/assets/img/aaa-gaussians/aaa-gaussians-fig7.webp" | relative_url}}' width="73%"></center>
<span style="display: block; margin: 1px 0;"></span>
<center><img src='{{"/assets/img/aaa-gaussians/aaa-gaussians-table3.webp" | relative_url}}' width="46%"></center>
<br>
다음은 여러 해상도에서 [MCMC](https://kimjy99.github.io/논문리뷰/3dgs-mcmc)와 비교한 결과이다.

<center><img src='{{"/assets/img/aaa-gaussians/aaa-gaussians-table4.webp" | relative_url}}' width="50%"></center>

### 3. Performance Timings
다음은 평균 렌더링 시간을 비교한 결과이다.

<center><img src='{{"/assets/img/aaa-gaussians/aaa-gaussians-table5.webp" | relative_url}}' width="42%"></center>