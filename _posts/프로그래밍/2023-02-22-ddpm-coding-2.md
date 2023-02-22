---
title: "[구현] Colab에서 DDPM 구현하기 - 다양한 결과 모음"
last_modified_at: 2023-02-22
categories:
  - 프로그래밍
tags:
  - Diffusion
  - Computer Vision
  - AI
excerpt: "Colab에서 DDPM 구현하기 - 다양한 결과 모음"
use_math: true
classes: wide
---

> 이전 글: [진행상황](https://kimjy99.github.io/프로그래밍/ddpm-coding/)

### 1. 괜찮은 샘플 모음 (Cherry Picking)

<center><img src='{{"/assets/img/ddpm-coding/samples.png" | relative_url}}' width="100%"></center>

### 2. Reverse process

<center><img src='{{"/assets/img/ddpm-coding/reverse4_5.gif" | relative_url}}'></center>
<br>
전체 reverse process ($t = 1000$에서 $t = 0$까지)

<center><img src='{{"/assets/img/ddpm-coding/reverse.png" | relative_url}}' width="100%"></center>
<br>
마지막 200 step ($t = 200$에서 $t = 0$까지)

<center><img src='{{"/assets/img/ddpm-coding/reverse2.png" | relative_url}}' width="100%"></center>
<br>
마지막 100 step ($t = 100$에서 $t = 0$까지)

<center><img src='{{"/assets/img/ddpm-coding/reverse3.png" | relative_url}}' width="100%"></center>

### 3. 같은 latent vector에서 샘플링

<center><img src='{{"/assets/img/ddpm-coding/same-latent.png" | relative_url}}' width="60%"></center>
<br>
맨 왼쪽 위 샘플의 latent vector ($x_T$)에서 15번 더 샘플링한 결과이다. 괜찮은 샘플의 latent vector에서 샘플링하니까 전체적으로 결과가 괜찮게 나오는 것 같다. 