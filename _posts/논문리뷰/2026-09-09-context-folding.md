---
title: "[논문리뷰] Scaling Long-Horizon LLM Agent via Context-Folding"
last_modified_at: 2026-09-09
categories:
  - 논문리뷰
tags:
  - NLP
  - Reinforcement Learning
  - GRPO
  - ICML
excerpt: "Context Folding 논문 리뷰 (ICML 2026)"
use_math: true
classes: wide
---

> ICML 2026. [[Paper](https://arxiv.org/abs/2510.11967)] [[Page](https://context-folding.github.io/)] [[Github](https://github.com/sunnweiwei/FoldAgent)]  
> Weiwei Sun, Miao Lu, Zhan Ling, Kang Liu, Xuesong Yao, Yiming Yang, Jiecao Chen  
> ByteDance Seed | Carnegie Mellon University | Stanford University  
> 13 Oct 2025  

<center><img src='{{"/assets/img/context-folding/context-folding-fig1.webp" | relative_url}}' width="100%"></center>

## Introduction
LLM 에이전트를 훨씬 더 긴 horizon으로 scaling하는 것은 근본적으로 에이전트 프레임워크 설계에 의해 제약된다. 이러한 프레임워크는 전체 상호작용 기록(추론, 도구 호출, 도구 호출 결과)을 단일하고 지속적으로 확장되는 컨텍스트에 선형적으로 축적하는데, 이로 인해 horizon이 확장됨에 따라 다음과 같은 긴 컨텍스트 관련 문제가 발생한다. 

1. LLM이 매우 긴 컨텍스트에서 관련 정보를 활용하는 데 어려움을 겪으면서 성능이 저하죈다.
2. Attention 메커니즘의 제곱에 비례하는 scaling과 KV 캐시 관리 오버헤드 증가로 인해 효율성이 저하된다.

Long-horizon LLM 에이전트의 scaling을 위한 기존 접근 방식은 크게 두 가지 카테고리로 나뉜다.

1. **요약 기반 방식**: 컨텍스트가 가득 차면 사후 요약 단계를 실행한다. 에이전트의 컨텍스트와 추론 흐름을 갑자기 중단시킨다.
2. **다중 에이전트 시스템**: 특화된 에이전트에 task를 분산하여 컨텍스트 길이를 관리한다. 수작업으로 설계된 문제별 워크플로우에 의존하기 때문에 일반화하기 어렵고 end-to-end 최적화에 부적합하다.

본 논문에서는 모델이 능동적으로 컨텍스트를 관리할 수 있도록 하는 에이전트 메커니즘인 **Context Folding**을 제안하였다. 구체적으로, 에이전트는 두 가지 특수 action을 사용하여 컨텍스트를 관리한다.

1. **branch**: sub-task를 위한 임시 하위 경로를 생성한다.
2. **return**: 결과를 요약하고 메인 스레드로 복귀한다. Branch 내의 중간 step들이 컨텍스트에서 folding되어 제거되고, 간결한 요약 정보만 남게 된다.

기존 방법과 비교하여 context folding은 능동적인 컨텍스트 관리에 대한 에이전트적 접근 방식을 가능하게 하며, 에이전트의 짧은 컨텍스트는 유지되고 긴 컨텍스트는 자동으로 관리된다.

Context Folding 프레임워크를 기반으로, 저자들은 복잡한 long-horizon task를 처리하는 LLM 에이전트 학습을 위한 새로운 end-to-end RL 알고리즘인 **FoldGRPO**를 제안하였다. FoldGRPO는 동적으로 folding된 LLM 컨텍스트와 context folding 동작을 직접적으로 유도하는 토큰 레벨의 process reward를 통합하여 표준 GRPO를 강화한 것이다.

구체적으로, 메인 컨텍스트에서 토큰 집약적인 연산을 억제하는 unfolding token penalty를 통해 문제를 sub-task로 효과적으로 분해하는 방법을 모델에 학습시켰다. 또한, out-of-scope penalty를 통해 sub-task에 대한 집중을 유지하고, 요약 정보에 중요한 정보를 보존하는 방법을 학습시켰다. 이러한 능력을 습득함으로써 에이전트는 훨씬 더 긴 상호작용 기록을 처리할 수 있으며, 이를 통해 에이전트의 유효 처리 범위를 확장하고 시스템 효율성을 전반적으로 향상시킬 수 있다.

## Method
### 1. Vanilla Formulation
질문 $q$가 주어지면, 에이전트는 다음과 같은 multi-turn 상호작용 궤적을 생성한다.

$$
\begin{equation}
\tau = (a_1, o_1, a_2, o_2, \ldots, a_T, o_T)
\end{equation}
$$

$a_i$는 step $i$에서의 LLM 출력(추론 및 도구 호출)이고, $o_i$는 해당 도구 호출 결과이다. 일반적인 [ReAct](https://arxiv.org/pdf/2210.03629) 스타일 에이전트는 다음과 같이 상호작용을 모델링한다.

$$
\begin{equation}
p_\theta^\textrm{ReAct} (\tau \mid q) = \prod_{i \in [T]} \pi_\theta (a_i \mid q, (a_1, o_1, \ldots, a_{i-1}, o_{i-1}))
\end{equation}
$$

이는 LLM 생성 시마다 전체 상호작용 기록을 컨텍스트에 추가하는 방식이다. 그러나 연구나 에이전트 코딩과 같은 long-horizon task에서는 광범위한 상호작용으로 인해 $\tau$가 빠르게 누적되어 컨텍스트가 지나치게 길어질 수 있다. 또한 컨텍스트가 확장됨에 따라 모델의 추론 및 instruction following 능력이 저하될 수 있으며, 이는 에이전트가 long-horizon task를 완료하는 데 더 큰 어려움을 초래한다.

### 2. Context Folding
<center><img src='{{"/assets/img/context-folding/context-folding-fig2a.webp" | relative_url}}' width="38%"></center>
<br>
저자들은 에이전트가 컨텍스트 관리를 위해 호출할 수 있는 두 가지 도구를 설계했다. 질문 $q$를 해결하기 위한 메인 스레드에서 시작하여 다음과 같은 도구를 사용할 수 있다.

1. `branch(description, prompt)`: 메인 스레드에서 분기하여 별도의 컨텍스트를 사용하여 sub-task $q^\prime$을 완료한다. `description`은 sub-task에 대한 간략한 요약이고 `prompt`는 이 branch에 대한 자세한 명령이다. 도구는 branch가 생성되었음을 나타내는 템플릿 메시지를 리턴한다.
2. `return(message)`: 이 branch에서 생성된 컨텍스트를 접어서 메인 스레드로 돌아간다. 메시지는 이 branch의 결과를 설명한다. 이 도구를 호출하면 에이전트 컨텍스트는 메인 스레드로 다시 전환되고, branch에서 생성된 `message`가 추가된다.

이 두 가지 도구를 사용하면 에이전트는 독립적인 sub-task를 해결하기 위해 별도의 컨텍스트로 분기하고, branch의 중간 step을 접고 branch 결과만 추가하여 메인 스레드로 복귀함으로써 컨텍스트를 능동적으로 관리할 수 있다.

$$
\begin{equation}
p_\theta^\textrm{Context Fold} (\tau \mid q) = \prod_{i \in [T]} \pi_\theta (a_i \mid q, \mathcal{F}(a_1, o_1, \ldots, a_{i-1}, o_{i-1}))
\end{equation}
$$

$\mathcal{F}$는 `branch`와 `return` 사이의 상호작용 이력을 통합하는 컨텍스트 관리자이다. 예를 들어, 컨텍스트 관리자는 다음과 같이 이전 branch의 모든 action-observation 쌍을 통합한다. ($a_2$와 $a_4$ 사이, $a_5$와 $a_8$ 사이의 구간이 접힘)

$$
\begin{equation}
\mathcal{F} (a_1, o_1, a_2, \underbrace{o_2, a_3, o_3, a_4}_{\textrm{branch 1}}, o_4, a_5, \underbrace{o_5, a_6, o_6, a_7, o_7, a_8}_{\textrm{branch 2}}, o_8, a_9, o_9, a_{10}, o_{10}) \\
\rightarrow (a_1, o_1, a_2, o_4, a_5, o_8, a_9, o_9, a_{10}, o_{10})
\end{equation}
$$

##### Inference 효율성
Inference 시에 에이전트는 컨텍스트 KV 캐시를 관리한다. `return`이 호출되면, 컨텍스트 prefix가 `branch` 호출 전과 일치하는 해당 `branch` 위치로 KV 캐시를 롤백한다. 이러한 방식으로 context folding은 inference 측면에서 효율적이다.

##### 구현 방식: plan-execution
Context folding을 구현하기 위해, 에이전트가 두 가지 state를 번갈아 수행하는 plan-execution 프레임워크를 채택했다.

1. **Planning state**: 에이전트는 메인 스레드에서 고수준 추론을 수행하고, task를 분해하며, sub-task에 대한 branch를 시작할 시점을 결정한다. 이 state에서는 메인 컨텍스트가 고수준 전략에 집중할 수 있도록 토큰 집약적인 도구 사용을 자제한다.
2. **Execution state**: 에이전트는 활성 branch 내에서 할당된 sub-task를 완료한다. 명확한 구조를 유지하고 중첩된 복잡성을 방지하기 위해 이 state에서는 새로운 branch 생성을 비활성화한다.

### 3. FoldGRPO: End-to-End RL for Context-Folding Agent
Context folding 에이전트를 최적화하기 위해, 저자들은 end-to-end RL 학습 프레임워크인 **Folded-context Group Relative Policy Optimization (FoldGRPO)**을 도입하였다. FoldGRPO는 메인 스레드와 sub-task branch를 포함한 전체 상호작용 궤적을 공동으로 최적화하는 동시에, context folding 모델링에 따라 rollout 기록을 접어 학습 중에 간결한 컨텍스트를 유지한다. 또한 FoldGRPO는 에이전트의 branching 동작 학습을 효율적으로 유도하기 위한 새로운 reward 디자인을 사용한다.

<center><img src='{{"/assets/img/context-folding/context-folding-fig2b.webp" | relative_url}}' width="43%"></center>

#### Overall Algorithm Design
각 학습 step에서 학습 데이터셋 $\mathcal{D}$의 task $q$에 대해 context folding 모델에 따라 old policy $$\pi_\textrm{old}$$에서 $G$개의 궤적 $$\{\tau_i\}_{i=1}^G$$가 샘플링된다. 각 완전한 궤적 $$\tau_i = (a_{i,1}, o_{i,1}, \cdots, a_{i,T}, o_{i,T})$$는 토큰들의 시퀀스이다. 각 궤적 $$\tau_i$$는 RLVR의 레시피에 따라 최종 reward $$R_i \in \{0, 1\}$$을 갖는다.

FoldGRPO의 학습 loss는 다음과 같이 정의된다.

$$
\begin{equation}
\mathcal{J}_\textrm{FoldGRPO} = \mathbb{E}_{q \sim \mathcal{D}, \{\tau_i\}_{i=1}^G \sim \pi_\textrm{old} (\cdot \vert q)} \left[ \frac{1}{\sum_{i=1}^G \vert \tau_i \vert} \sum_{i=1}^G \sum_{t=1}^{\vert \tau_i \vert} \min \left\{ r_{i,t} (\theta) \hat{A}_{i,t}, \textrm{clip} (r_{i,t} (\theta), 1 - \epsilon_\textrm{low}, 1 + \epsilon_\textrm{high}) \hat{A}_{i,t} \right\} \right] \\
\textrm{where} \quad r_{i,t} (\theta) = \frac{\pi_\theta (\tau_{i,t} \mid q, \mathcal{F}(\tau_{i, <t}))}{\pi_\textrm{old} (\tau_{i,t} \mid q, \mathcal{F}(\tau_{i, <t}))} \cdot \textbf{1}_{\tau_{i,t}}^\textrm{LLM}, \quad \hat{A}_{i,t} = \frac{\textrm{clip}(R_i + Q_{i,t}, 0, 1) - \textrm{mean}(\{R_i\}_{i=1}^G)}{\textrm{std}(\{R_i\}_{i=1}^G)}
\end{equation}
$$

($$\textbf{1}_{\tau_{i,t}}^\textrm{LLM}$$는 LLM에서 생성된 토큰만 최적화하고 tool observation에서 나온 토큰은 마스킹)

FoldGRPO의 두 가지 주요 기능은 다음과 같다.

1. **Context folding.** Policy 최적화 시 전체 상호작용 기록을 컨텍스트에 추가하는 기존의 GRPO와 달리, FoldGRPO는 branch-return action을 기반으로 토큰 $$\tau_{i,t}$$에 대한 컨텍스트를 folding하는 컨텍스트 관리자 $\mathcal{F}$를 $$\tau_{i, < t}$$에 적용한다.
2. **Process reward signal.** Advantage $$\hat{A}_{i,t}$$ 계산 시, 모델의 branch-return action을 정규화하기 위해 토큰 레벨의 process reward $Q_{i,t}$가 추가된다.

#### Process Reward Design
RLVR에서 에이전트는 일반적으로 task 성공 또는 실패에 기반한 바이너리 outcome reward를 통해 최적화된다. 그러나 이러한 sparse한 reward 신호는 효과적인 context folding 학습에 불충분하다. 구체적으로 두 가지 중요한 실패 모드가 나타난다.

1. 에이전트가 전략적으로 계획하지 못하여 토큰 집약적인 task를 메인 컨텍스트에 그대로 남겨두어 사용 가능한 토큰 예산을 빠르게 소진하는 경우
2. 에이전트가 적절한 branch 관리에 어려움을 겪어 sub-task가 완료된 후 하위 branch에서 복귀하지 않고 동일한 branch 내에서 후속 작업을 계속하는 경우

에이전트를 효과적으로 최적화하기 위해 메인 궤적 토큰과 branch 궤적 토큰에 대해 각각 별도의 토큰 레벨의 process reward를 도입한다.

##### Unfolded token penalty
메인 스레드의 컨텍스트 길이가 전체 컨텍스트 제한의 ​​50%를 초과하면, branch를 생성하는 턴의 토큰을 제외한 메인 스레드의 모든 토큰에 $Q_{i,t} = -1$을 적용한다. 이는 에이전트가 메인 스레드에서 branch 외부에 토큰 집약적인 행동을 수행하는 것에 대한 페널티를 부여하고, 에이전트가 이러한 행동을 별도의 branch에서 수행하도록 유도한다.

##### Out-scope penalty
각 branch에 대해 GPT-5-nano를 사용하여 branch 프롬프트와 return된 메시지를 기반으로 에이전트가 지정된 sub-task 외의 task를 수행했는지 여부를 판단한다. 만약 그렇다면, 해당 branch의 모든 토큰에 $Q_{i,t} = -0.2$를 적용하여 범위 이탈 행위에 대한 페널티를 부여한다.

##### Failure penalty
도구 호출에 실패한 턴의 모든 토큰에 $Q_{i,t} = -1$을 적용한다. 다른 모든 경우에는 $Q_{i,t} = 0$을 적용한다.

## Experiments
- 데이터셋: BrowseComp-Plus (연구), SWE-Bench Verified (소프트웨어 엔지니어링)
- Base model: Seed-OSS-36B-Instruct

### 1. Main Results
다음은 BrowseComp-Plus와 SWE-Bench Verified에서의 성능을 비교한 결과이다.

<center><img src='{{"/assets/img/context-folding/context-folding-table1.webp" | relative_url}}' width="88%"></center>

### 2. Performance by Task Difficulty
다음은 task 난이도에 따른 성능을 비교한 결과이다.

<center><img src='{{"/assets/img/context-folding/context-folding-fig3.webp" | relative_url}}' width="90%"></center>
<br>
다음은 task 난이도에 따른 learning dynamics를 비교한 결과이다.

<center><img src='{{"/assets/img/context-folding/context-folding-fig4.webp" | relative_url}}' width="95%"></center>

### 3. Ablation of RL Algorithm
다음은 RL 알고리즘에 대한 ablation 결과이다.

<center><img src='{{"/assets/img/context-folding/context-folding-table2.webp" | relative_url}}' width="85%"></center>

### 4. Performance by Context Length
다음은 (왼쪽) 최대 컨텍스트 길이와 (오른쪽) 결합된 질문 수에 대한 성능을 비교한 결과이다.

<center><img src='{{"/assets/img/context-folding/context-folding-fig5.webp" | relative_url}}' width="75%"></center>

### 5. Further Analysis
다음은 BrowseComp-Plus에서의 도구 호출 기록 및 컨텍스트 길이에 대한 예시이다.

<center><img src='{{"/assets/img/context-folding/context-folding-fig7.webp" | relative_url}}' width="100%"></center>
<br>
다음은 학습 시간을 비교한 결과이다.

<center><img src='{{"/assets/img/context-folding/context-folding-fig8.webp" | relative_url}}' width="37%"></center>