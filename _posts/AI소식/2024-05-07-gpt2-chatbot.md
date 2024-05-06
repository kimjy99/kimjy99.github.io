---
title: "[AI소식] gpt2-chatbot"
last_modified_at: 2024-04-30
categories:
  - AI소식
tags:
  - Natural Language Processing
  - AI
excerpt: "gpt2-chatbot"
use_math: true
classes: wide
---

LLM 리더보드인 [LMSYS](https://chat.lmsys.org/)에 **gpt2-chatbot**이라는 정체불명의 LLM이 등장했는데 특정 벤치마크에서 **GPT-4**보다 성능이 좋다고 한다. 

"⚔️ Arena (battle)"의 경우 벤치마킹을 위해 블라인드 처리되어 있으며 "💬 Direct Chat"을 통해 직접 사용해 볼 수 있다. 

이와 관련하여 OpenAI CEO인 샘 올트먼이 [트윗](https://x.com/sama/status/1785107943664566556)을 게시했다. 

<center><a href="https://x.com/sama/status/1785107943664566556"><img src='{{"/assets/img/gpt2-chatbot/gpt2-chatbot-fig1.PNG" | relative_url}}' width="45%"></a></center>

## 특징
누군가가 [rentry](https://rentry.org/gpt2)에 관련 내용과 LMSYS의 여러 결과들을 정리해 두었다. 

- 다양한 도메인의 평균 출력 품질은 최소 GPT-4 및 Claude Opus와 같은 고급 모델과 동일한 수준에 있다.
- OpenAI의 tiktoken 토크나이저를 사용하는 것으로 보인다. 이는 gpt2-chatbot과 기타 여러 모델에 대한 특수 토큰의 효과를 비교하여 확인되었다. 
- 보조 명령(assistant instruction)을 추출한 결과 GPT-4 아키텍처를 기반으로 하며 "Personality: v2"를 사용한다. 
- 디테일한 연락처를 요구하면 GPT-3.5/4보다 더 자세하게 OpenAI의 연락처 정보를 제공한다.
- 스스로를 GPT-4 기반이라 주장하며, 자신을 "ChatGPT" 또는 "a ChatGPT"로 부른다. 일반적으로 OpenAI 모델로 생성된 데이터셋으로 학습된 다른 모델들이 혼란스러운 답변을 하는 것과는 구별되게 스스로를 표현하는 방식이 다르다. 
- OpenAI 모델들과 동일한 프롬프트 삽입 (prompt injection) 취약점을 보인다. 
- 한 번도 OpenAI가 아닌 다른 단체에 속해 있다고 주장한 적이 없다.
- Anthropic, Meta, Mistral, Google 등의 모델들은 동일한 프롬프트에 대해 gpt2-chatbot과 다른 출력을 일관되게 출력한다. 
- 수정된 CoT(Chain-of-Thought)와 같은 기술의 영향을 많이 받는 것으로 보인다. 

<br>
성능이 GPT-5까지는 아니라서

1. GPT-4.5의 초기 버전
2. MoE (Mixture of Experts)와 비슷한 것을 사용하여 여러 LLM을 하나로 합친 모델
3. 이름대로 GPT-2를 기반으로 새로운 기술이 적용된 모델

등의 추측들이 나오고 있다. 

특히 마지막 추측의 경우, 최근 LMSYS를 후원하고 있는 MBZUAI의 [논문](https://arxiv.org/abs/2404.05405)에서 GPT-2 아키텍처의 세부 사항을 심층적으로 연구했는데, 해당 논문에서 다음과 같은 GPT-2의 장점이 확립되었기 때문이다. 

> "Rotary embedding을 갖춘 GPT-2 아키텍처는 특히 짧은 학습 기간 동안 지식 저장 측면에서 LLaMA/Mistral 아키텍처와 일치하거나 심지어 능가한다. 이는 LLaMA/Mistral이 덜 안정적이고 학습하기 어려운 GatedMLP를 사용하기 때문에 발생한다."

또한 LMSYS가 학습을 위해 LMSYS를 통해 생성된 데이터를 활용하였다면, gpt2-chatbot이 스스로를 GPT-4로 식별하는 강한 경향이 GPT-4에서 생성된 데이터를 주로 활용하였기 때문이라고 설명할 수 있다. 

## 예시
아래는 [rentry](https://rentry.org/gpt2)에 있는 Claude 3 Opus와의 비교 예시이다. 

<center><img src='{{"/assets/img/gpt2-chatbot/gpt2-chatbot-fig2.PNG" | relative_url}}' width="100%"></center>
<br>
프롬프트는 `Generate a level-3 Sierpinski triangle in ASCII.`이며, Claude 3 Opus보다 더 정확하고 자세한 결과를 보인다. 