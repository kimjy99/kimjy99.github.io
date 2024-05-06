---
title: "[AI소식] gpt2-chatbot"
last_modified_at: 2024-05-07
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

### 성능 및 특징
누군가가 [rentry](https://rentry.org/gpt2)에 관련 내용과 LMSYS의 여러 결과들을 정리해 두었다. 

- 다양한 도메인의 평균 출력 품질은 최소 GPT-4 및 Claude Opus와 같은 고급 모델과 동일한 수준에 있다.
- 메시지 개수 제한은 시간당 1000개이며 (일일 24,000개 메시지) 이는 gpt-4-1106-preview의 10배이다. 출력 품질을 고려할 때 대규모 컴퓨팅 능력 및/또는 매우 효율적인 모델임을 의미할 수 있다. 
- OpenAI의 tiktoken 토크나이저를 사용하는 것으로 보인다. 이는 gpt2-chatbot과 기타 여러 모델에 대한 특수 토큰의 효과를 비교하여 확인되었다. 
- 보조 명령(assistant instruction)을 추출한 결과 GPT-4 아키텍처를 기반으로 하며 "Personality: v2"를 사용한다. 
- 디테일한 연락처를 요구하면 GPT-3.5/4보다 더 자세하게 OpenAI의 연락처 정보를 제공한다.
- 스스로를 GPT-4 기반이라 주장하며, 자신을 "ChatGPT" 또는 "a ChatGPT"로 부른다. 일반적으로 OpenAI 모델로 생성된 데이터셋으로 학습된 다른 모델들이 혼란스러운 답변을 하는 것과는 구별되게 스스로를 표현하는 방식이 다르다. 
- OpenAI 모델들과 동일한 프롬프트 삽입 (prompt injection) 취약점을 보인다. 
- 한 번도 OpenAI가 아닌 다른 단체에 속해 있다고 주장한 적이 없다.
- Anthropic, Meta, Mistral, Google 등의 모델들은 동일한 프롬프트에 대해 gpt2-chatbot과 다른 출력을 일관되게 출력한다. 
- 수정된 CoT(Chain-of-Thought)와 같은 기술의 영향을 많이 받는 것으로 보인다. 

### 추측
gpt2-chatbot의 정체에 대한 여러 추측들이 나오고 있다. 

1. GPT-4.5의 초기 버전
  - 5월 2일, 샘 올트먼이 하버드 행사에서 gpt2-chatbot이 GPT-4.5가 아니라고 언급했다는 [기사](https://www.axios.com/2024/05/02/mystery-chatbot-openai-gpt2)가 나왔다. 이 기사에는 해당 진술의 출처나 샘 올트먼이 언급한 직접적인 내용이 제공되지 않았다. [Rentry](https://rentry.org/gpt2)의 저자가 해당 주장의 출처와 관련하여 Axios에 연락했지만 아직 답변을 받지 못했다고 한다. 
  - 한 하버드 학생이 해당 행사에 대한 [트윗](https://twitter.com/RishabJainK/status/1785807873626579183)을 올렸다. 그러나 샘 올트먼이 언급한 직접적인 내용이 뭔지를 묻는 질문에는 답하지 않았다고 한다. 
2. MoE (Mixture of Experts)와 비슷한 것을 사용하여 여러 LLM을 하나로 합친 모델
3. 이름대로 GPT-2를 기반으로 새로운 기술이 적용된 모델
  - 최근 LMSYS를 후원하고 있는 MBZUAI의 [논문](https://arxiv.org/abs/2404.05405)에서 GPT-2 아키텍처의 세부 사항을 심층적으로 연구했는데, 짧은 학습 기간 동안 지식 저장 측면에서 GPT-2가 LLaMA/Mistral 아키텍처와 일치하거나 심지어 능가한다고 한다. 
  - LMSYS가 학습을 위해 LMSYS를 통해 생성된 데이터를 활용하였다면, gpt2-chatbot이 스스로를 GPT-4로 식별하는 강한 경향이 GPT-4에서 생성된 데이터를 주로 활용하였기 때문이라고 설명할 수 있다. 
4. 더 큰 Phi-3 버전 (아마도 14B)
  -  5월 3일, Microsoft 연구원인 Sébastien Bubeck이 TikZ로 그린 유니콘을 [트윗](https://twitter.com/SebastienBubeck/status/1786108589700177954)했다. 해당 유니콘은 gpt2-chatbot의 벤치마크 테스트로 일반적으로 사용되는 `Draw a unicorn in TiKZ`라는 프롬프트의 결과이다. 
  - ChatGPT 공식 트위터 계정이 해당 트윗에 [답장](https://twitter.com/ChatGPTapp/status/1786290886017794280)을 했다. 

<center><img src='{{"/assets/img/gpt2-chatbot/gpt2-chatbot-fig5.PNG" | relative_url}}' width="60%"></center>

### 예시
아래는 [rentry](https://rentry.org/gpt2)에 있는 Claude 3 Opus와의 비교 예시이다. 

<center><img src='{{"/assets/img/gpt2-chatbot/gpt2-chatbot-fig2.PNG" | relative_url}}' width="100%"></center>
<br>
프롬프트는 `Generate a level-3 Sierpinski triangle in ASCII.`이며, Claude 3 Opus보다 더 정확하고 자세한 결과를 보인다. 

### Updates
5월 5일, 샘 올트먼이 `im-a-good-gpt2-chatbot`이라는 새 [트윗](https://twitter.com/sama/status/1787222050589028528)을 게시했다. 또한 해당 게시글의 댓글에 `you-are-not-a-good-user`라는 [답글](https://twitter.com/sama/status/1787223078554177750)을 달았다. 

<center><div style="display: flex; align-items: start; justify-content: center; width: 65%">
  <a href="https://twitter.com/sama/status/1787222050589028528"><img src='{{"/assets/img/gpt2-chatbot/gpt2-chatbot-fig3.PNG" | relative_url}}' width="100%"></a>
  &nbsp;
  <a href="https://twitter.com/sama/status/1787223078554177750"><img src='{{"/assets/img/gpt2-chatbot/gpt2-chatbot-fig4.PNG" | relative_url}}' width="100%"></a>
</div></center>
<br>
아마도 5월 9일에 열리는 [사내 이벤트](https://beebom.com/openai-search-engine-report/)에 대하여 관심을 끌기 위한 의도일 확률이 높다고 한다.