---
title: "[Github Page] 페이지 크기 최적화"
last_modified_at: 2026-11-14
categories:
  - 프로그래밍
tags:
  - Github Page
excerpt: "Jekyll Github Page 페이지 크기 최적화"
use_math: true
classes: wide
---

### 기존 문제점
올해 7월 [빌드 최적화](https://kimjy99.github.io/프로그래밍/build-optimization)를 하면서 `index.html`가 지나치게 큰 문제를 발견했다. 빌드 후 모든 페이지의 `index.html`이 약 **12만 줄**이고, 크기도 **1.90MB** 정도로 비정상적으로 컸다. 실제로 `index.html`의 내용을 보면 약 **99.4%**가 빈 줄이었다. 모든 페이지에 동일한 현상이 발생하는 거라, 모든 페이지에 들어가는 부분, 즉 Jekyll 레이아웃이나 include에서 공통된 HTML을 반복 생성되고 있는 것이다.

### 원인 및 해결
2023년 4월 16일에 사이드바를 리뉴얼하면서 기존에 있던 사이드바를 HTML 주석(`<!-- -->`)으로 처리해 두었다. 

{% raw %}
```html
<!-- <span class="nav__sub-title" v-on:click='togglet()'>tags</span>
<ul v-show="flag_t">

  {% assign tags_max = 0 %}
  {% for tag in site.tags %}
    {% if tag[1].size > tags_max %}
      {% assign tags_max = tag[1].size %}
    {% endif %}
  {% endfor %}

    {% for i in (1..tags_max) reversed %}
      {% for tag in site.tags %}
        {% if tag[1].size == i %}
          <li><a href="/tags/#{{tag[0] | slugify}}" class="">{{tag[0] | none }} ({{tag[1].size}})</a></li>
        {% endif %}
      {% endfor %}
    {% endfor %}
    
</ul> -->
```
{: data-file="_includes\nav_list"}
{% endraw %}

문제는 HTML 주석이라 Liquid가 코드를 그대로 실행하고 결과를 주석 안에 출력하고 있었다. Liquid의 {% raw %}`{% %}`{% endraw %} 태그 자체는 아무것도 출력하지 않지만 태그와 태그 사이에 있는 줄바꿈과 들여쓰기는 일반 텍스트라서 그대로 출력되고 있었다. 당시에는 글과 태그 수가 많지 않아 수천 줄 정도의 HTML만 생성되었기 때문에 문제를 인지하지 못했다. 그러나 이후 글과 태그 수가 증가하면서 빈 줄의 수도 약 13배 늘어났다.

3년 넘게 사용하지 않은 불필요한 주석이라 그냥 삭제했다.

### 페이지 크기 및 빌드 시간 비교
주석 삭제 전에는 `index.html`이 평균 1.9MB, 약 12만 줄이었고 `_site` 전체 용량도 4.4GB에 달했다. 주석을 삭제한 후에는 `index.html`이 평균 40KB, 약 2천 줄로 줄었으며 `_site` 전체 용량도 2.6GB로 감소했다.

<center><img src='{{"/assets/img/page-size-optimization/page-size-optimization-1.webp" | relative_url}}' width="90%"></center>
<br>
불필요한 반복문 실행이 줄어들면서 GitHub Actions 빌드 속도도 조금 빨라졌다. 지금까지 기록한 빌드 시간 중 가장 빠르다.