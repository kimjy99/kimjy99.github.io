---
title: "[Github Page] 빌드 최적화"
last_modified_at: 2026-07-04
categories:
  - 프로그래밍
tags:
  - Github Page
excerpt: "Jekyll Github Page 빌드 최적화"
use_math: true
classes: wide
---

### 커스텀 GitHub Actions 워크플로우
기존엔 GitHub Pages 기본 빌드(pages-build-deployment)가 돌았는데, 이게 jekyll 3.10 safe 모드에 캐시를 사용하지 않아 엄청 느렸다. 새 워크플로우는 로컬 머신에서 사용 중인 jekyll 4.3.4로 빌드하고, gem 캐시와 .jekyll-cache를 붙였다.

Github Settings $\rightarrow$ Pages $\rightarrow$ Source를 "GitHub Actions"로 전환해야 새 워크플로우가 배포까지 담당한다.

```yml
name: Build and deploy Jekyll site

on:
  push:
    branches: ["main"]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 1

      - name: Setup Ruby
        uses: ruby/setup-ruby@v1
        with:
          ruby-version: "3.1.6"
          bundler-cache: true

      - name: Cache Jekyll build
        uses: actions/cache@v4
        with:
          path: .jekyll-cache
          key: jekyll-${{ runner.os }}-${{ github.sha }}
          restore-keys: |
            jekyll-${{ runner.os }}-

      - name: Setup Pages
        id: pages
        uses: actions/configure-pages@v5

      - name: Build with Jekyll
        run: bundle exec jekyll build
        env:
          JEKYLL_ENV: production

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4

```
{: data-file=".github/workflows/build.yml"}

### include_cached로 사이드바 메뉴 캐싱
빌드 시간의 78%가 `sidebar/nav_list`였다. 모든 페이지마다 모든 카테고리/태그 메뉴(site.tags 순회하며 개수 계산)를 매번 다시 계산하고 있었다.

어차피 사이드바의 내용은 모든 페이지에서 전부 똑같기 때문에, {% raw %}`{% include nav_list %}`{% endraw %}를 {% raw %}`{% include_cached nav_list %}`{% endraw %}로 바꿔 딱 한 번만 렌더하게 했다.

### 빌드 시간 비교
<center><img src='{{"/assets/img/build-optimization/build-optimization-1.webp" | relative_url}}' width="90%"></center>
<br>
기존에는 빌드 및 배포에 약 70분 소요되었는데, 최적화 후 약 6분 소요된다. 커스텀 빌드라 run 이름이 git commit 메시지로 나온다.

### 남은 이슈
1. 아티팩트 크기가 2.31GB로 너무 크다. 대부분 이미지(webp만 9,500장)라서 CDN으로 옮기고 모든 이미지 경로를 일괄 수정해야 할 듯.
2. 빌드 후 모든 논문리뷰 글 페이지가 약 11만 줄로 비정상적으로 크다. 뭔가 템플릿이 페이지마다 대량의 markup을 뿜고 있는 듯.