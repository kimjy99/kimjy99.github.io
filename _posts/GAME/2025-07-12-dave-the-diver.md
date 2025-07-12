---
title: "[PC] 데이브 더 다이버"
last_modified_at: 2025-07-12
categories:
  - GAME
tags:
  - Game
use_math: true
excerpt: "DAVE THE DIVER - 도전과제 100%"
classes: wide
---

<center><img src='{{"/assets/img/game/dave-the-diver/dave-the-diver-1.webp" | relative_url}}' width="100%"></center>
<br>
<center><img src='{{"/assets/img/game/dave-the-diver/dave-the-diver-2.webp" | relative_url}}' width="50%"></center>
<br>
<div style="display:flex; justify-content:center; align-items:center; position:relative; user-select:none;">
  <button id="prevBtn" onclick="prevSlide()" style="border:none; background:none; cursor:pointer; padding:0; margin-right:10px; opacity:0.4; pointer-events:none; outline: none;">
    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2" stroke-linejoin="round">
      <polyline points="15 18 9 12 15 6"/>
    </svg>
  </button>
  <div style="width:85%; overflow:hidden;">
    <div id="slider" style="display:flex; transition:transform 0.3s ease;">
      <div style="min-width:100%; box-sizing:border-box; text-align:center;"><img src='{{"/assets/img/game/dave-the-diver/dave-the-diver-3.jpg" | relative_url}}' style="width:100%;"></div>
      <div style="min-width:100%; box-sizing:border-box; text-align:center;"><img src='{{"/assets/img/game/dave-the-diver/dave-the-diver-4.jpg" | relative_url}}' style="width:100%;"></div>
      <div style="min-width:100%; box-sizing:border-box; text-align:center;"><img src='{{"/assets/img/game/dave-the-diver/dave-the-diver-5.jpg" | relative_url}}' style="width:100%;"></div>
      <div style="min-width:100%; box-sizing:border-box; text-align:center;"><img src='{{"/assets/img/game/dave-the-diver/dave-the-diver-6.jpg" | relative_url}}' style="width:100%;"></div>
      <div style="min-width:100%; box-sizing:border-box; text-align:center;"><img src='{{"/assets/img/game/dave-the-diver/dave-the-diver-7.jpg" | relative_url}}' style="width:100%;"></div>
      <div style="min-width:100%; box-sizing:border-box; text-align:center;"><img src='{{"/assets/img/game/dave-the-diver/dave-the-diver-8.jpg" | relative_url}}' style="width:100%;"></div>
      <div style="min-width:100%; box-sizing:border-box; text-align:center;"><img src='{{"/assets/img/game/dave-the-diver/dave-the-diver-9.jpg" | relative_url}}' style="width:100%;"></div>
      <div style="min-width:100%; box-sizing:border-box; text-align:center;"><img src='{{"/assets/img/game/dave-the-diver/dave-the-diver-10.jpg" | relative_url}}' style="width:100%;"></div>
      <div style="min-width:100%; box-sizing:border-box; text-align:center;"><img src='{{"/assets/img/game/dave-the-diver/dave-the-diver-11.jpg" | relative_url}}' style="width:100%;"></div>
      <div style="min-width:100%; box-sizing:border-box; text-align:center;"><img src='{{"/assets/img/game/dave-the-diver/dave-the-diver-12.jpg" | relative_url}}' style="width:100%;"></div>
      <div style="min-width:100%; box-sizing:border-box; text-align:center;"><img src='{{"/assets/img/game/dave-the-diver/dave-the-diver-13.jpg" | relative_url}}' style="width:100%;"></div>
      <div style="min-width:100%; box-sizing:border-box; text-align:center;"><img src='{{"/assets/img/game/dave-the-diver/dave-the-diver-14.jpg" | relative_url}}' style="width:100%;"></div>
      <div style="min-width:100%; box-sizing:border-box; text-align:center;"><img src='{{"/assets/img/game/dave-the-diver/dave-the-diver-15.jpg" | relative_url}}' style="width:100%;"></div>
    </div>
  </div>
  <button id="nextBtn" onclick="nextSlide()" style="border:none; background:none; cursor:pointer; padding:0; margin-left:10px; opacity:1; outline: none;">
    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2" stroke-linejoin="round">
      <polyline points="9 18 15 12 9 6"/>
    </svg>
  </button>
</div>

<script>
  let currentSlide = 0;
  const slider = document.getElementById('slider');
  const totalSlides = slider.children.length;
  const prevBtn = document.getElementById('prevBtn');
  const nextBtn = document.getElementById('nextBtn');
  function updateSlide() {
    slider.style.transform = 'translateX(' + (-100 * currentSlide) + '%)';
    prevBtn.style.opacity = currentSlide === 0 ? 0.4 : 1;
    prevBtn.style.pointerEvents = currentSlide === 0 ? 'none' : 'auto';
    nextBtn.style.opacity = currentSlide === totalSlides - 1 ? 0.4 : 1;
    nextBtn.style.pointerEvents = currentSlide === totalSlides - 1 ? 'none' : 'auto';
  }
  function prevSlide() {
    if (currentSlide > 0) {
      currentSlide--;
      updateSlide();
    }
  }
  function nextSlide() {
    if (currentSlide < totalSlides - 1) {
      currentSlide++;
      updateSlide();
    }
  }
  updateSlide();
</script>