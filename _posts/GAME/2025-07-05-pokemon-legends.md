---
title: "[NS2] Pokémon LEGENDS 아르세우스"
last_modified_at: 2025-07-05
categories:
  - GAME
tags:
  - Game
use_math: true
excerpt: "Pokémon LEGENDS Arceus - 도감 100%"
classes: wide
---

<center><img src='{{"/assets/img/game/pokemon-legend-arceus/pokemon-legend-arceus-10.jpg" | relative_url}}' width="100%"></center>
<br>
<div style="display:flex; justify-content:center; align-items:center; position:relative; user-select:none;">
  <button id="prevBtn" onclick="prevSlide()" style="border:none; background:none; cursor:pointer; padding:0; margin-right:10px; opacity:0.4; pointer-events:none; outline: none;">
    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2" stroke-linejoin="round">
      <polyline points="15 18 9 12 15 6"/>
    </svg>
  </button>
  <div style="width:75%; overflow:hidden;">
    <div id="slider" style="display:flex; transition:transform 0.3s ease;">
      <div style="min-width:100%; box-sizing:border-box; text-align:center;"><img src='{{"/assets/img/game/pokemon-legend-arceus/pokemon-legend-arceus-1.jpg" | relative_url}}' style="width:100%;"></div>
      <div style="min-width:100%; box-sizing:border-box; text-align:center;"><img src='{{"/assets/img/game/pokemon-legend-arceus/pokemon-legend-arceus-2.jpg" | relative_url}}' style="width:100%;"></div>
      <div style="min-width:100%; box-sizing:border-box; text-align:center;"><img src='{{"/assets/img/game/pokemon-legend-arceus/pokemon-legend-arceus-3.jpg" | relative_url}}' style="width:100%;"></div>
      <div style="min-width:100%; box-sizing:border-box; text-align:center;"><img src='{{"/assets/img/game/pokemon-legend-arceus/pokemon-legend-arceus-4.jpg" | relative_url}}' style="width:100%;"></div>
      <div style="min-width:100%; box-sizing:border-box; text-align:center;"><img src='{{"/assets/img/game/pokemon-legend-arceus/pokemon-legend-arceus-5.jpg" | relative_url}}' style="width:100%;"></div>
      <div style="min-width:100%; box-sizing:border-box; text-align:center;"><img src='{{"/assets/img/game/pokemon-legend-arceus/pokemon-legend-arceus-6.jpg" | relative_url}}' style="width:100%;"></div>
      <div style="min-width:100%; box-sizing:border-box; text-align:center;"><img src='{{"/assets/img/game/pokemon-legend-arceus/pokemon-legend-arceus-7.jpg" | relative_url}}' style="width:100%;"></div>
      <div style="min-width:100%; box-sizing:border-box; text-align:center;"><img src='{{"/assets/img/game/pokemon-legend-arceus/pokemon-legend-arceus-8.jpg" | relative_url}}' style="width:100%;"></div>
      <div style="min-width:100%; box-sizing:border-box; text-align:center;"><img src='{{"/assets/img/game/pokemon-legend-arceus/pokemon-legend-arceus-9.jpg" | relative_url}}' style="width:100%;"></div>
    </div>
  </div>
  <button id="nextBtn" onclick="nextSlide()" style="border:none; background:none; cursor:pointer; padding:0; margin-left:10px; opacity:1; outline: none;">
    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2" stroke-linejoin="round">
      <polyline points="9 18 15 12 9 6"/>
    </svg>
  </button>
</div>
<br>
<center><img src='{{"/assets/img/game/pokemon-legend-arceus/pokemon-legend-arceus-11.jpg" | relative_url}}' width="85%"></center>

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