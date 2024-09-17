---
theme: penguin
class: text-center
lineNumbers: false
info: "Train a small language model for generating lo-fi music"
drawings:
  persist: false
transition: slide-left
title: "آموزش یک مدل زبانی كوچک برای ساخت موسیقی lo-fi"
mdc: true
defaults:
  layout: "new-def"
layout: intro
htmlAttrs:
  dir: "rtl"
  lang: "fa"
---

# آموزش یک مدل زبانی كوچک برای ساخت موسیقی lo-fi

<h3 class="font-bold text-3xl">سهیل سلیمی</h3>

__استاد راهنما: دکتر زجاجی__

---
layout: "default"
---
<Toc :columns="2"/>

---
src: "./pages/1-problems.md"
---

---
src: "./pages/2-literature.md"
---

---
src: "./pages/3-methodology.md"
---

---
src: "./pages/4-implementation.md"
---

---
src: "./pages/5-results.md"
---


---
layout: center
---
<div style="height: 10vh" class="flex flex-col justify-center items-center">
<span class="text-7xl font-bold bg-gradient-to-r from-orange-700 via-blue-500 to-green-400 text-transparent bg-clip-text bg-300% animate-gradient p-0 m-0 h-42">
 Thank You
</span>

<p style="height: 10vh" class="text-2xl font-bold bg-gradient-to-r from-orange-700 via-blue-500 to-green-400 text-transparent bg-clip-text bg-300% animate-gradient p-0 m-0">
 Hope you have good day
</p>
</div>

<style>
.animate-gradient {
  background-size: 300%;
  -webkit-animation: animatedgradient 6s ease infinite alternate;
  -moz-animation: animatedgradient 6s ease infinite alternate;
  animation: animatedgradient 6s ease infinite alternate;
}

@keyframes animatedgradient {
  0% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
  100% {
    background-position: 0% 50%;
  }
}
</style>
