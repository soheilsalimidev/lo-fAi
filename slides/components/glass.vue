<template>
  <div
    class="flex  m-auto"
  >

      <div
        class="absolute h-[270px] w-[450px] -translate-x-2/4 -translate-y-2/4 z-[-1] left-2/4 top-2/4"
      >
        <div
          class="absolute rounded-full h-60 w-60 left-[-250px] opacity-80 bottom-[-150px] circle-color circle-1"
        ></div>
        <div
          class="absolute rounded-full h-60 w-60 right-[-100px] opacity-80 bottom-[13.5rem] circle-color circle-2"
        ></div>
      </div>

      <div
        class="h-full w-full backdrop-blur-25 shadow-[0_0_80px_rgba(0,0,0,0.25)] overflow-hidden p-[30px] rounded-4 border-2 border-solid border-[rgba(255,255,255,0.1)] padding-2"
        style="background: rgba(255, 255, 255, 0.1)"
      >
        <slot />
      </div>
  </div>
</template>

<style>
@keyframes explode {
  0% {
    opacity: 1;
  }
  100% {
    transform: scale(1.2);
    opacity: 0;
  }
}
@keyframes float {
  0% {
    -webkit-transform: rotateX(0deg) translateY(0px);
    -moz-transform: rotateX(0deg) translateY(0px);
    -ms-transform: rotateX(0deg) translateY(0px);
    -o-transform: rotateX(0deg) translateY(0px);
    transform: rotateX(0deg) translateY(0px);
  }

  50% {
    -webkit-transform: rotateX(0deg) translateY(1px) translateX(5px);
    -moz-transform: rotateX(0deg) translateY(10px) translateX(5px);
    -ms-transform: rotateX(0deg) translateY(30px) translateX(5px);
    -o-transform: rotateX(0deg) translateY(40px) translateX(5px);
    transform: rotateX(0deg) translateY(30px) translateX(10px);
  }
  100% {
    -webkit-transform: rotateX(0deg) translateY(0px) translateX(1px);
    -moz-transform: rotateX(0deg) translateY(0px) translateX(1px);
    -ms-transform: rotateX(0deg) translateY(0px) translateX(1px);
    -o-transform: rotateX(0deg) translateY(0px) translateX(1px);
    transform: rotateX(0deg) translateY(0px) translateX(1px);
  }
}

.circle-color {
  background: linear-gradient(
    120deg,
    var(--slidev-theme-primary),
    var(--slidev-theme-secondary)
  );
}

.circle-1 {
  animation-timing-function: cubic-bezier(0.54, 0.085, 0.5, 0.92);
  animation-iteration-count: infinite;
  animation-name: float;
  -webkit-animation-name: float;
  animation-duration: 6s;
  -webkit-animation-duration: 6s;
  -webkit-animation-delay: 3.5s;
  animation-delay: 3.5s;
  animation-direction: alternate;
}
.circle-2 {
  animation-timing-function: cubic-bezier(0.54, 0.085, 0.5, 0.92);
  animation-iteration-count: infinite;
  animation-name: float;
  -webkit-animation-name: float;
  animation-duration: 6s;
  -webkit-animation-duration: 6s;
  -webkit-animation-delay: 2s;
  animation-delay: 2s;
  animation-direction: alternate;
}
</style>
