<template>
  <div class="bg-section slidev-layout new-section p-4">
    <CornerCurves class="absolute left-0 top-0 transform rotate-90" />
    <div class="flex flex-col justify-center text-center h-full w-full">
      <glass id="glass" class="min-h-2/3 w-4/5 text-start leading-7">
        <slot />
      </glass>

    </div>
    <AsGraphic type="zigzag" class="absolute top-20 -right-30" />
  </div>
</template>

<style>
#glass h1,#glass h2{
  text-align: center
}
.slidev-layout.new-section {
  h1 {
    @apply text-primary text-4xl leading-20 mb-0 font-title font-extrabold;
  }

  h1+p {
    @apply mt-4 opacity-100;
  }

  p>img {
    @apply w-2/3 mx-auto;
  }
}
</style>