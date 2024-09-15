<template>
  <div class="slidev-layout" :class="{ grid: $attrs.center }">
    <glass class="my-auto z-10 relative w-full h-full">
        <div class="grid grid-cols-2 gap-16 content-center w-full h-full">
      <div class="prose">
        <slot name="default" />
      </div>
      <div class="prose">
        <slot name="right" />
      </div>
    </div>
    </glass>
  </div>
</template>
