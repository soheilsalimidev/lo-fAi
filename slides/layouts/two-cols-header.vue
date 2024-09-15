<template>
  <div class="slidev-layout" :class="{ grid: $attrs.center }">
    <glass class="my-auto z-10 relative w-full h-full">
        <slot name="default" />
        <div class="grid grid-cols-2 gap-8 content-center w-full -mt-10">
      <div class="prose me-2">
        <slot name="left" />
      </div>
      <div class="prose ms-2">
        <slot name="right" />
      </div>
    </div>
    </glass>
  </div>
</template>
