<template>
  <div class="slidev-layout" :class="{ grid: $attrs.center }">
    <glass class="my-auto z-10 relative w-full h-full">
      <slot />
    </glass>
  </div>
</template>
