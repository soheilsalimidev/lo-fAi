<template>
  <div class="h-full flex flex-col">
    <div class="flex item-center  justify-center gap-2 h-11/12 p-2">
      <slot></slot>
    </div>
    <p class="text-center !m-0" :class="cla">
      {{ caption }}
    </p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  caption: string
  cla?: string
}>()
</script>