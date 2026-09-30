<script setup>
defineProps({
  status: { type: String, default: 'idle' },
  label: { type: String, default: '' },
})

const names = {
  idle: '空闲', busy: '执行任务', charging: '充电中', offline: '离线', error: '故障',
  pending: '待调度', assigned: '已分配', in_progress: '执行中', completed: '已完成',
  failed: '失败', cancelled: '已取消',
}
</script>

<template>
  <span class="badge" :class="`badge-${status}`">{{ label || names[status] || status }}</span>
</template>
