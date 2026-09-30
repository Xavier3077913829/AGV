<script setup>
import { computed } from 'vue'

const props = defineProps({
  result: { type: Object, default: null },
})

const evaluation = computed(() => props.result?.evaluation || {})
const baseline = computed(() => props.result?.baseline || {})
const taskMap = computed(() => Object.fromEntries((evaluation.value.tasks || []).map((task) => [task.task_id, task])))
const groups = computed(() => (evaluation.value.agv_summary || []).map((summary) => ({
  ...summary,
  tasks: (summary.task_ids || []).map((taskId) => taskMap.value[taskId]).filter(Boolean),
})))

const timeRange = computed(() => {
  const starts = groups.value.map((group) => new Date(group.start_time).getTime())
  const ends = groups.value.map((group) => new Date(group.end_time).getTime())
  const min = Math.min(...starts, Date.now())
  const max = Math.max(...ends, min + 1)
  return { min, max: Math.max(max, min + 1), duration: Math.max(max - min, 1) }
})

function barStyle(task) {
  const start = new Date(task.start_time).getTime()
  const end = new Date(task.end_time).getTime()
  const left = ((start - timeRange.value.min) / timeRange.value.duration) * 100
  const width = Math.max(((end - start) / timeRange.value.duration) * 100, 2)
  return { left: `${left}%`, width: `${width}%` }
}

function seconds(value) {
  return `${Number(value || 0).toFixed(1)}s`
}
</script>

<template>
  <section v-if="result" class="panel mt-16 optimization-result">
    <div class="panel-head">
      <div><div class="panel-title">优化结果与调度甘特图</div><div class="panel-desc">{{ result.run.objective_name }}</div></div>
      <span class="tag">{{ result.run.algorithm_display }}</span>
    </div>
    <div class="panel-body">
      <div class="grid metric-grid">
        <div class="optimization-metric"><span>任务数量</span><strong>{{ result.run.task_count }}</strong></div>
        <div class="optimization-metric"><span>使用 AGV</span><strong>{{ result.run.agv_count }} 台</strong></div>
        <div class="optimization-metric"><span>完工时间</span><strong>{{ seconds(result.run.makespan_seconds) }}</strong></div>
        <div class="optimization-metric"><span>总里程</span><strong>{{ Number(result.run.total_distance).toFixed(1) }} m</strong></div>
        <div class="optimization-metric"><span>避碰等待</span><strong>{{ seconds(result.run.total_wait_seconds) }}</strong></div>
        <div class="optimization-metric highlight"><span>综合目标改进</span><strong>{{ result.run.improvement_percent }}%</strong></div>
      </div>
      <div class="comparison-strip">
        <div><span class="muted">基准 makespan</span><strong>{{ seconds(baseline.makespan_seconds || result.run.baseline_makespan_seconds) }}</strong></div>
        <div><span class="muted">优化 makespan</span><strong>{{ seconds(result.run.makespan_seconds) }}</strong></div>
        <div><span class="muted">基准目标值</span><strong>{{ baseline.objective || result.run.metrics.baseline_objective }}</strong></div>
        <div><span class="muted">优化目标值</span><strong>{{ result.evaluation.objective }}</strong></div>
        <div><span class="muted">搜索迭代</span><strong>{{ result.run.iteration_count }} 次</strong></div>
      </div>
      <div class="gantt mt-16">
        <div v-for="group in groups" :key="group.agv_id" class="gantt-row">
          <div class="gantt-label"><strong>{{ group.agv_code }}</strong><span>{{ group.tasks.length }} 个任务</span></div>
          <div class="gantt-track">
            <div v-for="task in group.tasks" :key="task.task_id" class="gantt-bar" :style="barStyle(task)" :title="`${task.task_no} · ${seconds(task.duration_seconds)} · ${task.distance}m`"><span>{{ task.task_no }}</span></div>
          </div>
          <div class="gantt-end">{{ seconds((new Date(group.end_time) - new Date(group.start_time)) / 1000) }}</div>
        </div>
      </div>
    </div>
  </section>
</template>
