<script setup>
import { computed, ref } from 'vue'
import { api } from '../api'
import StatusBadge from '../components/StatusBadge.vue'

const props = defineProps({
  tasks: { type: Array, default: () => [] },
  nodes: { type: Array, default: () => [] },
})
const emit = defineEmits(['refresh', 'notify'])
const query = ref('')
const statusFilter = ref('')
const expanded = ref(null)
const busyId = ref(null)

const nodeMap = computed(() => Object.fromEntries(props.nodes.map((node) => [node.id, node])))
const filtered = computed(() => {
  const keyword = query.value.trim().toLowerCase()
  return props.tasks.filter((task) => {
    const matchesText = !keyword || `${task.task_no} ${task.cargo_name} ${task.assigned_agv_detail?.code || ''}`.toLowerCase().includes(keyword)
    const matchesStatus = !statusFilter.value || task.status === statusFilter.value
    return matchesText && matchesStatus
  })
})

function routeText(task) {
  return (task.route?.nodes || []).map((id) => nodeMap.value[id]?.code || id).join(' → ')
}

async function runAction(task, action) {
  busyId.value = `${task.id}-${action}`
  try {
    await api.taskAction(task.id, action)
    const messages = { start: '任务已开始执行', complete: '任务已完成，AGV 位置和电量已更新', cancel: '任务已取消' }
    emit('notify', { message: messages[action] })
    emit('refresh')
  } catch (error) {
    emit('notify', { message: error.message, type: 'error' })
  } finally {
    busyId.value = null
  }
}
</script>

<template>
  <section class="panel">
    <div class="panel-head">
      <div><div class="panel-title">运输任务中心</div><div class="panel-desc">管理任务全生命周期，查看调度路线和避碰等待信息</div></div>
      <div class="toolbar-right">
        <input v-model="query" class="input filter-input" placeholder="搜索任务、货物或 AGV..." />
        <select v-model="statusFilter" class="select" style="width: 130px">
          <option value="">全部状态</option><option value="pending">待调度</option><option value="assigned">已分配</option><option value="in_progress">执行中</option><option value="completed">已完成</option><option value="cancelled">已取消</option>
        </select>
      </div>
    </div>
    <div class="panel-body">
      <div class="table-wrap">
        <table>
          <thead><tr><th>任务</th><th>路线</th><th>货物</th><th>优先级</th><th>AGV</th><th>里程/耗时</th><th>状态</th><th>操作</th></tr></thead>
          <tbody>
            <template v-for="task in filtered" :key="task.id">
              <tr>
                <td><div class="cell-main mono">{{ task.task_no }} <span v-if="task.sequence_number" class="tag">#{{ task.sequence_number }}</span></div><div class="cell-sub">{{ task.schedule_run ? `调度批次 ${task.schedule_run}` : new Date(task.created_at).toLocaleString('zh-CN', { hour12: false }) }}</div></td>
                <td><span class="strong">{{ task.pickup_node_detail?.code }}</span> → <span class="strong">{{ task.dropoff_node_detail?.code }}</span><button v-if="task.route?.nodes?.length" class="link-btn" @click="expanded = expanded === task.id ? null : task.id">路线</button></td>
                <td>{{ task.cargo_name }}<div class="cell-sub">{{ task.cargo_weight }} kg</div></td>
                <td :class="`priority-${task.priority}`">{{ task.priority_display }}</td>
                <td>{{ task.assigned_agv_detail?.code || '--' }}</td>
                <td>{{ task.total_distance || 0 }} m<div class="cell-sub">{{ task.estimated_duration || 0 }} s</div></td>
                <td><StatusBadge :status="task.status" /></td>
                <td>
                  <div class="table-actions">
                    <button v-if="task.status === 'assigned'" class="btn btn-sm btn-primary" :disabled="busyId === `${task.id}-start`" @click="runAction(task, 'start')">开始</button>
                    <button v-if="task.status === 'in_progress'" class="btn btn-sm btn-success" :disabled="busyId === `${task.id}-complete`" @click="runAction(task, 'complete')">完成</button>
                    <button v-if="['pending', 'assigned'].includes(task.status)" class="btn btn-sm btn-danger" :disabled="busyId === `${task.id}-cancel`" @click="runAction(task, 'cancel')">取消</button>
                    <span v-if="['completed', 'cancelled'].includes(task.status)" class="muted small">--</span>
                  </div>
                </td>
              </tr>
              <tr v-if="expanded === task.id">
                <td colspan="8">
                  <div class="route-detail">
                    <div><span class="muted">规划路线：</span><span class="mono">{{ routeText(task) || '暂无路线' }}</span></div>
                    <div class="mt-12"><span class="muted">预计时间：</span>{{ task.scheduled_start ? new Date(task.scheduled_start).toLocaleTimeString('zh-CN', { hour12: false }) : '--' }} - {{ task.scheduled_end ? new Date(task.scheduled_end).toLocaleTimeString('zh-CN', { hour12: false }) : '--' }}</div>
                    <div v-if="task.route?.waits?.length" class="notice warning mt-12"><span>!</span><div>检测到 {{ task.route.waits.length }} 次节点避碰等待，已自动加入时间窗约束。</div></div>
                  </div>
                </td>
              </tr>
            </template>
            <tr v-if="!filtered.length"><td colspan="8"><div class="empty">没有匹配的任务</div></td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>
</template>
