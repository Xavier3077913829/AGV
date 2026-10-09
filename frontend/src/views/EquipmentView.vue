<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { api } from '../api'

const emit = defineEmits(['notify'])

const FEATURES = [
  { key: 'temperature_c', label: '温度 ℃', threshold: 74, color: '#d97934' },
  { key: 'vibration_mm_s', label: '振动 mm/s', threshold: 3.1, color: '#cf5260' },
  { key: 'current_a', label: '电流 A', threshold: 6.2, color: '#4f6fd1' },
  { key: 'spindle_speed_rpm', label: '主轴转速 rpm', threshold: null, color: '#2b9a76' },
  { key: 'tool_wear', label: '刀具磨损', threshold: 0.78, color: '#8b5bc7' },
]
const STATE_LABELS = { normal: '正常', worn: '磨损', fault: '故障' }
const STATE_COLORS = { normal: '#2b9a76', worn: '#d2902e', fault: '#cf5260' }

const subTab = ref('diagnose')
const sourceKind = ref('dataset')
const scenario = ref('worn')
const equipmentCode = ref('CNC-01')
const file = ref(null)
const objectUrl = ref('')
const selectedFeature = ref('vibration_mm_s')
const result = ref(null)
const overview = ref({})
const records = ref([])
const running = ref(false)
const loadingRecords = ref(false)
const error = ref('')

const feature = computed(
  () => FEATURES.find((item) => item.key === selectedFeature.value) || FEATURES[0],
)
const series = computed(() => result.value?.series || [])
const stateRows = computed(() => {
  const distribution = result.value?.state_distribution || {}
  return Object.entries(distribution).map(([key, value]) => ({
    key,
    label: value.label || STATE_LABELS[key] || key,
    count: value.count || 0,
    ratio: value.ratio || 0,
    color: STATE_COLORS[key] || '#89979d',
  }))
})

const lineChart = computed(() => {
  const width = 760
  const height = 250
  const padding = 36
  const values = series.value.map((row) => Number(row[selectedFeature.value]))
  if (values.length < 2) {
    return { points: '', width, height, min: 0, max: 1, ticks: [] }
  }

  const minimum = Math.min(...values)
  const maximum = Math.max(...values)
  const span = maximum - minimum || 1
  const chartMin = minimum - span * 0.1
  const chartMax = maximum + span * 0.1
  const pointList = values.map((value, index) => {
    const x = padding + (index / (values.length - 1)) * (width - padding * 2)
    const y =
      height -
      padding -
      ((value - chartMin) / (chartMax - chartMin)) * (height - padding * 2)
    return `${x.toFixed(2)},${y.toFixed(2)}`
  })
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => ({
    y: padding + ratio * (height - padding * 2),
    value: chartMax - ratio * (chartMax - chartMin),
  }))
  return {
    points: pointList.join(' '),
    width,
    height,
    min: chartMin,
    max: chartMax,
    ticks,
  }
})

function notify(message, type = 'success') {
  emit('notify', { message, type })
}

function formatTime(value) {
  if (!value) return '--'
  return String(value).replace('T', ' ').slice(0, 19)
}

function onFileChange(event) {
  const picked = event.target.files && event.target.files[0]
  if (objectUrl.value) URL.revokeObjectURL(objectUrl.value)
  file.value = picked || null
  objectUrl.value = picked ? URL.createObjectURL(picked) : ''
  result.value = null
  error.value = ''
}

async function loadOverview() {
  try {
    overview.value = await api.equipmentOverview()
  } catch (err) {
    notify(`设备概览加载失败：${err.message}`, 'error')
  }
}

async function loadRecords() {
  loadingRecords.value = true
  try {
    records.value = await api.equipmentRecords()
  } catch (err) {
    notify(`诊断记录加载失败：${err.message}`, 'error')
  } finally {
    loadingRecords.value = false
  }
}

async function runDiagnosis() {
  running.value = true
  error.value = ''
  try {
    let payload
    if (sourceKind.value === 'upload') {
      if (!file.value) throw new Error('请先选择传感器 CSV 文件')
      payload = new FormData()
      payload.append('file', file.value)
      payload.append('equipment_code', equipmentCode.value || 'CNC-01')
    } else {
      payload = {
        sample: scenario.value,
        equipment_code: equipmentCode.value || 'CNC-01',
      }
    }
    result.value = await api.equipmentDiagnose(payload)
    notify(`设备诊断完成：${result.value.predicted_state_label}`)
    await loadOverview()
    if (subTab.value === 'records') await loadRecords()
  } catch (err) {
    error.value = err.message
    notify(err.message, 'error')
  } finally {
    running.value = false
  }
}

function openRecords() {
  subTab.value = 'records'
  loadRecords()
}

onMounted(() => {
  loadOverview()
})

onUnmounted(() => {
  if (objectUrl.value) URL.revokeObjectURL(objectUrl.value)
})
</script>

<template>
  <section class="panel">
    <div class="panel-head">
      <div>
        <div class="panel-title">设备状态监测</div>
        <div class="panel-desc">任务 6：对温度、振动、电流、转速和刀具磨损进行状态分类与规则告警</div>
      </div>
      <div class="toolbar-right">
        <button class="btn btn-sm" :class="{ 'btn-primary': subTab === 'diagnose' }" @click="subTab = 'diagnose'">在线诊断</button>
        <button class="btn btn-sm" :class="{ 'btn-primary': subTab === 'records' }" @click="openRecords">诊断记录</button>
      </div>
    </div>

    <div class="panel-body">
      <div class="grid stats-grid">
        <div class="mini-stat panel"><span>累计诊断</span><strong>{{ overview.total || 0 }}</strong></div>
        <div class="mini-stat panel"><span>平均置信度</span><strong>{{ ((overview.average_confidence || 0) * 100).toFixed(1) }}%</strong></div>
        <div class="mini-stat panel"><span>模型准确率</span><strong>{{ ((overview.model_metrics?.accuracy || 0) * 100).toFixed(1) }}%</strong></div>
        <div class="mini-stat panel"><span>状态类别</span><strong>3</strong></div>
      </div>

      <div v-if="subTab === 'diagnose'" class="grid equipment-grid mt-16">
        <section class="panel">
          <div class="panel-head"><div><div class="panel-title">数据配置</div><div class="panel-desc">使用内置样例或上传带传感器列的 CSV</div></div></div>
          <div class="panel-body">
            <div class="form-grid">
              <div class="field">
                <label>设备编号</label>
                <input v-model="equipmentCode" class="input" type="text" />
              </div>
              <div class="field">
                <label>数据来源</label>
                <select v-model="sourceKind" class="select">
                  <option value="dataset">内置样例</option>
                  <option value="upload">上传 CSV</option>
                </select>
              </div>
              <div v-if="sourceKind === 'dataset'" class="field">
                <label>设备场景</label>
                <select v-model="scenario" class="select">
                  <option value="normal">正常运行</option>
                  <option value="worn">刀具磨损</option>
                  <option value="fault">设备故障</option>
                </select>
              </div>
              <div v-else class="field full">
                <label>传感器 CSV（温度、振动、电流、转速、磨损列）</label>
                <input class="input" type="file" accept=".csv,text/csv" @change="onFileChange" />
              </div>
            </div>

            <div class="notice mt-12">
              必需列：<span class="mono">temperature_c, vibration_mm_s, current_a, spindle_speed_rpm, tool_wear</span>
            </div>
            <div class="toolbar-row mt-16">
              <button class="btn btn-primary" :disabled="running" @click="runDiagnosis">
                {{ running ? '正在诊断...' : '开始诊断' }}
              </button>
              <span class="muted small">模型：RandomForest + 规则告警</span>
            </div>
            <div v-if="error" class="notice warning mt-12">{{ error }}</div>
          </div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <div><div class="panel-title">诊断结果</div><div class="panel-desc">当前状态、传感器趋势与规则告警</div></div>
            <span v-if="result" class="tag">{{ result.predicted_state_label }}</span>
          </div>
          <div class="panel-body">
            <div v-if="result" class="grid metric-grid">
              <div class="optimization-metric"><span>综合状态</span><strong :style="{ color: STATE_COLORS[result.predicted_state] }">{{ result.predicted_state_label }}</strong></div>
              <div class="optimization-metric"><span>样本数量</span><strong>{{ result.sample_count }}</strong></div>
              <div class="optimization-metric"><span>末样本置信度</span><strong>{{ (result.confidence * 100).toFixed(1) }}%</strong></div>
              <div class="optimization-metric"><span>规则告警</span><strong>{{ result.rule_alerts.length }}</strong></div>
            </div>
            <div v-else class="empty">请选择样例或上传 CSV 后开始诊断</div>

            <template v-if="result">
              <div class="section-toolbar mt-16">
                <select v-model="selectedFeature" class="select">
                  <option v-for="item in FEATURES" :key="item.key" :value="item.key">{{ item.label }}</option>
                </select>
                <span class="muted small">展示最近 {{ series.length }} 个采样点</span>
              </div>
              <svg class="equipment-chart" :viewBox="`0 0 ${lineChart.width} ${lineChart.height}`" role="img" aria-label="传感器趋势图">
                <line
                  v-for="tick in lineChart.ticks"
                  :key="tick.y"
                  x1="36"
                  :x2="lineChart.width - 36"
                  :y1="tick.y"
                  :y2="tick.y"
                  class="chart-grid"
                />
                <polyline :points="lineChart.points" fill="none" :stroke="feature.color" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
              </svg>

              <div class="state-bars mt-12">
                <div v-for="item in stateRows" :key="item.key" class="state-bar-row">
                  <span>{{ item.label }}</span>
                  <div class="progress-track"><div class="progress-bar" :style="{ width: `${item.ratio * 100}%`, background: item.color }"></div></div>
                  <strong>{{ item.count }}</strong>
                </div>
              </div>

              <div v-if="result.rule_alerts.length" class="notice warning mt-16">
                <strong>规则告警：</strong>
                <span>{{ result.rule_alerts.map((item) => item.message).join('；') }}</span>
              </div>
            </template>
          </div>
        </section>
      </div>

      <div v-else class="mt-16">
        <div class="section-toolbar">
          <span class="muted small">共 {{ records.length }} 条诊断记录</span>
          <button class="btn btn-sm" :disabled="loadingRecords" @click="loadRecords">{{ loadingRecords ? '加载中...' : '刷新记录' }}</button>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>ID</th><th>设备</th><th>来源</th><th>样本数</th><th>状态</th><th>置信度</th><th>告警</th><th>时间</th></tr></thead>
            <tbody>
              <tr v-for="record in records" :key="record.id">
                <td class="mono">{{ record.id }}</td>
                <td class="cell-main">{{ record.equipment_code }}</td>
                <td>{{ record.source_display }}</td>
                <td>{{ record.sample_count }}</td>
                <td><span class="badge" :style="{ color: STATE_COLORS[record.predicted_state] }">{{ record.state_display }}</span></td>
                <td>{{ (record.confidence * 100).toFixed(1) }}%</td>
                <td>{{ record.rule_alerts.length }}</td>
                <td class="small">{{ formatTime(record.created_at) }}</td>
              </tr>
              <tr v-if="!records.length"><td colspan="8"><div class="empty">暂无诊断记录</div></td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.equipment-grid { grid-template-columns: minmax(340px, 0.85fr) minmax(0, 1.35fr); align-items: start; }
.equipment-chart { width: 100%; min-height: 250px; margin-top: 4px; border-radius: 10px; background: #f8faf9; border: 1px solid #e1e8e5; }
.chart-grid { stroke: #e2e9e6; stroke-width: 1; }
.state-bars { display: grid; gap: 9px; }
.state-bar-row { display: grid; grid-template-columns: 56px minmax(0, 1fr) 40px; gap: 10px; align-items: center; color: #5f7179; font-size: 11px; }
@media (max-width: 980px) { .equipment-grid { grid-template-columns: minmax(0, 1fr); } }
</style>
