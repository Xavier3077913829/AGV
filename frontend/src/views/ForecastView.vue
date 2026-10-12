<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { api } from '../api'

const emit = defineEmits(['notify'])

const MODEL_LABELS = {
  auto: '自动选择',
  moving_average: '移动平均',
  holt_winters: 'Holt-Winters',
  arima: 'ARIMA',
}

const subTab = ref('predict')
const sourceKind = ref('dataset')
const model = ref('auto')
const horizon = ref(14)
const file = ref(null)
const objectUrl = ref('')
const result = ref(null)
const overview = ref({})
const runs = ref([])
const running = ref(false)
const loadingRuns = ref(false)
const error = ref('')

const selectedMetrics = computed(() => result.value?.metrics || {})
const candidates = computed(() => {
  const metrics = result.value?.candidate_metrics || {}
  return Object.entries(metrics).map(([key, value]) => ({
    key,
    label: MODEL_LABELS[key] || key,
    ...value,
  }))
})

const chart = computed(() => {
  const history = (result.value?.history || []).slice(-120)
  const forecast = result.value?.forecast || []
  if (!history.length || !forecast.length) {
    return { historyPoints: '', forecastPoints: '', width: 900, height: 280, ticks: [], historyEndX: 0 }
  }

  const width = 900
  const height = 280
  const padding = 42
  const allValues = [...history, ...forecast].map((item) => Number(item.order_count))
  const minimum = Math.min(...allValues)
  const maximum = Math.max(...allValues)
  const span = maximum - minimum || 1
  const chartMin = minimum - span * 0.12
  const chartMax = maximum + span * 0.12
  const total = history.length + forecast.length
  const xAt = (index) => padding + (index / (total - 1)) * (width - padding * 2)
  const yAt = (value) =>
    height - padding - ((value - chartMin) / (chartMax - chartMin)) * (height - padding * 2)

  const historyPoints = history
    .map((item, index) => `${xAt(index).toFixed(2)},${yAt(Number(item.order_count)).toFixed(2)}`)
    .join(' ')
  const forecastPoints = [
    `${xAt(history.length - 1).toFixed(2)},${yAt(Number(history.at(-1).order_count)).toFixed(2)}`,
    ...forecast.map((item, index) =>
      `${xAt(history.length + index).toFixed(2)},${yAt(Number(item.order_count)).toFixed(2)}`,
    ),
  ].join(' ')
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => ({
    y: padding + ratio * (height - padding * 2),
    value: chartMax - ratio * (chartMax - chartMin),
  }))

  return {
    historyPoints,
    forecastPoints,
    width,
    height,
    ticks,
    historyEndX: xAt(history.length - 1),
    startLabel: history[0]?.date,
    endLabel: forecast.at(-1)?.date,
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
    overview.value = await api.forecastOverview()
  } catch (err) {
    notify(`预测概览加载失败：${err.message}`, 'error')
  }
}

async function loadRuns() {
  loadingRuns.value = true
  try {
    runs.value = await api.forecastRuns()
  } catch (err) {
    notify(`预测记录加载失败：${err.message}`, 'error')
  } finally {
    loadingRuns.value = false
  }
}

async function runForecast() {
  running.value = true
  error.value = ''
  try {
    let payload
    if (sourceKind.value === 'upload') {
      if (!file.value) throw new Error('请先选择订单 CSV 文件')
      payload = new FormData()
      payload.append('file', file.value)
      payload.append('horizon', String(horizon.value))
      payload.append('model', model.value)
    } else {
      payload = { horizon: Number(horizon.value), model: model.value }
    }
    result.value = await api.forecastPredict(payload)
    notify(`预测完成，采用 ${result.value.selected_model_label}`)
    await loadOverview()
    if (subTab.value === 'records') await loadRuns()
  } catch (err) {
    error.value = err.message
    notify(err.message, 'error')
  } finally {
    running.value = false
  }
}

function openRuns() {
  subTab.value = 'records'
  loadRuns()
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
        <div class="panel-title">订单需求预测</div>
        <div class="panel-desc">任务 7：比较移动平均、Holt-Winters 与 ARIMA，并预测未来订单量</div>
      </div>
      <div class="toolbar-right">
        <button class="btn btn-sm" :class="{ 'btn-primary': subTab === 'predict' }" @click="subTab = 'predict'">在线预测</button>
        <button class="btn btn-sm" :class="{ 'btn-primary': subTab === 'records' }" @click="openRuns">预测记录</button>
      </div>
    </div>

    <div class="panel-body">
      <div class="grid stats-grid">
        <div class="mini-stat panel"><span>累计预测</span><strong>{{ overview.total || 0 }}</strong></div>
        <div class="mini-stat panel"><span>平均 MAPE</span><strong>{{ overview.average_mape || 0 }}%</strong></div>
        <div class="mini-stat panel"><span>历史预测总量</span><strong>{{ Math.round(overview.predicted_total || 0) }}</strong></div>
        <div class="mini-stat panel"><span>候选模型</span><strong>3</strong></div>
      </div>

      <div v-if="subTab === 'predict'" class="grid forecast-grid mt-16">
        <section class="panel">
          <div class="panel-head"><div><div class="panel-title">预测配置</div><div class="panel-desc">使用内置订单序列或上传 date/order_count CSV</div></div></div>
          <div class="panel-body">
            <div class="form-grid">
              <div class="field">
                <label>数据来源</label>
                <select v-model="sourceKind" class="select">
                  <option value="dataset">内置订单样例</option>
                  <option value="upload">上传 CSV</option>
                </select>
              </div>
              <div class="field">
                <label>预测算法</label>
                <select v-model="model" class="select">
                  <option value="auto">自动选择最优模型</option>
                  <option value="moving_average">移动平均</option>
                  <option value="holt_winters">Holt-Winters</option>
                  <option value="arima">ARIMA</option>
                </select>
              </div>
              <div class="field">
                <label>预测期数</label>
                <input v-model.number="horizon" class="input" type="number" min="1" max="120" />
              </div>
              <div v-if="sourceKind === 'upload'" class="field full">
                <label>订单 CSV</label>
                <input class="input" type="file" accept=".csv,text/csv" @change="onFileChange" />
              </div>
            </div>

            <div class="notice mt-12">
              必需列：<span class="mono">date, order_count</span>
            </div>
            <div class="toolbar-row mt-16">
              <button class="btn btn-primary" :disabled="running" @click="runForecast">
                {{ running ? '正在预测...' : '开始预测' }}
              </button>
              <span class="muted small">自动模式会按回测 RMSE 选择最佳模型</span>
            </div>
            <div v-if="error" class="notice warning mt-12">{{ error }}</div>
          </div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <div><div class="panel-title">预测结果</div><div class="panel-desc">历史订单与未来预测曲线</div></div>
            <span v-if="result" class="tag">{{ result.selected_model_label }}</span>
          </div>
          <div class="panel-body">
            <div v-if="result" class="grid metric-grid">
              <div class="optimization-metric"><span>预测模型</span><strong>{{ result.selected_model_label }}</strong></div>
              <div class="optimization-metric"><span>预测总量</span><strong>{{ Math.round(result.predicted_total) }}</strong></div>
              <div class="optimization-metric"><span>MAE</span><strong>{{ selectedMetrics.mae }}</strong></div>
              <div class="optimization-metric"><span>RMSE</span><strong>{{ selectedMetrics.rmse }}</strong></div>
              <div class="optimization-metric"><span>MAPE</span><strong>{{ selectedMetrics.mape }}%</strong></div>
              <div class="optimization-metric"><span>预测期数</span><strong>{{ result.horizon }}</strong></div>
            </div>
            <div v-else class="empty">请选择内置数据或上传 CSV 后开始预测</div>

            <template v-if="result">
              <svg class="forecast-chart" :viewBox="`0 0 ${chart.width} ${chart.height}`" role="img" aria-label="订单预测曲线">
                <line
                  v-for="tick in chart.ticks"
                  :key="tick.y"
                  x1="42"
                  :x2="chart.width - 42"
                  :y1="tick.y"
                  :y2="tick.y"
                  class="chart-grid"
                />
                <polyline :points="chart.historyPoints" fill="none" stroke="#0e8792" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
                <polyline :points="chart.forecastPoints" fill="none" stroke="#cf5260" stroke-width="4" stroke-dasharray="9 7" stroke-linecap="round" stroke-linejoin="round" />
                <line :x1="chart.historyEndX" :x2="chart.historyEndX" y1="34" :y2="chart.height - 34" class="forecast-divider" />
              </svg>
              <div class="forecast-axis">
                <span>{{ chart.startLabel }}</span>
                <span>未来预测</span>
                <span>{{ chart.endLabel }}</span>
              </div>

              <div class="table-wrap mt-16">
                <table>
                  <thead><tr><th>模型</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>状态</th></tr></thead>
                  <tbody>
                    <tr v-for="item in candidates" :key="item.key">
                      <td class="cell-main">{{ item.label }}</td>
                      <td>{{ item.mae ?? '--' }}</td>
                      <td>{{ item.rmse ?? '--' }}</td>
                      <td>{{ item.mape != null ? `${item.mape}%` : '--' }}</td>
                      <td>
                        <span v-if="item.key === result.selected_model" class="badge badge-completed">已采用</span>
                        <span v-else-if="item.error" class="badge badge-failed">失败</span>
                        <span v-else class="tag">对比模型</span>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </template>
          </div>
        </section>
      </div>

      <div v-else class="mt-16">
        <div class="section-toolbar">
          <span class="muted small">共 {{ runs.length }} 条预测记录</span>
          <button class="btn btn-sm" :disabled="loadingRuns" @click="loadRuns">{{ loadingRuns ? '加载中...' : '刷新记录' }}</button>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>ID</th><th>请求模型</th><th>采用模型</th><th>来源</th><th>历史量</th><th>预测期数</th><th>预测总量</th><th>MAPE</th><th>时间</th></tr></thead>
            <tbody>
              <tr v-for="run in runs" :key="run.id">
                <td class="mono">{{ run.id }}</td>
                <td>{{ run.model_display }}</td>
                <td class="cell-main">{{ MODEL_LABELS[run.selected_model] || run.selected_model }}</td>
                <td>{{ run.source_display }}</td>
                <td>{{ run.history_count }}</td>
                <td>{{ run.horizon }}</td>
                <td>{{ Math.round(run.predicted_total) }}</td>
                <td>{{ run.mape }}%</td>
                <td class="small">{{ formatTime(run.created_at) }}</td>
              </tr>
              <tr v-if="!runs.length"><td colspan="9"><div class="empty">暂无预测记录</div></td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.forecast-grid { grid-template-columns: minmax(340px, 0.78fr) minmax(0, 1.52fr); align-items: start; }
.forecast-chart { width: 100%; min-height: 280px; margin-top: 4px; border-radius: 10px; background: #f8faf9; border: 1px solid #e1e8e5; }
.chart-grid { stroke: #e2e9e6; stroke-width: 1; }
.forecast-divider { stroke: #b8c7c1; stroke-width: 1; stroke-dasharray: 4 5; }
.forecast-axis { display: flex; justify-content: space-between; padding: 6px 42px 0; color: var(--muted); font-size: 10px; }
@media (max-width: 980px) { .forecast-grid { grid-template-columns: minmax(0, 1fr); } }
</style>
