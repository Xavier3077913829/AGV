<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { api } from '../api'

const emit = defineEmits(['notify'])

const MODES = [
  { key: 'crack', label: '表面裂纹检测', groups: ['crack', 'free'] },
  { key: 'defect', label: '缺陷类型识别（NEU 6类）', groups: ['neu'], requires: 'defect' },
  { key: 'workpiece', label: '工件识别', groups: ['workpiece'] },
]
const SHAPE_LABELS = {
  circle: '圆形', triangle: '三角形', square: '正方形',
  rectangle: '矩形', hexagon: '六边形', unknown: '未识别',
}

const subTab = ref('detect')
const mode = ref('defect')
const sourceKind = ref('upload')
const sampleGroup = ref('neu')
const sampleName = ref('')
const file = ref(null)
const objectUrl = ref('')

const samples = ref({ crack: [], free: [], workpiece: [] })
const datasetSummary = ref({})
const overview = ref({})
const algoStatus = ref(null)
const records = ref([])
const recordFilter = ref('')
const recordCount = ref(0)
const recordPage = ref(1)
const recordHasNext = ref(false)
const recordHasPrevious = ref(false)
const recordDetail = ref(null)
const recordDetailLoading = ref(false)
const dragging = ref(false)

const running = ref(false)
const loadingRecords = ref(false)
const error = ref('')
const result = ref(null)

const crackAlgorithm = ref('frangi')
const deepcrackParams = ref({ algorithm: 'deepcrack', threshold: 0.72, min_area: 8, min_coverage: 0.0002, max_side: 512 })
const frangiParams = ref({
  algorithm: 'frangi',
  score_threshold: 0.54,
  vessel_threshold: 0.55,
  min_aspect: 2.5,
  mask_threshold_ratio: 0.575,
  mask_close_kernel: 3,
  mask_skeleton: false,
  mask_dilation: 3,
})
const workpieceParams = ref({ min_area: 300, approx_epsilon: 0.02, morph_kernel: 5 })

const currentMode = computed(() => MODES.find((item) => item.key === mode.value) || MODES[0])
const defectAvailable = computed(() => Boolean(algoStatus.value?.defect?.available))
const defectAccuracy = computed(() => algoStatus.value?.defect?.test_accuracy)
const defectAccuracyText = computed(() =>
  defectAccuracy.value ? `${(defectAccuracy.value * 100).toFixed(1)}%` : '未训练')
const deepcrackAvailable = computed(() => Boolean(algoStatus.value?.deepcrack?.weights_found))
const modeAvailable = (item) => !item.requires || Boolean(algoStatus.value?.[item.requires]?.available)
const currentModeReady = computed(() => {
  if (!modeAvailable(currentMode.value)) return false
  if (mode.value === 'crack' && crackAlgorithm.value === 'deepcrack') return deepcrackAvailable.value
  return true
})
const groups = computed(() => currentMode.value.groups)
const activeSamples = computed(() => samples.value[sampleGroup.value] || [])
const activeParams = computed(() => {
  if (mode.value === 'defect') return {}
  if (mode.value === 'crack') {
    return crackAlgorithm.value === 'deepcrack' ? deepcrackParams.value : frangiParams.value
  }
  return workpieceParams.value
})
const canRun = computed(() => currentModeReady.value
  && (sourceKind.value === 'upload' ? !!file.value : !!sampleName.value))
const datasetCounts = computed(() => {
  const summary = datasetSummary.value || {}
  return `${summary.crack || 0} / ${summary.free || 0} / ${summary.workpiece || 0} / ${summary.neu || 0}`
})
const datasetNote = computed(() => {
  const summary = datasetSummary.value || {}
  return `数据集：MT_Crack ${summary.crack || 0} 张（含标注掩码）· MT_Free 无缺陷 ${summary.free || 0} 张 · 合成几何工件 ${summary.workpiece || 0} 张 · NEU 六类样例 ${summary.neu || 0} 张`
})
const previewUrl = computed(() => {
  if (sourceKind.value === 'upload') return objectUrl.value
  if (!sampleName.value) return ''
  return api.visionSampleImageUrl(sampleGroup.value, sampleName.value)
})
const objects = computed(() => result.value?.objects || [])
const recordObjects = computed(() => recordDetail.value?.objects || [])

watch(mode, () => {
  sampleGroup.value = groups.value[0]
  sampleName.value = (samples.value[sampleGroup.value] || [])[0] || ''
  result.value = null
  error.value = ''
})
watch(sampleGroup, (group) => {
  sampleName.value = (samples.value[group] || [])[0] || ''
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
  setFile(picked)
}

function setFile(picked) {
  if (objectUrl.value) URL.revokeObjectURL(objectUrl.value)
  file.value = picked || null
  objectUrl.value = picked ? URL.createObjectURL(picked) : ''
  sourceKind.value = 'upload'
  result.value = null
  error.value = ''
}

function onDrop(event) {
  dragging.value = false
  const picked = event.dataTransfer?.files?.[0]
  if (!picked) return
  if (!picked.type.startsWith('image/')) {
    notify('请选择 jpg、png、bmp 或 webp 图片', 'error')
    return
  }
  setFile(picked)
}

function resetForm() {
  file.value = null
  if (objectUrl.value) URL.revokeObjectURL(objectUrl.value)
  objectUrl.value = ''
  result.value = null
  error.value = ''
}

async function loadSamples() {
  try {
    const data = await api.visionSamples()
    samples.value = data.samples || {}
    datasetSummary.value = data.summary || {}
    if (!activeSamples.value.length) {
      const first = groups.value.find((group) => (samples.value[group] || []).length)
      if (first) sampleGroup.value = first
    }
    if (!sampleName.value) sampleName.value = activeSamples.value[0] || ''
  } catch (err) {
    notify(`数据集样例加载失败：${err.message}`, 'error')
  }
}

async function loadOverview() {
  try {
    overview.value = await api.visionOverview()
  } catch (err) {
    notify(`视觉统计加载失败：${err.message}`, 'error')
  }
}

async function loadAlgoStatus() {
  try {
    algoStatus.value = await api.visionAlgoStatus()
  } catch {
    // 状态接口不可用时按“都未就绪”处理，避免误启用需要权重的任务。
    algoStatus.value = null
  }
}

async function loadRecords(page = 1) {
  loadingRecords.value = true
  try {
    const query = new URLSearchParams({ page: String(page) })
    if (recordFilter.value) query.set('task_type', recordFilter.value)
    const data = await api.visionRecords(`?${query.toString()}`)
    records.value = Array.isArray(data) ? data : (data.results || [])
    recordCount.value = Array.isArray(data) ? data.length : (data.count || 0)
    recordPage.value = page
    recordHasNext.value = Array.isArray(data) ? false : Boolean(data.next)
    recordHasPrevious.value = Array.isArray(data) ? false : Boolean(data.previous)
  } catch (err) {
    notify(`质检记录加载失败：${err.message}`, 'error')
  } finally {
    loadingRecords.value = false
  }
}

async function runDetect() {
  if (!canRun.value) return notify('请先选择或上传一张图片', 'error')
  running.value = true
  error.value = ''
  try {
    const params = { ...activeParams.value }
    let payload
    if (sourceKind.value === 'upload') {
      payload = new FormData()
      payload.append('image', file.value)
      payload.append('params', JSON.stringify(params))
    } else {
      payload = { sample: `${sampleGroup.value}/${sampleName.value}`, params }
    }
    if (mode.value === 'defect') {
      result.value = await api.visionDefect(payload)
    } else {
      result.value = mode.value === 'crack'
        ? await api.visionCrack(payload)
        : await api.visionDetect(payload)
    }
    if (result.value.task_type === 'defect' && result.value.verdict === 'uncertain') {
      notify('质检完成：不属于6类缺陷或无缺陷，请人工复核', 'warning')
    } else {
      notify(`质检完成，用时 ${result.value.elapsed_ms} ms`)
    }
    await loadOverview()
    if (subTab.value === 'records') await loadRecords(recordPage.value)
  } catch (err) {
    error.value = err.message
    notify(err.message, 'error')
  } finally {
    running.value = false
  }
}

async function removeRecord(record) {
  if (!window.confirm(`确认删除记录 #${record.id}？对应图片文件也会一并删除。`)) return
  try {
    await api.deleteVisionRecord(record.id)
    notify(`记录 #${record.id} 已删除`)
    if (recordDetail.value?.id === record.id) recordDetail.value = null
    await loadRecords(recordPage.value)
    await loadOverview()
  } catch (err) {
    notify(err.message, 'error')
  }
}

async function openRecord(record) {
  recordDetailLoading.value = true
  try {
    recordDetail.value = await api.visionRecord(record.id)
  } catch (err) {
    notify(`记录详情加载失败：${err.message}`, 'error')
  } finally {
    recordDetailLoading.value = false
  }
}

function openRecords() {
  subTab.value = 'records'
  loadRecords(1)
}

watch(sourceKind, () => {
  result.value = null
  error.value = ''
})

onMounted(() => {
  loadSamples()
  loadOverview()
  loadAlgoStatus()
})

onUnmounted(() => {
  if (objectUrl.value) URL.revokeObjectURL(objectUrl.value)
})
</script>

<template>
  <section class="panel">
    <div class="panel-head">
      <div>
        <div class="panel-title">物件质量检测</div>
        <div class="panel-desc">AGV 取料前拍照质检：判断六类表面缺陷、圈出可疑区域，并保存可追溯的质检记录</div>
      </div>
      <div class="toolbar-right">
        <button class="btn btn-sm" :class="{ 'btn-primary': subTab === 'detect' }" @click="subTab = 'detect'">质检工作台</button>
        <button class="btn btn-sm" :class="{ 'btn-primary': subTab === 'records' }" @click="openRecords">质检记录</button>
      </div>
    </div>

    <div class="panel-body">
      <div class="grid stats-grid">
        <div class="mini-stat panel"><span>累计质检</span><strong>{{ overview.total || 0 }}</strong></div>
        <div class="mini-stat panel"><span>裂纹检出率</span><strong>{{ overview.crack_hit_rate || 0 }}%</strong></div>
        <div class="mini-stat panel"><span>平均耗时</span><strong>{{ overview.average_elapsed_ms || 0 }} ms</strong></div>
        <div class="mini-stat panel"><span>内置数据集</span><strong class="vision-mini-value">{{ datasetCounts }}</strong></div>
      </div>
      <div v-if="subTab === 'detect'" class="grid vision-grid mt-16">
        <section class="panel">
          <div class="panel-head"><div><div class="panel-title">质检配置</div><div class="panel-desc">本地上传是主流程，样本演示仅用于验收和讲解</div></div></div>
          <div class="panel-body">
            <div class="vision-source-switch">
              <button type="button" :class="{ active: sourceKind === 'upload' }" @click="sourceKind = 'upload'">
                <strong>本地上传</strong><span>取料前拍照质检 · 推荐</span>
              </button>
              <button type="button" :class="{ active: sourceKind === 'dataset' }" @click="sourceKind = 'dataset'">
                <strong>样本演示</strong><span>使用内置 NEU / MT 数据</span>
              </button>
            </div>

            <label
              v-if="sourceKind === 'upload'"
              class="vision-upload-zone"
              :class="{ dragging, selected: !!file }"
              @dragover.prevent="dragging = true"
              @dragleave.prevent="dragging = false"
              @drop.prevent="onDrop"
            >
              <input class="file-hidden" type="file" accept="image/jpeg,image/png,image/bmp,image/webp" @change="onFileChange" />
              <span class="upload-mark">＋</span>
              <strong>{{ file ? file.name : '点击选择或拖入待检图片' }}</strong>
              <span>{{ file ? '待检图片已就绪，可在右侧预览后开始质检' : '支持 JPG / PNG / BMP / WEBP，建议使用清晰、正对表面的图片' }}</span>
            </label>

            <details v-else class="sample-browser">
              <summary>
                <strong>样本演示</strong>
                <span>展开选择内置数据，仅用于功能验收</span>
              </summary>
              <div class="form-grid mt-12">
                <div class="field">
                  <label>样例分组</label>
                  <select v-model="sampleGroup" class="select">
                    <option v-for="group in groups" :key="group" :value="group">
                      {{ group === 'crack' ? 'MT_Crack 有裂纹' : group === 'free' ? 'MT_Free 无缺陷' : group === 'neu' ? 'NEU 六类缺陷' : '合成工件' }}（{{ (samples[group] || []).length }}）
                    </option>
                  </select>
                </div>
                <div class="field">
                  <label>样例文件</label>
                  <select v-model="sampleName" class="select">
                    <option v-for="name in activeSamples" :key="name" :value="name">{{ name }}</option>
                  </select>
                </div>
              </div>
              <div class="muted small mt-12">{{ datasetNote }}</div>
            </details>

            <div class="form-grid mt-16">
              <div class="field full">
                <label>质检项目</label>
                <select v-model="mode" class="select">
                  <option v-for="item in MODES" :key="item.key" :value="item.key"
                          :disabled="!modeAvailable(item)">
                    {{ item.label }}{{ modeAvailable(item) ? '' : '（权重未就绪）' }}
                  </option>
                </select>
              </div>
              <template v-if="mode === 'crack'">
                <div class="field">
                  <label>裂纹算法</label>
                  <select v-model="crackAlgorithm" class="select">
                    <option value="frangi">Frangi 传统方法（推荐）</option>
                    <option value="deepcrack">DeepCrack 监督分割（实验：需域内微调权重）</option>
                  </select>
                </div>
              </template>
              <template v-if="mode === 'crack' && crackAlgorithm === 'deepcrack'">
                <div class="field">
                  <label>分割阈值 threshold</label>
                  <input v-model.number="deepcrackParams.threshold" class="input" type="number" step="0.05" min="0.05" max="0.95" />
                </div>
                <div class="field">
                  <label>最小裂纹面积 min_area</label>
                  <input v-model.number="deepcrackParams.min_area" class="input" type="number" step="1" min="0" max="200" />
                </div>
                <div class="field">
                  <label>判定覆盖率 min_coverage</label>
                  <input v-model.number="deepcrackParams.min_coverage" class="input" type="number" step="0.0002" min="0" max="0.01" />
                </div>
                <div class="field">
                  <label>推理长边上限 max_side</label>
                  <input v-model.number="deepcrackParams.max_side" class="input" type="number" step="64" min="128" max="2048" />
                </div>
              </template>
              <template v-if="mode === 'crack' && crackAlgorithm === 'frangi'">
                <div class="field">
                  <label>判定阈值 score_threshold</label>
                  <input v-model.number="frangiParams.score_threshold" class="input" type="number" step="0.02" min="0.05" max="1.5" />
                </div>
                <div class="field">
                  <label>黑脊响应阈值 vessel_threshold</label>
                  <input v-model.number="frangiParams.vessel_threshold" class="input" type="number" step="0.05" min="0.05" max="0.95" />
                </div>
                <div class="field">
                  <label>细长比下限 min_aspect</label>
                  <input v-model.number="frangiParams.min_aspect" class="input" type="number" step="0.1" min="1.2" max="8" />
                </div>
                <div class="field">
                  <label>掩码相对阈值 mask_threshold_ratio</label>
                  <input v-model.number="frangiParams.mask_threshold_ratio" class="input" type="number" step="0.025" min="0" max="1" />
                </div>
                <div class="field">
                  <label>掩码闭运算核 mask_close_kernel</label>
                  <input v-model.number="frangiParams.mask_close_kernel" class="input" type="number" step="2" min="1" max="15" />
                </div>
                <div class="field">
                  <label>中心线骨架化 mask_skeleton</label>
                  <select v-model="frangiParams.mask_skeleton" class="input">
                    <option :value="false">关闭（标定最优）</option>
                    <option :value="true">开启 Zhang-Suen</option>
                  </select>
                </div>
                <div class="field">
                  <label>掩码膨胀像素 mask_dilation</label>
                  <input v-model.number="frangiParams.mask_dilation" class="input" type="number" step="1" min="0" max="8" />
                </div>
              </template>
              <template v-if="mode === 'defect'">
                <div class="field full">
                  <div class="notice">
                    <span>
                      使用 NEU 热轧带钢 <b>6 类缺陷</b> CNN（crazing / inclusion / patches / pitted / rolled-in / scratches），
                      验证准确率 <b>{{ defectAccuracyText }}</b>；该任务<b>无需调参</b>，直接点「开始质检」。
                    </span>
                  </div>
                </div>
              </template>
              <template v-if="mode === 'workpiece'">
                <div class="field">
                  <label>最小面积 min_area</label>
                  <input v-model.number="workpieceParams.min_area" class="input" type="number" step="50" min="20" max="5000" />
                </div>
                <div class="field">
                  <label>多边形逼近精度 approx_epsilon</label>
                  <input v-model.number="workpieceParams.approx_epsilon" class="input" type="number" step="0.005" min="0.005" max="0.1" />
                </div>
                <div class="field">
                  <label>形态学核尺寸 morph_kernel</label>
                  <input v-model.number="workpieceParams.morph_kernel" class="input" type="number" step="2" min="1" max="15" />
                </div>
              </template>
            </div>

            <div v-if="mode === 'defect' && !defectAvailable" class="notice mt-12">
              NEU 六类缺陷分类权重未就绪（runs/vision/neu_cnn.pt）。
              请先运行 scripts/train_neu.py 训练，或改选其他质检项目。
            </div>
            <div v-if="mode === 'crack' && crackAlgorithm === 'deepcrack' && !deepcrackAvailable"
                 class="notice mt-12">
              未找到 DeepCrack 权重（models/deepcrack.pth），本次请求将自动回退到 Frangi。
            </div>

            <div class="toolbar-row mt-16">
              <button class="btn btn-primary" :disabled="running || !canRun" @click="runDetect">
                {{ running ? '正在质检...' : '开始质检' }}
              </button>
              <button class="btn" :disabled="running" @click="resetForm">重置</button>
            </div>
            <div v-if="error" class="notice mt-12">{{ error }}</div>
            <div v-if="!activeSamples.length && sourceKind === 'dataset'" class="notice mt-12">
              未发现内置样例。请先运行 <span class="mono">python manage.py generate_workpiece_dataset</span>，或改用本地上传。
            </div>
          </div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <div><div class="panel-title">质检结果</div><div class="panel-desc">左侧为待检图片，右侧为带红框的质检结果图</div></div>
            <span v-if="result" class="tag">{{ result.task_type === 'crack' ? '裂纹检测' : result.task_type === 'defect' ? '缺陷识别' : '工件识别' }}</span>
          </div>
          <div class="panel-body">
            <div class="vision-compare">
              <figure class="vision-figure">
                <figcaption class="cell-sub">输入图像</figcaption>
                <img v-if="previewUrl" class="vision-preview" :src="previewUrl" alt="输入图像" />
                <div v-else class="vision-placeholder">未选择图片</div>
              </figure>
              <figure class="vision-figure">
                <figcaption class="cell-sub">结果图像</figcaption>
                <img v-if="result && result.image_url" class="vision-preview" :src="result.image_url" alt="质检结果" />
                <div v-else-if="result" class="vision-placeholder">该任务无标注结果图</div>
                <div v-else class="vision-placeholder">尚未质检</div>
              </figure>
            </div>

            <template v-if="result && result.task_type === 'crack'">
              <div class="notice mt-16" :class="result.result === 'crack' ? 'warning' : 'success'">
                <strong>{{ result.result === 'crack' ? '判定：检出裂纹' : '判定：未见裂纹' }}</strong>
                <span v-if="result.fallback">（DeepCrack 不可用，已回退到 Frangi 传统方法）</span>
                <span v-else-if="result.result === 'crack' && !result.localized">（置信分超过阈值但未定位到裂纹区域，建议人工复核）</span>
              </div>
              <div class="grid metric-grid mt-16">
                <div class="optimization-metric"><span>判定结果</span><strong>{{ result.result === 'crack' ? 'crack' : 'normal' }}</strong></div>
                <div class="optimization-metric"><span>算法</span><strong>{{ result.algorithm === 'deepcrack' ? 'DeepCrack' : 'Frangi' }}</strong></div>
                <div class="optimization-metric"><span>裂纹区域数</span><strong>{{ result.count }}</strong></div>
                <div class="optimization-metric"><span>置信分</span><strong>{{ result.score }}</strong></div>
                <div v-if="result.algorithm === 'deepcrack'" class="optimization-metric"><span>裂纹覆盖率</span><strong>{{ result.coverage ?? '--' }}</strong></div>
                <div v-if="result.algorithm === 'deepcrack'" class="optimization-metric"><span>最大概率</span><strong>{{ result.max_prob ?? '--' }}</strong></div>
                <div v-if="result.algorithm !== 'deepcrack'" class="optimization-metric"><span>最大脊响应</span><strong>{{ result.max_response ?? '--' }}</strong></div>
                <div class="optimization-metric"><span>原图尺寸</span><strong>{{ result.width }}×{{ result.height }}</strong></div>
                <div class="optimization-metric"><span>耗时</span><strong>{{ result.elapsed_ms }} ms</strong></div>
              </div>
            </template>

            <template v-else-if="result && result.task_type === 'defect'">
              <div v-if="result.verdict === 'uncertain'" class="notice warning mt-16 vision-verdict">
                <strong>不属于6类缺陷或无缺陷，请人工复核</strong>
                <span>
                  最高分类为 {{ result.raw_predicted_label }}，但置信度 {{ (result.confidence * 100).toFixed(1) }}%
                  或类别间隔 {{ (result.class_margin * 100).toFixed(1) }}% 未达到安全阈值。
                </span>
              </div>
              <div v-else class="notice mt-16 vision-verdict" :class="result.is_line_like ? 'warning' : 'danger'">
                <strong>检出缺陷：{{ result.predicted_label }}</strong>
                <span>（置信度 {{ (result.confidence * 100).toFixed(1) }}%，已圈出 {{ result.boxes.length }} 个疑似区域）</span>
                <span v-if="result.is_line_like"> · 线状缺陷，红框为显著性近似定位</span>
              </div>
              <div v-if="result.ground_truth_class" class="notice mt-12"
                   :class="result.match ? 'success' : 'warning'">
                真值：{{ result.ground_truth_label }}　预测：{{ result.predicted_label }}　
                <strong>{{ result.match ? '一致' : '不一致，建议复核' }}</strong>
              </div>
              <div class="grid metric-grid mt-16">
                <div class="optimization-metric"><span>判定状态</span><strong>{{ result.verdict === 'defect' ? '有缺陷' : '需复核' }}</strong></div>
                <div class="optimization-metric"><span>分类结果</span><strong>{{ result.predicted_class || '未确认' }}</strong></div>
                <div class="optimization-metric"><span>中文</span><strong>{{ result.predicted_label }}</strong></div>
                <div class="optimization-metric"><span>置信度</span><strong>{{ (result.confidence * 100).toFixed(1) }}%</strong></div>
                <div class="optimization-metric"><span>红框数量</span><strong>{{ result.boxes.length }}</strong></div>
                <div class="optimization-metric"><span>定位方式</span><strong>{{ result.localization_method || '未定位' }}</strong></div>
                <div class="optimization-metric"><span>测试增强</span><strong>{{ result.tta_views || 1 }} 视图</strong></div>
                <div class="optimization-metric"><span>原图尺寸</span><strong>{{ result.width }}×{{ result.height }}</strong></div>
                <div class="optimization-metric"><span>耗时</span><strong>{{ result.elapsed_ms }} ms</strong></div>
              </div>
              <div class="mt-16">
                <div class="cell-sub mb-8">Top-3 概率</div>
                <div v-for="item in result.top3 || []" :key="item.class" class="defect-bar-row">
                  <span class="defect-bar-label">{{ item.label }}</span>
                  <div class="progress-track"><div class="progress-bar" :style="{ width: `${item.probability * 100}%` }"></div></div>
                  <strong>{{ (item.probability * 100).toFixed(1) }}%</strong>
                </div>
              </div>
              <div v-if="result.objects.length" class="table-wrap mt-16">
                <table>
                  <thead><tr><th>#</th><th>缺陷类别</th><th>红框 (x,y,w,h)</th><th>显著性</th><th>类别置信度</th></tr></thead>
                  <tbody>
                    <tr v-for="(item, index) in result.objects" :key="index">
                      <td>{{ index + 1 }}</td>
                      <td><span class="cell-main">{{ item.label_zh }}</span><div class="cell-sub mono">{{ item.label }}</div></td>
                      <td class="mono">{{ item.box.join(', ') }}</td>
                      <td>{{ (item.score * 100).toFixed(1) }}%</td>
                      <td>{{ (item.confidence * 100).toFixed(1) }}%</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div class="notice mt-16">
                {{ result.verdict === 'defect'
                  ? '红框由模型显著性热力图与局部反差融合生成，用于辅助人工复核，不代表标注级检测精度。'
                  : '该结果是安全拒识，不代表已确认无缺陷；建议调整拍摄角度、光照后重拍或转人工复核。' }}
              </div>
            </template>

            <template v-else-if="result">
              <div class="grid metric-grid mt-16">
                <div class="optimization-metric"><span>识别工件数</span><strong>{{ result.count }}</strong></div>
                <div class="optimization-metric"><span>类别数</span><strong>{{ Object.keys(result.class_counts || {}).length }}</strong></div>
                <div class="optimization-metric"><span>原图尺寸</span><strong>{{ result.width }}×{{ result.height }}</strong></div>
                <div class="optimization-metric"><span>耗时</span><strong>{{ result.elapsed_ms }} ms</strong></div>
              </div>
              <div class="legend-list mt-16">
                <span v-for="(value, key) in result.class_counts || {}" :key="key" class="tag">
                  {{ SHAPE_LABELS[key] || key }} × {{ value }}
                </span>
                <span v-if="!Object.keys(result.class_counts || {}).length" class="cell-sub">未识别到工件</span>
              </div>
              <div class="table-wrap mt-16">
                <table>
                  <thead><tr><th>#</th><th>类别</th><th>置信分</th><th>外接框 (x,y,w,h)</th><th>面积</th><th>顶点数</th></tr></thead>
                  <tbody>
                    <tr v-for="(item, index) in objects" :key="index">
                      <td>{{ index + 1 }}</td>
                      <td><span class="cell-main">{{ SHAPE_LABELS[item.label] || item.label }}</span><span class="cell-sub mono">{{ item.label }}</span></td>
                      <td>{{ item.score }}</td>
                      <td class="mono">{{ item.box.join(', ') }}</td>
                      <td>{{ item.area }}</td>
                      <td>{{ item.vertices }}</td>
                    </tr>
                    <tr v-if="!objects.length"><td colspan="6"><div class="empty">未识别到工件</div></td></tr>
                  </tbody>
                </table>
              </div>
            </template>
            <div v-else class="empty mt-16">请在左侧选择待检图片后点击「开始质检」</div>
          </div>
        </section>
      </div>

      <div v-else class="mt-16">
        <section v-if="recordDetailLoading" class="panel record-detail-panel"><div class="empty">正在加载记录详情...</div></section>
        <section v-else-if="recordDetail" class="panel record-detail-panel">
          <div class="panel-head">
            <div>
              <div class="panel-title">质检记录 #{{ recordDetail.id }}</div>
              <div class="panel-desc">{{ recordDetail.task_type_display }} · {{ formatTime(recordDetail.created_at) }}</div>
            </div>
            <div class="toolbar-right">
              <span class="tag">{{ recordDetail.verdict_display }}</span>
              <button class="btn btn-sm" @click="recordDetail = null">返回记录列表</button>
              <button class="btn btn-sm btn-danger" @click="removeRecord(recordDetail)">删除记录</button>
            </div>
          </div>
          <div class="panel-body">
            <div class="record-detail-grid">
              <figure class="vision-figure">
                <figcaption class="cell-sub">原始输入</figcaption>
                <img v-if="recordDetail.input_image_url" class="vision-preview record-preview" :src="recordDetail.input_image_url" alt="记录输入图" />
                <div v-else class="vision-placeholder record-preview">没有输入图</div>
              </figure>
              <figure class="vision-figure">
                <figcaption class="cell-sub">质检结果</figcaption>
                <img v-if="recordDetail.result_image_url" class="vision-preview record-preview" :src="recordDetail.result_image_url" alt="记录结果图" />
                <div v-else class="vision-placeholder record-preview">没有结果图</div>
              </figure>
            </div>
            <div class="grid metric-grid mt-16">
              <div class="optimization-metric"><span>来源</span><strong>{{ recordDetail.source_display }}</strong></div>
              <div class="optimization-metric"><span>判定</span><strong>{{ recordDetail.verdict_display }}</strong></div>
              <div class="optimization-metric"><span>目标数量</span><strong>{{ recordDetail.object_count }}</strong></div>
              <div class="optimization-metric"><span>耗时</span><strong>{{ recordDetail.elapsed_ms }} ms</strong></div>
              <div class="optimization-metric"><span>置信度</span><strong>{{ recordDetail.metrics?.confidence != null ? `${(recordDetail.metrics.confidence * 100).toFixed(1)}%` : '--' }}</strong></div>
              <div class="optimization-metric"><span>定位方式</span><strong>{{ recordDetail.metrics?.localization_method || '--' }}</strong></div>
            </div>
            <div v-if="recordObjects.length" class="table-wrap mt-16">
              <table>
                <thead><tr><th>#</th><th>类别</th><th>红框 (x,y,w,h)</th><th>显著性</th><th>类别置信度</th></tr></thead>
                <tbody>
                  <tr v-for="(item, index) in recordObjects" :key="index">
                    <td>{{ index + 1 }}</td>
                    <td>{{ item.label_zh || item.label || recordDetail.task_type_display }}</td>
                    <td class="mono">{{ (item.box || []).join(', ') }}</td>
                    <td>{{ item.score != null ? `${(item.score * 100).toFixed(1)}%` : '--' }}</td>
                    <td>{{ item.confidence != null ? `${(item.confidence * 100).toFixed(1)}%` : '--' }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <details class="record-raw mt-16">
              <summary>查看参数与原始指标</summary>
              <pre>{{ JSON.stringify(recordDetail.parameters, null, 2) }}</pre>
              <pre>{{ JSON.stringify(recordDetail.metrics, null, 2) }}</pre>
            </details>
          </div>
        </section>

        <div class="section-toolbar toolbar-row">
          <div class="toolbar-right">
            <select v-model="recordFilter" class="select" @change="loadRecords(1)">
              <option value="">全部类型</option>
              <option value="crack">裂纹检测</option>
              <option value="defect">六类缺陷识别</option>
              <option value="workpiece">工件识别</option>
            </select>
            <button class="btn btn-sm" :disabled="loadingRecords" @click="loadRecords(recordPage)">
              {{ loadingRecords ? '加载中...' : '刷新记录' }}
            </button>
          </div>
          <span class="muted small">共 {{ recordCount }} 条记录</span>
        </div>
        <div class="table-wrap mt-16">
          <table>
            <thead>
              <tr><th>ID</th><th>类型</th><th>来源</th><th>输入</th><th>判定</th><th>数量</th><th>耗时</th><th>结果图</th><th>时间</th><th>操作</th></tr>
            </thead>
            <tbody>
              <tr v-for="record in records" :key="record.id">
                <td class="mono">{{ record.id }}</td>
                <td>{{ record.task_type_display }}</td>
                <td>{{ record.source_display }}</td>
                <td><span class="cell-main mono small">{{ record.input_name || '--' }}</span></td>
                <td>{{ record.verdict_display }}</td>
                <td>{{ record.object_count }}</td>
                <td>{{ record.elapsed_ms }} ms</td>
                <td><img v-if="record.result_image_url" class="vision-thumb" :src="record.result_image_url" alt="结果缩略图" /></td>
                <td class="small">{{ formatTime(record.created_at) }}</td>
                <td>
                  <div class="table-actions">
                    <button class="btn btn-sm" @click="openRecord(record)">查看</button>
                    <button class="btn btn-sm btn-danger" @click="removeRecord(record)">删除</button>
                  </div>
                </td>
              </tr>
              <tr v-if="!records.length"><td colspan="10"><div class="empty">暂无质检记录</div></td></tr>
            </tbody>
          </table>
        </div>
        <div class="record-pagination">
          <span>第 {{ recordPage }} 页 · {{ recordCount }} 条记录</span>
          <div class="toolbar-right">
            <button class="btn btn-sm" :disabled="!recordHasPrevious || loadingRecords" @click="loadRecords(recordPage - 1)">上一页</button>
            <button class="btn btn-sm" :disabled="!recordHasNext || loadingRecords" @click="loadRecords(recordPage + 1)">下一页</button>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.vision-grid { grid-template-columns: minmax(0, 420px) minmax(0, 1fr); align-items: start; }
.vision-source-switch {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  padding: 5px;
  margin-bottom: 12px;
  border: 1px solid #d9e4e0;
  border-radius: 12px;
  background: #f1f6f4;
}
.vision-source-switch button {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 10px 12px;
  border: 1px solid transparent;
  border-radius: 9px;
  color: #667880;
  background: transparent;
  text-align: left;
}
.vision-source-switch button strong { color: #30474f; font-size: 12px; }
.vision-source-switch button span { font-size: 9px; }
.vision-source-switch button.active {
  border-color: #b9d8d2;
  color: #4b6d70;
  background: #ffffff;
  box-shadow: 0 4px 12px rgba(20, 45, 54, 0.07);
}
.vision-source-switch button.active strong { color: var(--cyan); }
.vision-upload-zone {
  min-height: 142px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 7px;
  padding: 20px;
  border: 1.5px dashed #a9c9c2;
  border-radius: 13px;
  color: #75868d;
  background:
    radial-gradient(circle at 50% 20%, rgba(14, 135, 146, 0.08), transparent 55%),
    #f8fbfa;
  text-align: center;
  cursor: pointer;
  transition: border-color 160ms ease, background 160ms ease, transform 160ms ease;
}
.vision-upload-zone:hover, .vision-upload-zone.dragging {
  border-color: var(--cyan);
  background-color: #eef8f6;
  transform: translateY(-1px);
}
.vision-upload-zone.selected { border-style: solid; border-color: #7fc1b7; background-color: #f1faf7; }
.vision-upload-zone strong { color: #27454b; font-size: 13px; }
.vision-upload-zone > span:last-child { max-width: 330px; font-size: 10px; line-height: 1.55; }
.upload-mark {
  width: 34px;
  height: 34px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  color: #ffffff;
  background: linear-gradient(180deg, #18a0a8, #0e7f88);
  font-size: 21px;
  line-height: 1;
  box-shadow: 0 7px 16px rgba(14, 135, 146, 0.22);
}
.file-hidden { position: absolute; width: 1px; height: 1px; opacity: 0; pointer-events: none; }
.sample-browser {
  padding: 13px;
  border: 1px solid #dce5e2;
  border-radius: 11px;
  background: #f8fbfa;
}
.sample-browser summary { display: flex; justify-content: space-between; gap: 10px; cursor: pointer; }
.sample-browser summary strong { color: #2b454d; font-size: 12px; }
.sample-browser summary span { color: #849297; font-size: 10px; }
.vision-verdict { align-items: center; flex-wrap: wrap; }
.vision-verdict.danger { border-color: #f0c7cc; background: #fff5f6; color: #9f3945; }
.vision-compare { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.vision-figure { margin: 0; display: flex; flex-direction: column; gap: 6px; }
.vision-preview {
  width: 100%; height: 240px; object-fit: contain; border-radius: 10px;
  border: 1px solid var(--line-strong); background: rgba(0, 0, 0, 0.28);
}
.vision-placeholder {
  height: 240px; display: flex; align-items: center; justify-content: center;
  border: 1px dashed var(--line-strong); border-radius: 10px;
  color: var(--muted); font-size: 12px;
}
.vision-thumb {
  width: 66px; height: 44px; object-fit: cover; border-radius: 6px;
  border: 1px solid var(--line-strong);
}
.record-detail-panel { margin-bottom: 16px; }
.record-detail-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.record-preview { height: 310px; }
.record-raw summary { color: #60747b; font-size: 11px; cursor: pointer; }
.record-raw pre {
  max-height: 240px;
  padding: 12px;
  overflow: auto;
  border: 1px solid #e0e7e4;
  border-radius: 9px;
  color: #425a62;
  background: #f7faf9;
  font: 10px/1.55 "Cascadia Mono", Consolas, monospace;
  white-space: pre-wrap;
  word-break: break-word;
}
.record-pagination { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-top: 12px; color: #7a898f; font-size: 11px; }
.legend-list { display: flex; flex-wrap: wrap; gap: 8px; }
.mini-stat span { white-space: nowrap; }
.vision-mini-value { font-size: 19px; white-space: nowrap; }
.btn { white-space: nowrap; }

/* ---- 缺陷识别 top-3 概率条 ---- */
.defect-bar-row { display: grid; grid-template-columns: 96px minmax(0, 1fr) 56px; gap: 10px; align-items: center; font-size: 12px; margin-bottom: 7px; }
.defect-bar-label { color: #5f7179; }
@media (max-width: 760px) {
  .record-detail-grid { grid-template-columns: 1fr; }
  .record-pagination { align-items: flex-start; flex-direction: column; }
  .vision-source-switch { grid-template-columns: 1fr; }
  .sample-browser summary { flex-direction: column; }
}
</style>
