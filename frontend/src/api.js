const API_BASE = import.meta.env.VITE_API_BASE || '/api'

async function request(path, options = {}) {
  const { headers: extraHeaders, ...rest } = options
  const isFormData = typeof FormData !== 'undefined' && rest.body instanceof FormData
  const headers = { ...(extraHeaders || {}) }
  if (!isFormData) {
    headers['Content-Type'] = 'application/json'
  }
  const response = await fetch(`${API_BASE}${path}`, { headers, ...rest })
  const text = await response.text()
  let data = null
  if (text) {
    try {
      data = JSON.parse(text)
    } catch {
      data = { detail: text.slice(0, 180) }
    }
  }
  if (!response.ok) {
    throw new Error(data?.detail || data?.message || `请求失败 (${response.status})`)
  }
  return data
}

function unwrap(data) {
  return Array.isArray(data) ? data : (data?.results || [])
}

function postVision(path, payload) {
  if (typeof FormData !== 'undefined' && payload instanceof FormData) {
    return request(path, { method: 'POST', body: payload })
  }
  return request(path, { method: 'POST', body: JSON.stringify(payload) })
}

export const api = {
  overview: () => request('/overview/'),
  map: () => request('/map/'),
  nodes: () => request('/nodes/').then(unwrap),
  edges: () => request('/edges/').then(unwrap),
  agvs: () => request('/agvs/').then(unwrap),
  tasks: () => request('/tasks/').then(unwrap),
  dispatches: () => request('/dispatches/').then(unwrap),
  scheduleRuns: () => request('/schedule-runs/').then(unwrap),
  createTask: (payload) => request('/tasks/', { method: 'POST', body: JSON.stringify(payload) }),
  updateTask: (id, payload) => request(`/tasks/${id}/`, { method: 'PATCH', body: JSON.stringify(payload) }),
  updateAgv: (id, payload) => request(`/agvs/${id}/`, { method: 'PATCH', body: JSON.stringify(payload) }),
  singleDispatch: (payload) => request('/schedules/single/', { method: 'POST', body: JSON.stringify(payload) }),
  batchDispatch: (payload) => request('/schedules/batch/', { method: 'POST', body: JSON.stringify(payload) }),
  planPath: (payload) => request('/path-planning/', { method: 'POST', body: JSON.stringify(payload) }),
  taskAction: (id, action) => request(`/tasks/${id}/${action}/`, { method: 'POST', body: '{}' }),

  // --- 视觉检测（ManufacturingVision，任务 4/5）---
  visionOverview: () => request('/vision/overview/'),
  visionSamples: () => request('/vision/samples/'),
  visionCrack: (payload) => postVision('/vision/crack/', payload),
  visionDefect: (payload) => postVision('/vision/algo/defect/', payload),
  visionDetect: (payload) => postVision('/vision/detect/', payload),
  visionRecords: (query = '') => request(`/vision/records/${query}`).then(unwrap),
  deleteVisionRecord: (id) => request(`/vision/records/${id}/`, { method: 'DELETE' }),
  visionSampleImageUrl: (group, name) =>
    `${API_BASE}/vision/samples/${encodeURIComponent(group)}/${encodeURIComponent(name)}/`,

  // --- 设备状态监测（任务 6）---
  equipmentOverview: () => request('/equipment/overview/'),
  equipmentSample: (scenario = 'worn') =>
    request(`/equipment/sample/?scenario=${encodeURIComponent(scenario)}`),
  equipmentModel: () => request('/equipment/model/'),
  equipmentDatasetCompare: () => request('/equipment/algo/dataset-compare/'),
  equipmentDiagnose: (payload) => postVision('/equipment/diagnose/', payload),
  equipmentRecords: (query = '') => request(`/equipment/records/${query}`).then(unwrap),
  deleteEquipmentRecord: (id) =>
    request(`/equipment/records/${id}/`, { method: 'DELETE' }),

  // --- 订单需求预测（任务 7）---
  forecastOverview: () => request('/forecast/overview/'),
  forecastSample: (days = 240) => request(`/forecast/sample/?days=${days}`),
  forecastPredict: (payload) => postVision('/forecast/predict/', payload),
  forecastRuns: (query = '') => request(`/forecast/runs/${query}`).then(unwrap),
  deleteForecastRun: (id) => request(`/forecast/runs/${id}/`, { method: 'DELETE' }),
}
