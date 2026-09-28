const API_BASE = import.meta.env.VITE_API_BASE || '/api'

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  })
  const text = await response.text()
  const data = text ? JSON.parse(text) : null
  if (!response.ok) {
    throw new Error(data?.detail || data?.message || `请求失败 (${response.status})`)
  }
  return data
}

function unwrap(data) {
  return Array.isArray(data) ? data : (data?.results || [])
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
  taskAction: (id, action) => request(`/tasks/${id}/${action}/`, { method: 'POST', body: '{}' }),
}
