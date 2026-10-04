<script setup>
import { computed, ref, watch } from 'vue'
import { api } from '../api'

const props = defineProps({ mapData: { type: Object, default: () => ({}) } })
const emit = defineEmits(['notify'])
const selectedRoute = ref(null)
const hoveredNode = ref(null)
const width = 920
const height = 880
const padding = 36
const colors = ['#0e8792', '#2f6fda', '#7858ce', '#c58a1b', '#29976a', '#cf5260']

const nodes = computed(() => props.mapData.nodes || [])
const edges = computed(() => props.mapData.edges || [])
const agvs = computed(() => props.mapData.agvs || [])
const routes = computed(() => props.mapData.active_routes || [])
const executingRoutes = computed(() => routes.value.filter((item) => ['assigned', 'in_progress'].includes(item.status)))
const routePage = ref(1)
const routePageSize = 2
const routeTotalPages = computed(() => Math.max(1, Math.ceil(executingRoutes.value.length / routePageSize)))
const pagedExecutingRoutes = computed(() => {
  const start = (routePage.value - 1) * routePageSize
  return executingRoutes.value.slice(start, start + routePageSize)
})
const obstacles = computed(() => props.mapData.obstacles || [])
const nodeMap = computed(() => Object.fromEntries(nodes.value.map((node) => [node.id, node])))
const extent = computed(() => {
  const nodeXs = nodes.value.map((node) => node.x)
  const nodeYs = nodes.value.map((node) => node.y)
  const obstacleXs = obstacles.value.map((item) => item.x + item.width)
  const obstacleYs = obstacles.value.map((item) => item.y + item.height)
  const rawMinX = Math.min(...nodeXs, ...obstacleXs, 0)
  const rawMaxX = Math.max(...nodeXs, ...obstacleXs, 1)
  const rawMinY = Math.min(...nodeYs, ...obstacleYs, 0)
  const rawMaxY = Math.max(...nodeYs, ...obstacleYs, 1)
  return {
    minX: rawMinX - 0.5,
    maxX: rawMaxX + 0.5,
    minY: rawMinY - 0.5,
    maxY: rawMaxY + 0.5,
  }
})

const gridXs = computed(() => {
  const count = Math.round(extent.value.maxX - extent.value.minX)
  return Array.from({ length: count + 1 }, (_, index) => extent.value.minX + index)
})
const gridYs = computed(() => {
  const count = Math.round(extent.value.maxY - extent.value.minY)
  return Array.from({ length: count + 1 }, (_, index) => extent.value.minY + index)
})

const hoverTooltipWidth = computed(() => {
  if (!hoveredNode.value) return 150
  return Math.max(150, (hoveredNode.value.code.length + hoveredNode.value.name.length) * 7 + 28)
})

const startNodeId = ref('')
const endNodeId = ref('')
const plannerAgvId = ref('')
const pathResult = ref(null)
const pathPlanning = ref(false)

watch(executingRoutes, () => {
  if (routePage.value > routeTotalPages.value) routePage.value = routeTotalPages.value
})

watch(nodes, (items) => {
  if (!items.length) return
  if (!startNodeId.value) startNodeId.value = items.find((item) => item.code === 'S2')?.id || items[0].id
  if (!endNodeId.value) endNodeId.value = items.find((item) => item.code === 'S6')?.id || items[items.length - 1].id
}, { immediate: true })

function coordinatePoint(x, y) {
  const { minX, maxX, minY, maxY } = extent.value
  return {
    x: padding + ((x - minX) / Math.max(maxX - minX, 1)) * (width - padding * 2),
    y: height - padding - ((y - minY) / Math.max(maxY - minY, 1)) * (height - padding * 2),
  }
}

function point(nodeId) {
  const node = nodeMap.value[nodeId]
  if (!node) return { x: 0, y: 0 }
  return coordinatePoint(node.x, node.y)
}

function points(path) {
  return (path || []).map((id) => point(id)).map((item) => item.x + ',' + item.y).join(' ')
}

function obstacleRect(item) {
  const topLeft = coordinatePoint(item.x, item.y)
  const bottomRight = coordinatePoint(item.x + item.width, item.y + item.height)
  return {
    x: Math.min(topLeft.x, bottomRight.x) + 6,
    y: Math.min(topLeft.y, bottomRight.y) + 6,
    width: Math.max(Math.abs(bottomRight.x - topLeft.x) - 12, 8),
    height: Math.max(Math.abs(bottomRight.y - topLeft.y) - 12, 8),
  }
}

function nodeColor(type) {
  return { parking: '#7b8d98', pickup: '#29976a', dropoff: '#c58a1b', charging: '#7858ce', intersection: '#67808d' }[type] || '#67808d'
}

function routeColor(index) {
  return colors[index % colors.length]
}

function routeText(route) {
  return (route.route?.nodes || []).map((id) => nodeMap.value[id]?.code || id).join(' -> ')
}

async function planPath() {
  if (!startNodeId.value || !endNodeId.value) return
  pathPlanning.value = true
  try {
    pathResult.value = await api.planPath({
      start_node_id: Number(startNodeId.value),
      end_node_id: Number(endNodeId.value),
      agv_id: plannerAgvId.value ? Number(plannerAgvId.value) : null,
    })
    selectedRoute.value = null
  } catch (error) {
    emit('notify', { message: error.message, type: 'error' })
  } finally {
    pathPlanning.value = false
  }
}

function clearPath() {
  pathResult.value = null
}
</script>

<template>
  <div class="map-layout map-layout-wide">
    <section class="panel">
      <div class="panel-head">
        <div><div class="panel-title">仓库障碍地图与路径规划</div><div class="panel-desc">灰色路网、黑色障碍块、S1-S10 工作站以及 AGV 彩色路线</div></div>
        <div class="toolbar-right"><span class="tag">节点 {{ nodes.length }}</span><span class="tag">道路 {{ edges.length }}</span><span class="tag">障碍 {{ obstacles.length }}</span></div>
      </div>
      <div class="panel-body">
        <div class="map-canvas warehouse-map">
          <svg :viewBox="`0 0 ${width} ${height}`" preserveAspectRatio="xMidYMid meet" role="img" aria-label="AGV 路径规划地图">
            <line v-for="x in gridXs" :key="`v-${x}`" class="map-grid-line" :x1="coordinatePoint(x, extent.minY).x" :y1="coordinatePoint(x, extent.minY).y" :x2="coordinatePoint(x, extent.maxY).x" :y2="coordinatePoint(x, extent.maxY).y" />
            <line v-for="y in gridYs" :key="`h-${y}`" class="map-grid-line" :x1="coordinatePoint(extent.minX, y).x" :y1="coordinatePoint(extent.minX, y).y" :x2="coordinatePoint(extent.maxX, y).x" :y2="coordinatePoint(extent.maxX, y).y" />
            <line v-for="edge in edges" :key="edge.id" class="map-edge" :x1="point(edge.start_node).x" :y1="point(edge.start_node).y" :x2="point(edge.end_node).x" :y2="point(edge.end_node).y" />
            <rect v-for="item in obstacles" :key="item.id" class="map-obstacle" v-bind="obstacleRect(item)" rx="3" />
            <polyline v-for="(route, index) in routes" :key="route.task_id" class="map-route" :class="{ 'route-selected': selectedRoute === route.task_id }" :points="points(route.route?.nodes)" :style="{ stroke: routeColor(index), color: routeColor(index) }" />
            <polyline v-if="pathResult" class="map-route manual-route" :points="points(pathResult.route.nodes)" />
            <g v-for="node in nodes" :key="node.id" class="map-node-group" @mouseenter="hoveredNode = node" @mouseleave="hoveredNode = null">
              <title>{{ node.code }} · {{ node.name }}</title>
              <circle class="map-node-hit" :cx="point(node.id).x" :cy="point(node.id).y" r="11" />
              <circle class="map-node" :cx="point(node.id).x" :cy="point(node.id).y" :r="node.node_type === 'intersection' ? 3 : 10" :fill="nodeColor(node.node_type)" />
              <text v-if="node.node_type !== 'intersection'" class="map-node-label" :x="point(node.id).x" :y="point(node.id).y + 23" text-anchor="middle">{{ node.code }}</text>
            </g>
            <g v-if="pathResult" class="planner-endpoints">
              <circle :cx="point(pathResult.route.nodes[0]).x" :cy="point(pathResult.route.nodes[0]).y" r="15" fill="none" stroke="#29976a" stroke-width="4" />
              <circle :cx="point(pathResult.route.nodes[pathResult.route.nodes.length - 1]).x" :cy="point(pathResult.route.nodes[pathResult.route.nodes.length - 1]).y" r="15" fill="none" stroke="#cf5260" stroke-width="4" />
            </g>
            <g v-for="(agv, index) in agvs" :key="agv.id" class="map-agv">
              <circle :cx="point(agv.current_node).x + (index % 2) * 10 - 5" :cy="point(agv.current_node).y - 10 - Math.floor(index / 2) * 10" r="12" fill="#0e8792" opacity="0.2" />
              <circle :cx="point(agv.current_node).x + (index % 2) * 10 - 5" :cy="point(agv.current_node).y - 10 - Math.floor(index / 2) * 10" r="6" fill="#0e8792" stroke="#ffffff" stroke-width="2" />
              <text :x="point(agv.current_node).x + (index % 2) * 10 - 5" :y="point(agv.current_node).y - 28 - Math.floor(index / 2) * 10" text-anchor="middle" fill="#2f6f75" font-size="10">{{ agv.code }}</text>
            </g>
            <g v-if="hoveredNode" class="map-node-tooltip" pointer-events="none">
              <rect :x="point(hoveredNode.id).x + 12" :y="point(hoveredNode.id).y - 42" :width="hoverTooltipWidth" height="30" rx="6" />
              <text :x="point(hoveredNode.id).x + 22" :y="point(hoveredNode.id).y - 23">{{ hoveredNode.code }} · {{ hoveredNode.name }}</text>
            </g>
          </svg>
        </div>
      </div>
    </section>

    <aside class="panel">
      <div class="panel-head"><div><div class="panel-title">起点到终点路径规划</div><div class="panel-desc">自动绕开障碍并满足道路、电池约束</div></div></div>
      <div class="panel-body">
        <div class="path-planner">
          <div class="field"><label>起点</label><select v-model="startNodeId" class="select"><option v-for="node in nodes" :key="node.id" :value="node.id">{{ node.code }} · {{ node.name }}</option></select></div>
          <div class="field"><label>终点</label><select v-model="endNodeId" class="select"><option v-for="node in nodes" :key="node.id" :value="node.id">{{ node.code }} · {{ node.name }}</option></select></div>
          <div class="field"><label>执行 AGV（可选）</label><select v-model="plannerAgvId" class="select"><option value="">通用 AGV</option><option v-for="agv in agvs" :key="agv.id" :value="agv.id">{{ agv.code }} · 电量 {{ agv.battery_percent }}%</option></select></div>
          <div class="toolbar-row"><button class="btn btn-primary" :disabled="pathPlanning" @click="planPath">{{ pathPlanning ? '规划中...' : '规划最短路径' }}</button><button class="btn" @click="clearPath">清除</button></div>
        </div>
        <div v-if="pathResult" class="planner-result mt-16">
          <div class="planner-metrics"><div><span>距离</span><strong>{{ pathResult.distance }} m</strong></div><div><span>预计耗时</span><strong>{{ pathResult.estimated_duration }} s</strong></div><div><span>节点数</span><strong>{{ pathResult.route.nodes.length }}</strong></div></div>
          <div class="route-path mt-12">{{ routeText(pathResult) }}</div>
          <div v-if="pathResult.agv" class="notice mt-12" :class="pathResult.agv.battery_ok ? 'success' : 'warning'">预计耗电 {{ pathResult.agv.battery_required }}%，{{ pathResult.agv.battery_ok ? '电量满足约束' : '电量不足' }}</div>
        </div>
        <div class="legend-list mt-16">
          <div class="legend-row"><span class="legend-name"><span class="legend-dot" style="--dot: #000"></span>障碍区域</span></div>
          <div class="legend-row"><span class="legend-name"><span class="legend-dot" style="--dot: #29976a"></span>取货点</span></div>
          <div class="legend-row"><span class="legend-name"><span class="legend-dot" style="--dot: #c58a1b"></span>放货点</span></div>
          <div class="legend-row"><span class="legend-name"><span class="legend-dot" style="--dot: #cf5260"></span>规划路径</span></div>
        </div>
        <div class="executing-routes mt-16">
          <div class="route-section-head"><strong>正在执行任务路线</strong><span>{{ executingRoutes.length }} 条</span></div>
          <div v-if="executingRoutes.length" class="route-list">
            <button v-for="(route, index) in pagedExecutingRoutes" :key="route.task_id" class="route-card route-button" :class="{ active: selectedRoute === route.task_id }" @click="selectedRoute = selectedRoute === route.task_id ? null : route.task_id">
              <div class="route-card-top">
                <div><span class="strong mono">{{ route.task_no }}</span><span class="tag" style="margin-left: 7px">{{ route.agv_code || '未分配' }}</span></div>
                <span class="legend-dot" :style="{ '--dot': routeColor(routes.indexOf(route)) }"></span>
              </div>
              <div class="route-path">{{ nodeMap[route.pickup_node]?.code || '--' }} → {{ nodeMap[route.dropoff_node]?.code || '--' }} · 顺序 #{{ route.sequence_number || '-' }}</div>
              <div class="route-path">{{ route.total_distance || 0 }} m · {{ route.estimated_duration || 0 }} s · {{ route.status === 'in_progress' ? '执行中' : '已分配' }}</div>
              <div class="route-full-path"><span>完整路径</span><code>{{ routeText(route) || '暂无路径' }}</code></div>
            </button>
          </div>
          <div v-else class="empty small-empty">暂无正在执行的任务路线</div>
          <div v-if="executingRoutes.length" class="route-pagination">
            <button class="btn btn-sm" :disabled="routePage <= 1" @click="routePage -= 1">上一页</button>
            <span>第 {{ routePage }} / {{ routeTotalPages }} 页</span>
            <button class="btn btn-sm" :disabled="routePage >= routeTotalPages" @click="routePage += 1">下一页</button>
          </div>
        </div>
      </div>
    </aside>
  </div>
</template>
