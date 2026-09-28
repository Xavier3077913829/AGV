<script setup>
import { computed, ref, watch } from 'vue'
import { api } from '../api'

const props = defineProps({ mapData: { type: Object, default: () => ({}) } })
const emit = defineEmits(['notify'])
const selectedRoute = ref(null)
const width = 920
const height = 520
const padding = 58
const colors = ['#23d5e6', '#4b8cff', '#9b7cff', '#ffbd4a', '#41d69a', '#ff6372']

const nodes = computed(() => props.mapData.nodes || [])
const edges = computed(() => props.mapData.edges || [])
const agvs = computed(() => props.mapData.agvs || [])
const routes = computed(() => props.mapData.active_routes || [])
const obstacles = computed(() => props.mapData.obstacles || [])
const nodeMap = computed(() => Object.fromEntries(nodes.value.map((node) => [node.id, node])))
const extent = computed(() => {
  const nodeXs = nodes.value.map((node) => node.x)
  const nodeYs = nodes.value.map((node) => node.y)
  const obstacleXs = obstacles.value.map((item) => item.x + item.width)
  const obstacleYs = obstacles.value.map((item) => item.y + item.height)
  return {
    minX: Math.min(...nodeXs, ...obstacleXs, 0),
    maxX: Math.max(...nodeXs, ...obstacleXs, 1),
    minY: Math.min(...nodeYs, ...obstacleYs, 0),
    maxY: Math.max(...nodeYs, ...obstacleYs, 1),
  }
})

const startNodeId = ref('')
const endNodeId = ref('')
const plannerAgvId = ref('')
const pathResult = ref(null)
const pathPlanning = ref(false)

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
  return { parking: '#66819c', pickup: '#41d69a', dropoff: '#ffbd4a', charging: '#9b7cff', intersection: '#3b6e9e' }[type] || '#3b6e9e'
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
          <svg :viewBox="`0 0 ${width} ${height}`" role="img" aria-label="AGV 路径规划地图">
            <line v-for="edge in edges" :key="edge.id" class="map-edge" :x1="point(edge.start_node).x" :y1="point(edge.start_node).y" :x2="point(edge.end_node).x" :y2="point(edge.end_node).y" />
            <rect v-for="item in obstacles" :key="item.id" class="map-obstacle" v-bind="obstacleRect(item)" rx="3" />
            <polyline v-for="(route, index) in routes" :key="route.task_id" class="map-route" :points="points(route.route?.nodes)" :style="{ stroke: routeColor(index), color: routeColor(index) }" />
            <polyline v-if="pathResult" class="map-route manual-route" :points="points(pathResult.route.nodes)" />
            <g v-for="node in nodes" :key="node.id" class="map-node-group">
              <circle class="map-node" :cx="point(node.id).x" :cy="point(node.id).y" :r="node.node_type === 'intersection' ? 3 : 10" :fill="nodeColor(node.node_type)" />
              <text v-if="node.node_type !== 'intersection'" class="map-node-label" :x="point(node.id).x" :y="point(node.id).y + 23" text-anchor="middle">{{ node.code }}</text>
            </g>
            <g v-if="pathResult" class="planner-endpoints">
              <circle :cx="point(pathResult.route.nodes[0]).x" :cy="point(pathResult.route.nodes[0]).y" r="15" fill="none" stroke="#41d69a" stroke-width="4" />
              <circle :cx="point(pathResult.route.nodes[pathResult.route.nodes.length - 1]).x" :cy="point(pathResult.route.nodes[pathResult.route.nodes.length - 1]).y" r="15" fill="none" stroke="#ff6372" stroke-width="4" />
            </g>
            <g v-for="(agv, index) in agvs" :key="agv.id" class="map-agv">
              <circle :cx="point(agv.current_node).x + (index % 2) * 10 - 5" :cy="point(agv.current_node).y - 10 - Math.floor(index / 2) * 10" r="12" fill="#23d5e6" opacity="0.2" />
              <circle :cx="point(agv.current_node).x + (index % 2) * 10 - 5" :cy="point(agv.current_node).y - 10 - Math.floor(index / 2) * 10" r="6" fill="#23d5e6" stroke="#03131f" stroke-width="2" />
              <text :x="point(agv.current_node).x + (index % 2) * 10 - 5" :y="point(agv.current_node).y - 28 - Math.floor(index / 2) * 10" text-anchor="middle" fill="#8feaf2" font-size="10">{{ agv.code }}</text>
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
          <div class="legend-row"><span class="legend-name"><span class="legend-dot" style="--dot: #41d69a"></span>取货点</span></div>
          <div class="legend-row"><span class="legend-name"><span class="legend-dot" style="--dot: #ffbd4a"></span>放货点</span></div>
          <div class="legend-row"><span class="legend-name"><span class="legend-dot" style="--dot: #ff6372"></span>规划路径</span></div>
        </div>
      </div>
    </aside>
  </div>
</template>