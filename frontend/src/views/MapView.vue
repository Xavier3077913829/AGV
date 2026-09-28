<script setup>
import { computed, ref } from 'vue'
import StatusBadge from '../components/StatusBadge.vue'

const props = defineProps({ mapData: { type: Object, default: () => ({}) } })
const selectedRoute = ref(null)
const width = 920
const height = 520
const padding = 58
const colors = ['#23d5e6', '#4b8cff', '#9b7cff', '#ffbd4a', '#41d69a', '#ff6372']

const nodes = computed(() => props.mapData.nodes || [])
const edges = computed(() => props.mapData.edges || [])
const agvs = computed(() => props.mapData.agvs || [])
const routes = computed(() => props.mapData.active_routes || [])
const nodeMap = computed(() => Object.fromEntries(nodes.value.map((node) => [node.id, node])))
const extent = computed(() => {
  const xs = nodes.value.map((node) => node.x)
  const ys = nodes.value.map((node) => node.y)
  return { minX: Math.min(...xs, 0), maxX: Math.max(...xs, 1), minY: Math.min(...ys, 0), maxY: Math.max(...ys, 1) }
})

function point(nodeId) {
  const node = nodeMap.value[nodeId]
  if (!node) return { x: 0, y: 0 }
  const { minX, maxX, minY, maxY } = extent.value
  const x = padding + ((node.x - minX) / Math.max(maxX - minX, 1)) * (width - padding * 2)
  const y = height - padding - ((node.y - minY) / Math.max(maxY - minY, 1)) * (height - padding * 2)
  return { x, y }
}

function points(path) {
  return (path || []).map((id) => `${point(id).x},${point(id).y}`).join(' ')
}

function nodeColor(type) {
  return { parking: '#66819c', pickup: '#41d69a', dropoff: '#ffbd4a', charging: '#9b7cff', intersection: '#3b6e9e' }[type] || '#3b6e9e'
}

function routeColor(index) {
  return colors[index % colors.length]
}

function routeText(route) {
  return (route.route?.nodes || []).map((id) => nodeMap.value[id]?.code || id).join(' → ')
}
</script>

<template>
  <div class="map-layout">
    <section class="panel">
      <div class="panel-head">
        <div><div class="panel-title">仓库地图与实时路线</div><div class="panel-desc">彩色流动路线为当前执行中的任务路径</div></div>
        <div class="toolbar-right"><span class="tag">节点 {{ nodes.length }}</span><span class="tag">道路 {{ edges.length }}</span><span class="tag">活动路线 {{ routes.length }}</span></div>
      </div>
      <div class="panel-body">
        <div class="map-canvas">
          <svg :viewBox="`0 0 ${width} ${height}`" role="img" aria-label="AGV 仓库地图">
            <line v-for="edge in edges" :key="edge.id" class="map-edge" :x1="point(edge.start_node).x" :y1="point(edge.start_node).y" :x2="point(edge.end_node).x" :y2="point(edge.end_node).y" />
            <polyline v-for="(route, index) in routes" :key="route.task_id" class="map-route" :points="points(route.route?.nodes)" :style="{ stroke: routeColor(index), color: routeColor(index) }" />
            <g v-for="node in nodes" :key="node.id" class="map-node-group">
              <circle class="map-node" :cx="point(node.id).x" :cy="point(node.id).y" :r="node.node_type === 'intersection' ? 8 : 11" :fill="nodeColor(node.node_type)" />
              <text class="map-node-label" :x="point(node.id).x" :y="point(node.id).y + 24" text-anchor="middle">{{ node.code }}</text>
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
      <div class="panel-head"><div><div class="panel-title">活动任务路线</div><div class="panel-desc">点击路线查看经过节点</div></div></div>
      <div class="panel-body">
        <div v-if="routes.length" class="route-list">
          <button v-for="(route, index) in routes" :key="route.task_id" class="route-card route-button" :class="{ active: selectedRoute === route.task_id }" @click="selectedRoute = selectedRoute === route.task_id ? null : route.task_id">
            <div class="route-card-top">
              <div><span class="strong mono">{{ route.task_no }}</span><span class="tag" style="margin-left: 7px">{{ route.agv_code || '待定' }}</span></div>
              <span class="legend-dot" :style="{ '--dot': routeColor(index) }"></span>
            </div>
            <div v-if="selectedRoute === route.task_id" class="route-path">{{ routeText(route) }}</div>
            <div class="route-path">{{ route.pickup_node ? nodeMap[route.pickup_node]?.code : '--' }} → {{ route.dropoff_node ? nodeMap[route.dropoff_node]?.code : '--' }}</div>
          </button>
        </div>
        <div v-else class="empty">暂无执行中的路线<br />可在调度中心创建并分配任务</div>
        <div class="legend-list mt-16">
          <div class="legend-row"><span class="legend-name"><span class="legend-dot" style="--dot: #41d69a"></span>取货点</span></div>
          <div class="legend-row"><span class="legend-name"><span class="legend-dot" style="--dot: #ffbd4a"></span>放货点</span></div>
          <div class="legend-row"><span class="legend-name"><span class="legend-dot" style="--dot: #9b7cff"></span>充电点</span></div>
        </div>
      </div>
    </aside>
  </div>
</template>
