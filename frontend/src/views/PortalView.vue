<script setup>
import { reactive, ref } from 'vue'
import { api } from '../api'
import OrderTimeline from '../components/OrderTimeline.vue'
import StatusBadge from '../components/StatusBadge.vue'

const form = reactive({ order_no: '', phone_tail: '' })
const result = ref(null)
const loading = ref(false)
const error = ref('')

function formatTime(value) {
  if (!value) return '--'
  return new Date(value).toLocaleString('zh-CN', { hour12: false })
}

async function lookup() {
  loading.value = true
  error.value = ''
  result.value = null
  try {
    result.value = await api.lookupOrder({
      order_no: form.order_no.trim(),
      phone_tail: form.phone_tail.trim(),
    })
  } catch (err) {
    error.value = err.message
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <section class="portal-shell">
    <div class="portal-hero">
      <span class="portal-kicker">CUSTOMER PORTAL</span>
      <h2>查询您的运输订单</h2>
      <p>输入订单号和下单手机号后四位，查看脱敏后的运输进度与业务时间轴。</p>
    </div>

    <form class="panel portal-form" @submit.prevent="lookup">
      <div class="field">
        <label>订单编号</label>
        <input v-model="form.order_no" class="input mono" required maxlength="32" placeholder="例如：ORD-20261102-ABC123" autocomplete="off" />
      </div>
      <div class="field">
        <label>手机号后四位</label>
        <input v-model="form.phone_tail" class="input mono" required maxlength="4" inputmode="numeric" pattern="\d{4}" placeholder="0000" autocomplete="off" />
      </div>
      <button class="btn btn-primary portal-submit" type="submit" :disabled="loading">{{ loading ? '正在查询...' : '查询订单进度' }}</button>
    </form>

    <div v-if="error" class="notice warning portal-message">{{ error }}</div>

    <section v-if="result" class="panel portal-result">
      <div class="panel-head">
        <div>
          <div class="panel-title mono">{{ result.order_no }}</div>
          <div class="panel-desc">{{ result.customer_name }} · {{ result.customer_phone }}</div>
        </div>
        <StatusBadge :status="result.status" />
      </div>
      <div class="panel-body">
        <div class="portal-summary">
          <div><span>货物</span><strong>{{ result.cargo_name }} · {{ result.cargo_weight }} kg</strong></div>
          <div><span>路线</span><strong>{{ result.pickup_point }} → {{ result.dropoff_point }}</strong></div>
          <div><span>计划时间</span><strong>{{ formatTime(result.scheduled_start) }} 至 {{ formatTime(result.scheduled_end) }}</strong></div>
        </div>
        <div class="portal-timeline">
          <div class="panel-title">运输进度</div>
          <OrderTimeline :events="result.events" />
        </div>
      </div>
    </section>

    <p class="privacy-note">为保护隐私，页面仅显示脱敏客户信息，不公开车辆编号、内部路线和完整手机号。</p>
  </section>
</template>

<style scoped>
.portal-shell { max-width: 920px; margin: 0 auto; }
.portal-hero {
  padding: 30px 32px;
  margin-bottom: 16px;
  overflow: hidden;
  border-radius: 18px;
  color: #edf8f5;
  background:
    radial-gradient(circle at 90% 10%, rgba(84, 199, 190, 0.3), transparent 36%),
    linear-gradient(135deg, #16323b, #0e5d64);
  box-shadow: 0 18px 38px rgba(20, 58, 65, 0.12);
}
.portal-kicker { color: #79d3cb; font-size: 10px; font-weight: 800; letter-spacing: 0.2em; }
.portal-hero h2 { margin: 10px 0 7px; font-size: 27px; letter-spacing: -0.04em; }
.portal-hero p { max-width: 600px; margin: 0; color: #b8d4d1; font-size: 12px; line-height: 1.7; }
.portal-form {
  display: grid;
  grid-template-columns: minmax(0, 1.5fr) 170px auto;
  gap: 13px;
  align-items: end;
  padding: 18px;
}
.portal-submit { min-height: 40px; white-space: nowrap; }
.portal-message { margin-top: 16px; }
.portal-result { margin-top: 16px; }
.portal-summary { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.portal-summary > div { padding: 13px; border: 1px solid #e0e8e4; border-radius: 10px; background: #f8fbfa; }
.portal-summary span { display: block; margin-bottom: 6px; color: #7b898e; font-size: 10px; }
.portal-summary strong { color: #284149; font-size: 12px; line-height: 1.5; }
.portal-timeline { margin-top: 20px; }
.portal-timeline > .panel-title { margin-bottom: 14px; }
.privacy-note { margin: 15px 4px 0; color: #89969b; font-size: 10px; text-align: center; }
@media (max-width: 760px) {
  .portal-form, .portal-summary { grid-template-columns: 1fr; }
  .portal-hero { padding: 23px; }
}
</style>
