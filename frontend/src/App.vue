<script setup>
import { onMounted, ref } from 'vue'
import { checkHealth, sendChat, getTools, getMetrics } from './api'

const userId = ref('user_001')
const sessionId = ref(localStorage.getItem('smart_cs_session_id') || '')
const input = ref('')
const loading = ref(false)

const health = ref(null)
const tools = ref([])
const metrics = ref(null)

const messages = ref([
  {
    role: 'assistant',
    content: '您好，我是智能客服多Agent系统。请输入您的问题。',
  },
])

async function refreshStatus() {
  try {
    health.value = await checkHealth()
  } catch (e) {
    health.value = { status: 'offline', error: e.message }
  }

  try {
    const res = await getTools()
    tools.value = res.tools || []
  } catch (e) {
    tools.value = []
  }

  try {
    metrics.value = await getMetrics()
  } catch (e) {
    metrics.value = null
  }
}

async function handleSend() {
  const text = input.value.trim()
  if (!text || loading.value) return

  messages.value.push({
    role: 'user',
    content: text,
  })

  input.value = ''
  loading.value = true

  try {
    const res = await sendChat({
      message: text,
      userId: userId.value,
      sessionId: sessionId.value || null,
    })

    sessionId.value = res.session_id
    localStorage.setItem('smart_cs_session_id', res.session_id)

    messages.value.push({
      role: 'assistant',
      content: res.response,
      intent: res.intent,
      compliance_passed: res.compliance_passed,
    })

    refreshStatus()
  } catch (e) {
    messages.value.push({
      role: 'assistant',
      content: `请求失败：${e.message}`,
    })
  } finally {
    loading.value = false
  }
}

function newSession() {
  sessionId.value = ''
  localStorage.removeItem('smart_cs_session_id')
  messages.value = [
    {
      role: 'assistant',
      content: '已开启新会话，请输入您的问题。',
    },
  ]
}

onMounted(() => {
  refreshStatus()
})
</script>

<template>
  <div class="page">
    <aside class="sidebar">
      <h2>Smart CS</h2>
      <p>智能客服多Agent系统</p>

      <div class="card">
        <label>用户ID</label>
        <input v-model="userId" />
      </div>

      <div class="card">
        <label>会话ID</label>
        <p class="session">{{ sessionId || '暂无会话' }}</p>
      </div>

      <div class="card">
        <label>后端状态</label>
        <p>{{ health?.status || 'checking...' }}</p>
      </div>

      <button @click="newSession">新建会话</button>
      <button @click="refreshStatus">刷新状态</button>
    </aside>

    <main class="chat">
      <header>
        <h1>智能客服控制台</h1>
        <p>意图识别 · 多Agent调度 · 工具调用 · 合规审查</p>
      </header>

      <section class="messages">
        <div
          v-for="(msg, index) in messages"
          :key="index"
          class="message"
          :class="msg.role"
        >
          <div class="bubble">
            <div>{{ msg.content }}</div>

            <div v-if="msg.intent" class="meta">
              意图：{{ msg.intent }} ｜ 合规：
              {{ msg.compliance_passed ? '通过' : '未通过' }}
            </div>
          </div>
        </div>

        <div v-if="loading" class="loading">
          多Agent正在处理...
        </div>
      </section>

      <footer>
        <textarea
          v-model="input"
          placeholder="请输入问题，例如：我想查询订单状态 / 怎么退款 / 如何开户？"
          @keydown.enter.exact.prevent="handleSend"
        />
        <button :disabled="loading || !input.trim()" @click="handleSend">
          {{ loading ? '处理中' : '发送' }}
        </button>
      </footer>
    </main>

    <aside class="right">
      <div class="card">
        <h3>MCP 工具</h3>
        <div v-if="tools.length === 0">暂无工具信息</div>
        <div v-for="tool in tools" :key="tool.name" class="tool">
          <strong>{{ tool.name }}</strong>
          <p>{{ tool.description || '无描述' }}</p>
        </div>
      </div>

      <div class="card">
        <h3>系统指标</h3>
        <pre>{{ metrics ? JSON.stringify(metrics, null, 2) : '暂无指标' }}</pre>
      </div>
    </aside>
  </div>
</template>