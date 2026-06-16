const BASE = '/api'

async function request(url, options = {}) {
  const res = await fetch(url, options)

  if (!res.ok) {
    let msg = '请求失败'
    try {
      const data = await res.json()
      msg = data.detail || msg
    } catch (_) {}
    throw new Error(msg)
  }

  return res.json()
}

export async function checkHealth() {
  return request('/health')
}

export async function sendChat({ message, userId, sessionId }) {
  return request(`${BASE}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message,
      user_id: userId || 'user_001',
      session_id: sessionId || null,
    }),
  })
}

export async function getTools() {
  return request(`${BASE}/tools`)
}

export async function getMetrics() {
  return request(`${BASE}/metrics`)
}