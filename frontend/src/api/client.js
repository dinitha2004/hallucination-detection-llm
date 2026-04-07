import axios from 'axios'
const api = axios.create({ baseURL: 'http://localhost:8000', timeout: 120000 })
export const detectHallucination = (prompt, maxNewTokens=50) =>
  api.post('/api/detect', { prompt, max_new_tokens: maxNewTokens }).then(r => r.data)
export const getHealth = () => api.get('/api/health').then(r => r.data)
export const getConfig = () => api.get('/api/config').then(r => r.data)
export const updateConfig = (u) => api.post('/api/config', u).then(r => r.data)
