/**
 * client.js — Axios API Client
 * Wraps all calls to the FastAPI backend.
 *
 * Author: Chalani Dinitha (20211032)
 */

import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 120_000,   // 2 min — model inference can be slow on CPU
})

/**
 * POST /api/detect
 * Run hallucination detection on a prompt.
 *
 * @param {string} prompt
 * @param {number} maxNewTokens
 * @returns {Promise<DetectionResponse>}
 */
export async function detectHallucination(prompt, maxNewTokens = 50) {
  const { data } = await api.post('/api/detect', {
    prompt,
    max_new_tokens: maxNewTokens,
  })
  return data
}

/**
 * GET /api/health
 * Check if the server and pipeline are ready.
 */
export async function getHealth() {
  const { data } = await api.get('/api/health')
  return data
}

/**
 * GET /api/config
 * Get current pipeline configuration.
 */
export async function getConfig() {
  const { data } = await api.get('/api/config')
  return data
}

/**
 * POST /api/config
 * Update threshold at runtime (FR14).
 *
 * @param {{ hallucination_threshold?: number, suspicious_threshold_low?: number }} updates
 */
export async function updateConfig(updates) {
  const { data } = await api.post('/api/config', updates)
  return data
}

export default api
