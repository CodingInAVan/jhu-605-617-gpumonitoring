const API_BASE = import.meta.env.VITE_API_BASE || '/api';

// Simple auth helpers (localStorage)
const AUTH_KEY = 'auth.user';
export function getAuth() {
  try { return JSON.parse(localStorage.getItem(AUTH_KEY) || 'null'); } catch { return null; }
}
export function setAuth(user) {
  localStorage.setItem(AUTH_KEY, JSON.stringify(user));
}
export function clearAuth() {
  localStorage.removeItem(AUTH_KEY);
}
function authHeaders() {
  const auth = getAuth();
  const headers = { };
  if (auth?.apiKey) headers['X-API-Key'] = auth.apiKey;
  return headers;
}

export async function fetchGpus({ start = null, end = null } = {}) {
  const params = new URLSearchParams();
  if (start) params.set('start', start);
  if (end) params.set('end', end);
  const res = await fetch(`${API_BASE}/gpus?${params.toString()}`, {
    headers: { ...authHeaders() },
  });
  if (!res.ok) throw new Error(`Failed to fetch GPUs: ${res.status}`);
  const data = await res.json();
  const list = Array.isArray(data?.gpus) ? data.gpus : [];
  // Map backend fields (gpuUuid, gpuName, lastSeen) to frontend expectations
  const gpus = list.map((g) => ({
    gpuId: g.gpuUuid ?? null, // keep legacy name used by UI selection
    gpuUuid: g.gpuUuid ?? null,
    gpuName: g.gpuName ?? null,
    lastSeen: g.lastSeen ?? null,
  }));
  return { gpus };
}

export async function fetchMetrics({ gpuId = null, hostname = null, gpuName = null, metric = null, start = null, end = null, limit = 500, order = 'asc', aggregate = null, field = null }) {
  const params = new URLSearchParams();
  if (metric) params.set('metric', metric);
  if (gpuId) params.set('gpuId', gpuId);
  if (hostname) params.set('hostname', hostname);
  if (gpuName) params.set('gpuName', gpuName);
  if (start) params.set('start', start);
  if (end) params.set('end', end);
  if (aggregate) params.set('aggregate', aggregate);
  if (field) params.set('field', field);
  params.set('limit', String(limit));
  params.set('order', order);
  const res = await fetch(`${API_BASE}/metrics?${params.toString()}`, {
    headers: { ...authHeaders() },
  });
  if (!res.ok) throw new Error(`Failed to fetch metrics: ${res.status}`);
  return res.json();
}

export function normalizeIsoToMsUtc(tsRaw) {
  if (!tsRaw || typeof tsRaw !== 'string') return null;
  let ts = tsRaw.trim();
  ts = ts.replace(' ', 'T');
  const m = ts.match(/^(.*T\d{2}:\d{2}:\d{2})(\.(\d+))?(Z|[+-]\d{2}:\d{2})?$/);
  if (m) {
    const prefix = m[1];
    const frac = m[3] || '';
    const tz = m[4] || '';
    const frac3 = frac ? '.' + frac.slice(0, 3).padEnd(3, '0') : '';
    const tzFinal = tz || 'Z';
    ts = `${prefix}${frac3}${tzFinal}`;
  } else {
    ts = ts.replace(/(\.\d{3})\d+/, '$1');
    if (!/(Z|[+-]\d{2}:\d{2})$/i.test(ts)) ts += 'Z';
  }
  const d = new Date(ts);
  if (isNaN(d.getTime())) return null;
  return d.getTime();
}

export function mapItemsToSeries(items, field) {
  const points = [];
  for (const it of items) {
    // Try to get value from payload first (for new consolidated format)
    let v = it?.extra?.[field];

    // If not found in payload and field matches specific patterns, map to consolidated field names
    if (v === undefined || v === null) {
      const fieldMapping = {
        'usedMiB': 'usedTotalMemoryMiB',
        'totalMiB': 'totalMemoryMiB',
        'freeMiB': 'freeMemoryMiB',
        'gpuPercent': 'gpuUtilPercent',
        'memoryPercent': 'memUtilPercent',
        'celsius': 'temperatureCelsius',
        'watts': 'powerMilliwatts',
        'graphicsMHz': 'graphicsClockMHz',
        'memoryMHz': 'memClockMHz',
        'smMHz': 'smClockMHz'
      };

      const mappedField = fieldMapping[field] || field;
      v = it?.extra?.[mappedField];

      // Convert watts from milliwatts if needed
      if (field === 'watts' && mappedField === 'powerMilliwatts' && typeof v === 'number') {
        v = v / 1000.0;
      }
    }

    const rawTs = it?.timestamp || it?.extra?.timestamp;
    const ms = normalizeIsoToMsUtc(rawTs);
    const processName = it?.extra?.processName;

    if (typeof v === 'number' && ms !== null) {
      const point = { x: ms, y: v };
      if (processName) {
        point.processName = processName;
      }
      points.push(point);
    }
  }
  return points;
}

// Fetch process metrics
export async function fetchProcessMetrics({ gpuId = null, hostname = null, gpuName = null, start = null, end = null, limit = 1000, order = 'asc' }) {
  const params = new URLSearchParams();
  if (gpuId) params.set('gpuId', gpuId);
  if (hostname) params.set('hostname', hostname);
  if (gpuName) params.set('gpuName', gpuName);
  if (start) params.set('start', start);
  if (end) params.set('end', end);
  params.set('limit', String(limit));
  params.set('order', order);
  // In the new backend, process events are served via unified /metrics
  const res = await fetch(`${API_BASE}/metrics?${params.toString()}`, {
    headers: { ...authHeaders() },
  });
  if (!res.ok) throw new Error(`Failed to fetch process metrics: ${res.status}`);
  return res.json();
}

// Register a new user and receive an API key
export async function registerUser({ email, name, password }) {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, name, password }),
  });
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (_) { /* ignore */ }
  if (!res.ok) {
    const msg = data?.message || data?.error || text || `Register failed: ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

// Login existing user
export async function loginUser({ email, password }) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (_) { /* ignore */ }
  if (!res.ok) {
    const msg = data?.message || data?.error || text || `Login failed: ${res.status}`;
    throw new Error(msg);
  }
  return data;
}
