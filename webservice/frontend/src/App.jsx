import React, { useEffect, useMemo, useState, useCallback } from 'react';
import { fetchGpus, fetchMetrics, fetchProcessMetrics, mapItemsToSeries, getAuth, clearAuth } from './api.js';
import MetricChart from './components/MetricChart.jsx';

const METRICS = [
  { key: 'utilization', label: 'Utilization', fields: ['gpuPercent', 'memoryPercent'], defaultField: 'gpuPercent' },
  { key: 'memory', label: 'Memory', fields: ['usedMiB', 'freeMiB', 'totalMiB'], defaultField: 'usedMiB' },
  { key: 'power', label: 'Power', fields: ['watts'], defaultField: 'watts' },
  { key: 'temperature', label: 'Temperature', fields: ['celsius'], defaultField: 'celsius' },
  { key: 'clocks', label: 'Clocks', fields: ['graphicsMHz', 'memoryMHz'], defaultField: 'graphicsMHz' },
];

export default function App() {
  const [gpus, setGpus] = useState([]);
  const [selectedGpuKey, setSelectedGpuKey] = useState(''); // prefer gpuId if present else hostname::gpuName
  const [timeRange, setTimeRange] = useState('1h'); // 1h, 6h, 24h, 7d, 30d

  const [metricsData, setMetricsData] = useState({}); // { [metricKey]: items[] }
  const [processMetrics, setProcessMetrics] = useState([]); // process metrics items

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [intervalSec, setIntervalSec] = useState(5);

  useEffect(() => {
    // load GPUs on mount
    (async () => {
      try {
        const data = await fetchGpus();
        const gArr = data.gpus || [];
        setGpus(gArr);
        if (gArr.length > 0) {
          const first = gArr[0];
          const key = first.gpuId || `${first.hostname || ''}::${first.gpuName || ''}`;
          setSelectedGpuKey(key);
        }
      } catch (e) {
        setError(String(e));
      }
    })();
  }, []);

  const getTimeRangeParams = () => {
    const now = new Date();
    const end = now.toISOString();
    let start;

    switch (timeRange) {
      case '1h':
        start = new Date(now.getTime() - 60 * 60 * 1000).toISOString();
        break;
      case '6h':
        start = new Date(now.getTime() - 6 * 60 * 60 * 1000).toISOString();
        break;
      case '24h':
        start = new Date(now.getTime() - 24 * 60 * 60 * 1000).toISOString();
        break;
      case '7d':
        start = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000).toISOString();
        break;
      case '30d':
        start = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000).toISOString();
        break;
      default:
        start = new Date(now.getTime() - 60 * 60 * 1000).toISOString();
    }

    return { start, end };
  };

  const loadMetrics = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const sel = gpus.find(g => (g.gpuId || `${g.hostname || ''}::${g.gpuName || ''}`) === selectedGpuKey);
      const { start, end } = getTimeRangeParams();
      const baseParams = { start, end, limit: 10000, order: 'asc' };
      if (sel?.gpuId) baseParams.gpuId = sel.gpuId;
      else {
        if (sel?.hostname) baseParams.hostname = sel.hostname;
        if (sel?.gpuName) baseParams.gpuName = sel.gpuName;
      }

      // Load all metrics and process metrics in parallel
      const [metricResults, processData] = await Promise.all([
        Promise.all(
          METRICS.map(async (m) => {
            const data = await fetchMetrics({ ...baseParams, metric: m.key });
            return { key: m.key, items: data.items || [] };
          })
        ),
        fetchProcessMetrics(baseParams)
      ]);

      const newMetricsData = {};
      metricResults.forEach(r => {
        newMetricsData[r.key] = r.items;
      });
      setMetricsData(newMetricsData);
      setProcessMetrics(processData.items || []);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [gpus, selectedGpuKey, timeRange]);

  useEffect(() => {
    if (gpus.length === 0 || !selectedGpuKey) return;
    loadMetrics();
    if (!autoRefresh) return;
    const id = setInterval(loadMetrics, intervalSec * 1000);
    return () => clearInterval(id);
  }, [loadMetrics, autoRefresh, intervalSec, gpus.length, selectedGpuKey]);

  const palette = ['rgb(75, 192, 192)', 'rgb(255, 99, 132)', 'rgb(54, 162, 235)', 'rgb(255, 159, 64)', 'rgb(153, 102, 255)', 'rgb(201, 203, 207)', 'rgb(255, 205, 86)'];

  const allChartsData = useMemo(() => {
    return METRICS.map(m => {
      const items = metricsData[m.key] || [];
      const flds = m.fields || [];
      const datasets = flds.map((f, i) => ({
        label: `${f}`,
        points: mapItemsToSeries(items, f),
        color: palette[i % palette.length],
      }));
      return { metric: m, datasets };
    });
  }, [metricsData]);

  const processChartData = useMemo(() => {
    // Group process metrics by process name
    const processGroups = {};
    processMetrics.forEach(item => {
      const procName = item.processName || `PID ${item.pid}`;
      if (!processGroups[procName]) {
        processGroups[procName] = [];
      }
      const timestamp = item.timestamp;
      const ms = timestamp ? new Date(timestamp).getTime() : null;
      const memory = item.usedMemoryMiB;
      if (ms && typeof memory === 'number') {
        processGroups[procName].push({ x: ms, y: memory });
      }
    });

    // Create datasets for each process
    const processNames = Object.keys(processGroups);
    const datasets = processNames.map((name, i) => ({
      label: name,
      points: processGroups[name],
      color: palette[i % palette.length],
    }));

    return datasets;
  }, [processMetrics]);

  const auth = getAuth();
  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', padding: '16px', maxWidth: 1000, margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <h1 style={{ margin: 0 }}>GPU Monitoring</h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 14 }}>
          <span>{auth?.name || auth?.email}</span>
          <button onClick={() => { clearAuth(); window.location.hash = '#/login'; }}>Logout</button>
        </div>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center', marginBottom: 12 }}>
        <label>
          GPU:
          <select value={selectedGpuKey} onChange={(e) => setSelectedGpuKey(e.target.value)} style={{ marginLeft: 6 }}>
            {gpus.map((g, idx) => {
              const key = g.gpuId || `${g.hostname || ''}::${g.gpuName || ''}`;
              const label = g.gpuId ? `${g.gpuName || 'GPU'} (${g.gpuId})` : `${g.hostname || 'host'} :: ${g.gpuName || 'gpu'}`;
              return <option key={key || idx} value={key}>{label}</option>;
            })}
          </select>
        </label>
        <label>
          Time Range:
          <select value={timeRange} onChange={(e) => setTimeRange(e.target.value)} style={{ marginLeft: 6 }}>
            <option value="1h">Last 1 Hour</option>
            <option value="6h">Last 6 Hours</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} /> Auto refresh
        </label>
        <label>
          Every:
          <input type="number" min={1} max={60} step={1} value={intervalSec} onChange={(e) => setIntervalSec(Number(e.target.value))} style={{ width: 60, marginLeft: 6 }} /> sec
        </label>
        <button onClick={loadMetrics} disabled={loading}>
          {loading ? 'Loading…' : 'Refresh'}
        </button>
      </div>

      {error && (
        <div style={{ color: 'white', background: '#c0392b', padding: 8, borderRadius: 4, marginBottom: 12 }}>
          {error}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: 16 }}>
        {allChartsData.map(({ metric, datasets }) => (
          <div key={metric.key} style={{ border: '1px solid #ddd', borderRadius: 6, padding: 12 }}>
            <h3 style={{ margin: '0 0 12px 0', fontSize: 16 }}>{metric.label}</h3>
            <MetricChart datasets={datasets} label={metric.label} />
          </div>
        ))}

        {/* Process Memory Chart */}
        <div style={{ border: '1px solid #ddd', borderRadius: 6, padding: 12 }}>
          <h3 style={{ margin: '0 0 12px 0', fontSize: 16 }}>Process Memory Usage</h3>
          {processChartData.length > 0 ? (
            <MetricChart datasets={processChartData} label="Memory (MiB)" />
          ) : (
            <div style={{ height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#999' }}>
              No process data available
            </div>
          )}
        </div>
      </div>

      <div style={{ marginTop: 12, fontSize: 12, color: '#666' }}>
        Backend is Spring Boot at http://localhost:8080 (proxied as /api). Endpoints used: GET /metrics, GET /health.
      </div>
    </div>
  );
}
