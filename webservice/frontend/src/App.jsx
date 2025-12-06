import React, { useEffect, useMemo, useState, useCallback } from 'react';
import { fetchGpus, fetchProcessMetrics, getAuth, clearAuth } from './api.js';
import SimpleTimelineTable from './components/SimpleTimelineTable.tsx';
import PerformanceTable from './components/PerformanceTable.tsx';
import { Tabs, Tab } from '@mui/material';
import DetailDialog from './components/DetailDialog.tsx';
import MetricChart from './components/MetricChart.jsx';

export default function App() {
  const [gpus, setGpus] = useState([]);
  const [selectedGpuKey, setSelectedGpuKey] = useState(''); // prefer gpuId if present else hostname::gpuName
  const [timeRange, setTimeRange] = useState('1h'); // hour-level defaults: 1h, 2h, 6h, 12h, 24h
  const [tab, setTab] = useState(0); // 0 = Timeline, 1 = Performance

  const [processMetrics, setProcessMetrics] = useState([]); // process metrics items
  const [detailBar, setDetailBar] = useState(null);

  // Performance filters
  const [selectedApp, setSelectedApp] = useState('');
  const [selectedTag, setSelectedTag] = useState('');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  // Auto-refresh removed per request

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
        start = new Date(now.getTime() - 1 * 60 * 60 * 1000).toISOString();
        break;
      case '2h':
        start = new Date(now.getTime() - 2 * 60 * 60 * 1000).toISOString();
        break;
      case '6h':
        start = new Date(now.getTime() - 6 * 60 * 60 * 1000).toISOString();
        break;
      case '12h':
        start = new Date(now.getTime() - 12 * 60 * 60 * 1000).toISOString();
        break;
      case '24h':
      default:
        start = new Date(now.getTime() - 24 * 60 * 60 * 1000).toISOString();
    }

    return { start, end };
  };

  const loadMetrics = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const sel = gpus.find(g => (g.gpuId || `${g.hostname || ''}::${g.gpuName || ''}`) === selectedGpuKey);
      const { start, end } = getTimeRangeParams();
      // Shorter windows imply fewer items; keep reasonable cap
      const baseParams = { start, end, limit: 5000, order: 'asc' };
      if (sel?.gpuId) baseParams.gpuId = sel.gpuId;
      else {
        if (sel?.hostname) baseParams.hostname = sel.hostname;
        if (sel?.gpuName) baseParams.gpuName = sel.gpuName;
      }

      // Load process events only (unified endpoint)
      const processData = await fetchProcessMetrics(baseParams);
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
  }, [loadMetrics, gpus.length, selectedGpuKey]);

  // Adapter: map backend items into MetricEvent shape used by visual components
  const events = useMemo(() => {
    const out = [];
    for (const it of processMetrics || []) {
      const payload = it.payload || {};
      const type = it.type; // rely on backend-provided type directly
      if (type !== 'kernel' && type !== 'scope_end' && type !== 'scope_begin' && type !== 'scope_sample' && type !== 'process_sample' && type !== 'system_sample') {
        continue;
      }
      const name = payload.kernel || it.processName || payload.name || undefined;
      // Normalize devices from backend (snake_case fields)
      const rawDevices = Array.isArray(it.devices) ? it.devices : (Array.isArray(payload.devices) ? payload.devices : []);
      const devices = rawDevices.map((d) => ({
        id: d.id,
        uuid: d.uuid,
        name: d.name,
        pci_bus: d.pci_bus,
        used_mib: d.used_mib,
        free_mib: d.free_mib,
        total_mib: d.total_mib,
        util_gpu: d.util_gpu,
        util_mem: d.util_mem,
        temp_c: d.temp_c,
        power_mw: d.power_mw,
        clk_gfx: d.clk_gfx,
        clk_sm: d.clk_sm,
        clk_mem: d.clk_mem,
      }));
      const sumUsedFromDevices = devices.length ? devices.reduce((acc, d) => acc + (typeof d.used_mib === 'number' ? d.used_mib : 0), 0) : undefined;
      const ev = {
        timestamp: it.timestamp,
        type,
        pid: it.pid,
        appName: it.app || payload.app || 'app',
        tag: it.tag || payload.tag,
        name,
        usedMemoryMiB: (typeof sumUsedFromDevices === 'number' && sumUsedFromDevices > 0) ? sumUsedFromDevices : it.usedMemoryMiB,
        durationNs: it.durationNs,
        tsStartNs: payload.ts_start_ns || it.tsStartNs,
        tsEndNs: payload.ts_end_ns || it.tsEndNs,
        gpuName: payload.gpuName || it.gpuName || payload.name || 'GPU',
        uuid: it.gpuUuid || payload.gpuUuid || payload.uuid || 'unknown',
        devices,
        // include extra metadata from backend for detail view
        extra: it.extra || payload,
      };
      out.push(ev);
    }
    return out;
  }, [processMetrics]);

  // Program and Name options derived from current events
  const programOptions = useMemo(() => {
    const set = new Set(events.map(e => e.appName).filter(Boolean));
    return Array.from(set).sort();
  }, [events]);

  const nameOptions = useMemo(() => {
    // "Name" means tag selection as per previous grouping
    const filtered = selectedApp ? events.filter(e => e.appName === selectedApp) : events;
    const set = new Set(filtered.map(e => e.tag || 'default'));
    return Array.from(set).sort();
  }, [events, selectedApp]);

  // Initialize/adjust selectedApp and selectedTag when options change
  useEffect(() => {
    if (!selectedApp && programOptions.length > 0) {
      setSelectedApp(programOptions[0]);
    } else if (selectedApp && !programOptions.includes(selectedApp)) {
      setSelectedApp(programOptions[0] || '');
    }
  }, [programOptions, selectedApp]);

  useEffect(() => {
    if (!selectedTag && nameOptions.length > 0) {
      setSelectedTag(nameOptions[0]);
    } else if (selectedTag && !nameOptions.includes(selectedTag)) {
      setSelectedTag(nameOptions[0] || '');
    }
  }, [nameOptions, selectedTag]);

  const timeRangeMs = useMemo(() => {
    switch (timeRange) {
      case '1h': return 1 * 60 * 60 * 1000;
      case '2h': return 2 * 60 * 60 * 1000;
      case '6h': return 6 * 60 * 60 * 1000;
      case '12h': return 12 * 60 * 60 * 1000;
      case '24h':
      default: return 24 * 60 * 60 * 1000;
    }
  }, [timeRange]);

  const auth = getAuth();
  
  // Build GPU Utilization datasets from system_sample events
  const gpuUtilDatasets = useMemo(() => {
    const nowMs = Date.now();
    const startMs = nowMs - timeRangeMs;
    // Collect per-device series
    const seriesMap = new Map();
    // Apply Program/Name filters for consistency
    const filteredEvents = events.filter(it => (!selectedApp || it.appName === selectedApp) && (!selectedTag || (it.tag || 'default') === selectedTag));
    for (const ev of filteredEvents) {
      if (ev.type !== 'system_sample') continue;
      const t = new Date(ev.timestamp).getTime();
      if (!Number.isFinite(t)) continue;
      // keep only points within selected window
      if (t < startMs || t > nowMs) continue;
      const devs = Array.isArray(ev.devices) ? ev.devices : [];
      for (const d of devs) {
        const key = d.uuid || d.name || 'unknown';
        const util = typeof d.util_gpu === 'number' ? d.util_gpu : null;
        if (util == null) continue;
        if (!seriesMap.has(key)) {
          seriesMap.set(key, { label: `${d.name || 'GPU'} (${d.uuid || 'unknown'})`, points: [] });
        }
        seriesMap.get(key).points.push({ x: t, y: util });
      }
    }
    // Assign colors (basic palette)
    const colors = ['rgb(59,130,246)','rgb(234,88,12)','rgb(16,185,129)','rgb(139,92,246)','rgb(244,63,94)','rgb(245,158,11)'];
    let i = 0;
    return Array.from(seriesMap.values()).map((s) => ({ ...s, color: colors[i++ % colors.length], points: s.points.sort((a,b)=>a.x-b.x) }));
  }, [events, selectedApp, selectedTag, timeRangeMs]);

  // Build Temperature (°C) datasets
  const tempDatasets = useMemo(() => {
    const nowMs = Date.now();
    const startMs = nowMs - timeRangeMs;
    const seriesMap = new Map();
    const filteredEvents = events.filter(it => (!selectedApp || it.appName === selectedApp) && (!selectedTag || (it.tag || 'default') === selectedTag));
    for (const ev of filteredEvents) {
      if (ev.type !== 'system_sample') continue;
      const t = new Date(ev.timestamp).getTime();
      if (!Number.isFinite(t) || t < startMs || t > nowMs) continue;
      const devs = Array.isArray(ev.devices) ? ev.devices : [];
      for (const d of devs) {
        const key = d.uuid || d.name || 'unknown';
        const v = typeof d.temp_c === 'number' ? d.temp_c : null;
        if (v == null) continue;
        if (!seriesMap.has(key)) seriesMap.set(key, { label: `${d.name || 'GPU'} (${d.uuid || 'unknown'})`, points: [] });
        seriesMap.get(key).points.push({ x: t, y: v });
      }
    }
    const colors = ['rgb(59,130,246)','rgb(234,88,12)','rgb(16,185,129)','rgb(139,92,246)','rgb(244,63,94)','rgb(245,158,11)'];
    let i = 0;
    return Array.from(seriesMap.values()).map((s) => ({ ...s, color: colors[i++ % colors.length], points: s.points.sort((a,b)=>a.x-b.x) }));
  }, [events, selectedApp, selectedTag, timeRangeMs]);

  // Build Memory Utilization (%) datasets
  const memUtilDatasets = useMemo(() => {
    const nowMs = Date.now();
    const startMs = nowMs - timeRangeMs;
    const seriesMap = new Map();
    const filteredEvents = events.filter(it => (!selectedApp || it.appName === selectedApp) && (!selectedTag || (it.tag || 'default') === selectedTag));
    for (const ev of filteredEvents) {
      if (ev.type !== 'system_sample') continue;
      const t = new Date(ev.timestamp).getTime();
      if (!Number.isFinite(t) || t < startMs || t > nowMs) continue;
      const devs = Array.isArray(ev.devices) ? ev.devices : [];
      for (const d of devs) {
        const key = d.uuid || d.name || 'unknown';
        const v = typeof d.util_mem === 'number' ? d.util_mem : null;
        if (v == null) continue;
        if (!seriesMap.has(key)) seriesMap.set(key, { label: `${d.name || 'GPU'} (${d.uuid || 'unknown'})`, points: [] });
        seriesMap.get(key).points.push({ x: t, y: v });
      }
    }
    const colors = ['rgb(59,130,246)','rgb(234,88,12)','rgb(16,185,129)','rgb(139,92,246)','rgb(244,63,94)','rgb(245,158,11)'];
    let i = 0;
    return Array.from(seriesMap.values()).map((s) => ({ ...s, color: colors[i++ % colors.length], points: s.points.sort((a,b)=>a.x-b.x) }));
  }, [events, selectedApp, selectedTag, timeRangeMs]);

  // Helper to build clock datasets by field name
  const buildClockDatasets = useCallback((field) => {
    const nowMs = Date.now();
    const startMs = nowMs - timeRangeMs;
    const seriesMap = new Map();
    const filteredEvents = events.filter(it => (!selectedApp || it.appName === selectedApp) && (!selectedTag || (it.tag || 'default') === selectedTag));
    for (const ev of filteredEvents) {
      if (ev.type !== 'system_sample') continue;
      const t = new Date(ev.timestamp).getTime();
      if (!Number.isFinite(t) || t < startMs || t > nowMs) continue;
      const devs = Array.isArray(ev.devices) ? ev.devices : [];
      for (const d of devs) {
        const key = d.uuid || d.name || 'unknown';
        const vRaw = d?.[field];
        const v = typeof vRaw === 'number' ? vRaw : null;
        if (v == null) continue;
        if (!seriesMap.has(key)) seriesMap.set(key, { label: `${d.name || 'GPU'} (${d.uuid || 'unknown'})`, points: [] });
        seriesMap.get(key).points.push({ x: t, y: v });
      }
    }
    const colors = ['rgb(59,130,246)','rgb(234,88,12)','rgb(16,185,129)','rgb(139,92,246)','rgb(244,63,94)','rgb(245,158,11)'];
    let i = 0;
    return Array.from(seriesMap.values()).map((s) => ({ ...s, color: colors[i++ % colors.length], points: s.points.sort((a,b)=>a.x-b.x) }));
  }, [events, selectedApp, selectedTag, timeRangeMs]);

  const clkGfxDatasets = useMemo(() => buildClockDatasets('clk_gfx'), [buildClockDatasets]);
  const clkSmDatasets = useMemo(() => buildClockDatasets('clk_sm'), [buildClockDatasets]);
  const clkMemDatasets = useMemo(() => buildClockDatasets('clk_mem'), [buildClockDatasets]);
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
            <option value="2h">Last 2 Hours</option>
            <option value="6h">Last 6 Hours</option>
            <option value="12h">Last 12 Hours</option>
            <option value="24h">Last 24 Hours</option>
          </select>
        </label>
        {/* Auto-refresh and manual refresh removed per request */}
      </div>

      {error && (
        <div style={{ color: 'white', background: '#c0392b', padding: 8, borderRadius: 4, marginBottom: 12 }}>
          {error}
        </div>
      )}

      <div style={{ border: '1px solid #ddd', borderRadius: 6 }}>
        <Tabs value={tab} onChange={(_, v) => setTab(v)} aria-label="Views" variant="fullWidth">
          <Tab label="GPU Utilization" />
          <Tab label="Timeline" />
          <Tab label="Performance" />
        </Tabs>
        <div style={{ padding: 12 }}>
          {tab === 0 && (
            <>
              <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12 }}>
                <label>
                  Program:
                  <select value={selectedApp} onChange={(e) => setSelectedApp(e.target.value)} style={{ marginLeft: 6 }}>
                    {programOptions.map((opt) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Name:
                  <select value={selectedTag} onChange={(e) => setSelectedTag(e.target.value)} style={{ marginLeft: 6 }}>
                    {nameOptions.map((opt) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </label>
              </div>
              <div>
                <h3 style={{ margin: '4px 0 8px' }}>GPU Utilization (%)</h3>
                <MetricChart
                  datasets={gpuUtilDatasets}
                  label="GPU Utilization (%)"
                  timeStartMs={Date.now() - timeRangeMs}
                  timeEndMs={Date.now()}
                />
              </div>
              <div>
                <h3 style={{ margin: '16px 0 8px' }}>Memory Utilization (%)</h3>
                <MetricChart
                  datasets={memUtilDatasets}
                  label="Memory Utilization (%)"
                  timeStartMs={Date.now() - timeRangeMs}
                  timeEndMs={Date.now()}
                />
              </div>
              <div>
                <h3 style={{ margin: '16px 0 8px' }}>Temperature (°C)</h3>
                <MetricChart
                  datasets={tempDatasets}
                  label="Temperature (°C)"
                  timeStartMs={Date.now() - timeRangeMs}
                  timeEndMs={Date.now()}
                />
              </div>
              <div>
                <h3 style={{ margin: '16px 0 8px' }}>Clock - Graphics (MHz)</h3>
                <MetricChart
                  datasets={clkGfxDatasets}
                  label="Clock - Graphics (MHz)"
                  timeStartMs={Date.now() - timeRangeMs}
                  timeEndMs={Date.now()}
                />
              </div>
              <div>
                <h3 style={{ margin: '16px 0 8px' }}>Clock - SM (MHz)</h3>
                <MetricChart
                  datasets={clkSmDatasets}
                  label="Clock - SM (MHz)"
                  timeStartMs={Date.now() - timeRangeMs}
                  timeEndMs={Date.now()}
                />
              </div>
              <div>
                <h3 style={{ margin: '16px 0 8px' }}>Clock - Memory (MHz)</h3>
                <MetricChart
                  datasets={clkMemDatasets}
                  label="Clock - Memory (MHz)"
                  timeStartMs={Date.now() - timeRangeMs}
                  timeEndMs={Date.now()}
                />
              </div>
            </>
          )}
          {tab === 1 && (
            <>
              <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12 }}>
                <label>
                  Program:
                  <select value={selectedApp} onChange={(e) => setSelectedApp(e.target.value)} style={{ marginLeft: 6 }}>
                    {programOptions.map((opt) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Name:
                  <select value={selectedTag} onChange={(e) => setSelectedTag(e.target.value)} style={{ marginLeft: 6 }}>
                    {nameOptions.map((opt) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </label>
              </div>
              <SimpleTimelineTable
                events={events.filter(it => (!selectedApp || it.appName === selectedApp) && (!selectedTag || (it.tag || 'default') === selectedTag))}
                timeRangeMs={timeRangeMs}
                onBarClick={(bar) => setDetailBar(bar)}
              />
            </>
          )}
          {tab === 2 && (
            <>
              <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12 }}>
                <label>
                  Program:
                  <select value={selectedApp} onChange={(e) => setSelectedApp(e.target.value)} style={{ marginLeft: 6 }}>
                    {programOptions.map((opt) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Name:
                  <select value={selectedTag} onChange={(e) => setSelectedTag(e.target.value)} style={{ marginLeft: 6 }}>
                    {nameOptions.map((opt) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </label>
              </div>
              <PerformanceTable
                events={events}
                timeRangeMs={timeRangeMs}
                appFilter={selectedApp}
                tagFilter={selectedTag}
              />
            </>
          )}
        </div>
      </div>

      <div style={{ marginTop: 12, fontSize: 12, color: '#666' }}>
        Backend is Spring Boot at http://localhost:8080 (proxied as /api). Endpoint used: GET /metrics.
      </div>

      <DetailDialog open={!!detailBar} bar={detailBar} onClose={() => setDetailBar(null)} />
    </div>
  );
}
