import React, { useEffect, useMemo, useState, useCallback } from 'react';
import { fetchGpus, fetchProcessMetrics, getAuth, clearAuth } from './api.js';
import SimpleTimelineTable from './components/SimpleTimelineTable.tsx';
import PerformanceTable from './components/PerformanceTable.tsx';
import { Tabs, Tab } from '@mui/material';
import DetailDialog from './components/DetailDialog.tsx';

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
      if (type !== 'kernel' && type !== 'scope_end' && type !== 'scope_begin' && type !== 'process_sample') {
        continue;
      }
      const name = payload.kernel || it.processName || payload.name || undefined;
      const ev = {
        timestamp: it.timestamp,
        type,
        pid: it.pid,
        appName: it.app || payload.app || 'app',
        tag: it.tag || payload.tag,
        name,
        usedMemoryMiB: it.usedMemoryMiB,
        durationNs: it.durationNs,
        tsStartNs: payload.ts_start_ns || it.tsStartNs,
        tsEndNs: payload.ts_end_ns || it.tsEndNs,
        gpuName: payload.gpuName || it.gpuName || payload.name || 'GPU',
        uuid: it.gpuUuid || payload.gpuUuid || payload.uuid || 'unknown',
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
        {/* Performance filters moved to top as requested */}
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
        {/* Auto-refresh and manual refresh removed per request */}
      </div>

      {error && (
        <div style={{ color: 'white', background: '#c0392b', padding: 8, borderRadius: 4, marginBottom: 12 }}>
          {error}
        </div>
      )}

      <div style={{ border: '1px solid #ddd', borderRadius: 6 }}>
        <Tabs value={tab} onChange={(_, v) => setTab(v)} aria-label="Views" variant="fullWidth">
          <Tab label="Timeline" />
          <Tab label="Performance" />
        </Tabs>
        <div style={{ padding: 12 }}>
          {tab === 0 && (
            <SimpleTimelineTable
              events={events}
              timeRangeMs={timeRangeMs}
              onBarClick={(bar) => setDetailBar(bar)}
            />
          )}
          {tab === 1 && (
            <PerformanceTable
              events={events}
              timeRangeMs={timeRangeMs}
              appFilter={selectedApp}
              tagFilter={selectedTag}
            />
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
