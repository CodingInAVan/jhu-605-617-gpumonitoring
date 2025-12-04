import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { MetricEvent } from '../types/metrics';

export interface DashboardProps {
  events: MetricEvent[];
  timeRangeMs: number;
  nowMs?: number;
  topN?: number;
  appFilter?: string; // selected program (appName)
  tagFilter?: string; // selected name (tag)
}

function formatMsDuration(ms: number): string {
  if (ms < 1) return `${Math.round(ms * 1000)}µs`;
  if (ms < 1000) return `${ms.toFixed(3)} ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(3)} s`;
  const m = Math.floor(s / 60);
  const ss = (s % 60).toFixed(0);
  return `${m}m ${ss}s`;
}

function useWindowed(events: MetricEvent[], timeEnd: number, timeRangeMs: number) {
  const timeStart = timeEnd - timeRangeMs;
  return useMemo(() => {
    return events.filter((e) => {
      const t = new Date(e.timestamp).getTime();
      return Number.isFinite(t) && t >= timeStart && t <= timeEnd;
    });
  }, [events, timeStart, timeEnd]);
}

export default function Dashboard({ events, timeRangeMs, nowMs, topN = 10, appFilter, tagFilter }: DashboardProps) {
  const timeEnd = nowMs ?? Date.now();
  const windowed = useWindowed(events, timeEnd, timeRangeMs);
  const filtered = useMemo(() => {
    return windowed.filter((e) => {
      if (appFilter && e.appName !== appFilter) return false;
      if (tagFilter && (e.tag || 'default') !== tagFilter) return false;
      return true;
    });
  }, [windowed, appFilter, tagFilter]);

  // 1) Top memory usage by process (group by app+pid, max usedMemoryMiB from process_sample)
  const topProcessMem = useMemo(() => {
    const byKey = new Map<string, { key: string; app: string; pid: number; tag?: string; peak: number }>();
    for (const ev of filtered) {
      if (ev.type !== 'process_sample') continue;
      const used = typeof ev.usedMemoryMiB === 'number' ? ev.usedMemoryMiB : undefined;
      if (used == null) continue;
      const key = `${ev.appName}|${ev.pid}`;
      const rec = byKey.get(key) || { key, app: ev.appName, pid: ev.pid, tag: ev.tag, peak: used };
      rec.peak = Math.max(rec.peak, used);
      byKey.set(key, rec);
    }
    const arr = Array.from(byKey.values()).sort((a, b) => b.peak - a.peak).slice(0, topN);
    return arr;
  }, [filtered, topN]);

  // 2) Top N memory events (individual samples by usedMemoryMiB)
  const topMemEvents = useMemo(() => {
    const arr = filtered
      .filter((e) => e.type === 'process_sample' && typeof e.usedMemoryMiB === 'number')
      .map((e) => ({
        app: e.appName,
        pid: e.pid,
        tag: e.tag,
        used: e.usedMemoryMiB as number,
        t: new Date(e.timestamp).getTime(),
      }))
      .sort((a, b) => b.used - a.used)
      .slice(0, topN);
    return arr;
  }, [filtered, topN]);

  // Helpers for durations
  const nsToMs = (ns?: number) => (ns && ns > 0 ? ns / 1_000_000 : 0);

  // 3) Top N duration scope events
  const topScopeDuration = useMemo(() => {
    const arr = filtered
      .filter((e) => e.type === 'scope_end')
      .map((e) => ({ name: e.name || 'scope', app: e.appName, pid: e.pid, tag: e.tag, durMs: nsToMs(e.durationNs) }))
      .sort((a, b) => b.durMs - a.durMs)
      .slice(0, topN);
    return arr;
  }, [filtered, topN]);

  // 4) Top N memory kernel events (by usedMemoryMiB if available)
  const topKernelMem = useMemo(() => {
    const arr = filtered
      .filter((e) => e.type === 'kernel' && typeof e.usedMemoryMiB === 'number')
      .map((e) => ({ name: e.name || 'kernel', app: e.appName, pid: e.pid, tag: e.tag, used: e.usedMemoryMiB as number }))
      .sort((a, b) => b.used - a.used)
      .slice(0, topN);
    return arr;
  }, [filtered, topN]);

  // 5) Top N duration kernel events
  const topKernelDuration = useMemo(() => {
    const arr = filtered
      .filter((e) => e.type === 'kernel')
      .map((e) => ({ name: e.name || 'kernel', app: e.appName, pid: e.pid, tag: e.tag, durMs: nsToMs(e.durationNs) }))
      .sort((a, b) => b.durMs - a.durMs)
      .slice(0, topN);
    return arr;
  }, [filtered, topN]);

  function barOptionH(categories: string[], values: number[], title: string, valueFormatter?: (v: number) => string) {
    return {
      title: { text: title, left: 'center' },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        valueFormatter: valueFormatter,
      },
      grid: { left: 100, right: 20, top: 40, bottom: 30 },
      xAxis: { type: 'value' },
      yAxis: { type: 'category', data: categories },
      series: [{ type: 'bar', data: values, itemStyle: { color: '#3b82f6' } }],
    } as any;
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
      {/* Top memory usage by process */}
      <div style={{ border: '1px solid #eee', borderRadius: 6, padding: 8 }}>
        <ReactECharts
          notMerge
          option={barOptionH(
            topProcessMem.map((r) => `${r.app} (PID ${r.pid})`),
            topProcessMem.map((r) => r.peak),
            'Top Memory Usage by Process (MiB)'
          )}
          style={{ height: 300 }}
        />
      </div>

      {/* Top memory events */}
      <div style={{ border: '1px solid #eee', borderRadius: 6, padding: 8 }}>
        <ReactECharts
          notMerge
          option={barOptionH(
            topMemEvents.map((r) => `${r.app} (PID ${r.pid})`),
            topMemEvents.map((r) => r.used),
            'Top Memory Events (MiB)'
          )}
          style={{ height: 300 }}
        />
      </div>

      {/* Top scope durations */}
      <div style={{ border: '1px solid #eee', borderRadius: 6, padding: 8 }}>
        <ReactECharts
          notMerge
          option={barOptionH(
            topScopeDuration.map((r) => `${r.name}`),
            topScopeDuration.map((r) => r.durMs),
            'Top Scope Durations',
            (v) => formatMsDuration(v)
          )}
          style={{ height: 300 }}
        />
      </div>

      {/* Top kernel memory events */}
      <div style={{ border: '1px solid #eee', borderRadius: 6, padding: 8 }}>
        <ReactECharts
          notMerge
          option={barOptionH(
            topKernelMem.map((r) => `${r.name} [CUDA]`),
            topKernelMem.map((r) => r.used),
            'Top CUDA Kernel Memory Events (MiB)'
          )}
          style={{ height: 300 }}
        />
      </div>

      {/* Top kernel durations */}
      <div style={{ border: '1px solid #eee', borderRadius: 6, padding: 8 }}>
        <ReactECharts
          notMerge
          option={barOptionH(
            topKernelDuration.map((r) => `${r.name} [CUDA]`),
            topKernelDuration.map((r) => r.durMs),
            'Top CUDA Kernel Durations',
            (v) => formatMsDuration(v)
          )}
          style={{ height: 300 }}
        />
      </div>
    </div>
  );
}
