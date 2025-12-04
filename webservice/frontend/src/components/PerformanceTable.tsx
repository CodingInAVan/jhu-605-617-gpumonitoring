import React, { useMemo } from 'react';
import { MetricEvent } from '../types/metrics';

export interface PerformanceTableProps {
  events: MetricEvent[];
  timeRangeMs: number;
  nowMs?: number;
  appFilter?: string;
  tagFilter?: string;
  maxExtraCols?: number; // optional cap on number of extra columns
}

function formatTimeMs(ms: number): string {
  const d = new Date(ms);
  const y = d.getFullYear();
  const M = String(d.getMonth() + 1).padStart(2, '0');
  const D = String(d.getDate()).padStart(2, '0');
  const h = String(d.getHours()).padStart(2, '0');
  const m = String(d.getMinutes()).padStart(2, '0');
  const s = String(d.getSeconds()).padStart(2, '0');
  const S = String(d.getMilliseconds()).padStart(3, '0');
  return `${y}-${M}-${D} ${h}:${m}:${s}.${S}`;
}

function nsToMs(ns?: number) {
  return ns && ns > 0 ? ns / 1_000_000 : 0;
}

function compactVal(v: any): string {
  if (v == null) return '';
  if (typeof v === 'string') return v;
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);
  if (Array.isArray(v)) return v.join(', ');
  try { return JSON.stringify(v); } catch { return String(v); }
}

export default function PerformanceTable({ events, timeRangeMs, nowMs, appFilter, tagFilter, maxExtraCols = 16 }: PerformanceTableProps) {
  const timeEnd = nowMs ?? Date.now();
  const timeStart = timeEnd - timeRangeMs;

  const windowed = useMemo(() => {
    return events.filter((e) => {
      const t = new Date(e.timestamp).getTime();
      return Number.isFinite(t) && t >= timeStart && t <= timeEnd;
    });
  }, [events, timeStart, timeEnd]);

  const filtered = useMemo(() => {
    return windowed.filter((e) => {
      if (appFilter && e.appName !== appFilter) return false;
      if (tagFilter && (e.tag || 'default') !== tagFilter) return false;
      return true;
    });
  }, [windowed, appFilter, tagFilter]);

  // Derive extra keys from top-level of extra
  const extraKeys = useMemo(() => {
    const set = new Set<string>();
    for (const e of filtered) {
      const extra = (e as any).extra || {};
      if (extra && typeof extra === 'object') {
        for (const k of Object.keys(extra)) {
          // Skip if conflicts with base columns
          if (['kernel','name','ts_start_ns','ts_end_ns'].includes(k)) {
            // allow kernel/name since we still show Name from event
          }
          set.add(k);
        }
      }
    }
    return Array.from(set).slice(0, maxExtraCols).sort();
  }, [filtered, maxExtraCols]);

  const rows = useMemo(() => {
    const arr = filtered.map((e) => {
      const t = new Date(e.timestamp).getTime();
      const extra = (e as any).extra || {};
      return {
        timeMs: t,
        type: e.type,
        name: e.name || extra.kernel || extra.name || '',
        app: e.appName,
        pid: e.pid,
        tag: e.tag || 'default',
        durationMs: nsToMs(e.durationNs),
        usedMiB: typeof e.usedMemoryMiB === 'number' ? e.usedMemoryMiB : '',
        gpu: `${e.gpuName || 'GPU'} (${e.uuid || 'unknown'})`,
        extra,
      };
    });
    arr.sort((a, b) => a.timeMs - b.timeMs);
    return arr;
  }, [filtered]);

  if (rows.length === 0) {
    return <div style={{ fontSize: 12, color: '#666' }}>No events found for the selected filters and time range.</div>;
  }

  return (
    <div style={{ width: '100%', overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr>
            <th style={th}>Time</th>
            <th style={th}>Type</th>
            <th style={th}>Name</th>
            <th style={th}>PID</th>
            <th style={th}>Tag</th>
            <th style={th}>Duration (ms)</th>
            <th style={th}>Used MiB</th>
            <th style={th}>GPU</th>
            {extraKeys.map((k) => (
              <th key={`h-${k}`} style={th}>{k}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, idx) => (
            <tr key={idx} style={{ borderTop: '1px solid #f2f2f2' }}>
              <td style={td}>{formatTimeMs(r.timeMs)}</td>
              <td style={td}>{r.type === 'kernel' ? 'Kernel (CUDA)' : r.type}</td>
              <td style={td}>{r.name || '—'}</td>
              <td style={tdNum}>{r.pid}</td>
              <td style={td}>{r.tag}</td>
              <td style={tdNum}>{r.durationMs ? r.durationMs.toFixed(3) : ''}</td>
              <td style={tdNum}>{r.usedMiB}</td>
              <td style={td}>{r.gpu}</td>
              {extraKeys.map((k) => (
                <td key={`c-${idx}-${k}`} style={td}>{compactVal(r.extra?.[k])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const th: React.CSSProperties = { textAlign: 'left', padding: '8px 6px', borderBottom: '1px solid #e5e7eb', whiteSpace: 'nowrap' };
const td: React.CSSProperties = { padding: '6px', verticalAlign: 'top', whiteSpace: 'nowrap' };
const tdNum: React.CSSProperties = { ...td, textAlign: 'right' } as React.CSSProperties;
