import React, { useMemo, useState } from 'react';
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

type ScopeGroup = {
  id: string;
  app: string;
  pid: number;
  tag: string;
  name: string;
  begin: MetricEvent;
  end?: MetricEvent;
  startMs: number;
  endMs?: number;
  samples: MetricEvent[];
};

function eventTimeMs(e: MetricEvent): number {
  return new Date(e.timestamp).getTime();
}

function deriveUsedMiB(e: any): number | null {
  if (typeof e.usedMemoryMiB === 'number') return e.usedMemoryMiB;
  if (Array.isArray(e.devices)) {
    const v = e.devices.reduce((acc: number, d: any) => acc + (typeof d.used_mib === 'number' ? d.used_mib : 0), 0);
    return Number.isFinite(v) && v > 0 ? v : null;
  }
  return null;
}

export default function PerformanceTable({ events, timeRangeMs, nowMs, appFilter, tagFilter, maxExtraCols = 16 }: PerformanceTableProps) {
  const timeEnd = nowMs ?? Date.now();
  const timeStart = timeEnd - timeRangeMs;
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [detail, setDetail] = useState<{ title: string; ev: any } | null>(null);

  const windowed = useMemo(() => {
    return events.filter((e) => {
      const t = new Date(e.timestamp).getTime();
      return Number.isFinite(t) && t >= timeStart && t <= timeEnd;
    });
  }, [events, timeStart, timeEnd]);

  const filtered = useMemo(() => {
    return windowed.filter((e) => {
      // Show SystemMonitor/system_sample as well per new request
      if (appFilter && e.appName !== appFilter) return false;
      if (tagFilter && (e.tag || 'default') !== tagFilter) return false;
      return true;
    });
  }, [windowed, appFilter, tagFilter]);

  // Build scope groups and standalone rows (kernels)
  const { scopeGroups, kernelRows, samples } = useMemo(() => {
    const asc = [...filtered].sort((a, b) => eventTimeMs(a) - eventTimeMs(b));
    const openMap = new Map<string, ScopeGroup[]>(); // key -> stack of open scopes
    const groups: ScopeGroup[] = [];
    const kernels: MetricEvent[] = [];
    const sampleList: MetricEvent[] = [];
    for (const e of asc) {
      const key = `${e.appName}|${e.pid}|${e.tag || 'default'}|${e.name || ''}`;
      if (e.type === 'scope_begin') {
        const g: ScopeGroup = {
          id: `${key}#${eventTimeMs(e)}`,
          app: e.appName,
          pid: e.pid,
          tag: e.tag || 'default',
          name: e.name || 'scope',
          begin: e,
          startMs: eventTimeMs(e),
          samples: [],
        };
        if (!openMap.has(key)) openMap.set(key, []);
        openMap.get(key)!.push(g);
        groups.push(g);
      } else if (e.type === 'scope_end') {
        const stack = openMap.get(key);
        if (stack && stack.length > 0) {
          const g = stack.shift()!; // match earliest open
          g.end = e;
          g.endMs = eventTimeMs(e);
        } else {
          // unmatched end; ignore or treat as standalone
        }
      } else if (e.type === 'scope_sample') {
        const stack = openMap.get(key);
        if (stack && stack.length > 0) {
          // assign to the most recent open group (last)
          const current = stack[stack.length - 1];
          current.samples.push(e);
        } else {
          // no open group; ignore or later attach if group appears
        }
      } else if (e.type === 'kernel') {
        kernels.push(e);
      } else if (e.type === 'system_sample' || e.type === 'process_sample') {
        sampleList.push(e);
      }
    }
    return { scopeGroups: groups, kernelRows: kernels, samples: sampleList };
  }, [filtered]);

  // Derive extra keys from top-level of extra (kernels only for now)
  const extraKeys = useMemo(() => {
    const set = new Set<string>();
    for (const e of kernelRows) {
      const extra = (e as any).extra || {};
      if (extra && typeof extra === 'object') {
        for (const k of Object.keys(extra)) set.add(k);
      }
    }
    return Array.from(set).slice(0, maxExtraCols).sort();
  }, [kernelRows, maxExtraCols]);

  if (scopeGroups.length === 0 && kernelRows.length === 0 && samples.length === 0) {
    return <div style={{ fontSize: 12, color: '#666' }}>No events found for the selected filters and time range.</div>;
  }

  return (
    <div style={{ width: '100%', overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr>
            <th style={th}></th>
            <th style={th}>Start Time</th>
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
          {/* Scope groups */}
          {scopeGroups.map((g) => {
            const gpuList = Array.isArray(g.begin.devices) && g.begin.devices.length > 0
              ? g.begin.devices.map((d: any) => `${d.name || 'GPU'} (${d.uuid || 'unknown'})`).join('; ')
              : `${g.begin.gpuName || 'GPU'} (${g.begin.uuid || 'unknown'})`;
            const durMs = (g.endMs != null) ? (g.endMs - g.startMs) : 0;
            const used = deriveUsedMiB(g.begin);
            const isOpen = !!expanded[g.id];
            return (
              <React.Fragment key={g.id}>
                <tr style={{ borderTop: '1px solid #f2f2f2' }}>
                  <td style={td}>
                    <button onClick={() => setExpanded((p) => ({ ...p, [g.id]: !isOpen }))} style={{ cursor: 'pointer' }}>{isOpen ? '▾' : '▸'}</button>
                  </td>
                  <td style={td}>{formatTimeMs(g.startMs)}</td>
                  <td style={td}>scope</td>
                  <td style={td}><span onClick={() => setDetail({ title: 'Scope', ev: g.begin })} style={{ cursor: 'pointer', textDecoration: 'underline' }}>{g.name}</span></td>
                  <td style={tdNum}>{g.pid}</td>
                  <td style={td}>{g.tag}</td>
                  <td style={tdNum}>{g.endMs ? durMs.toFixed(3) : ''}</td>
                  <td style={tdNum}>{used ?? ''}</td>
                  <td style={td}>{gpuList}</td>
                  {extraKeys.map((k) => (
                    <td key={`sg-${g.id}-${k}`} style={td}></td>
                  ))}
                </tr>
                {isOpen && (
                  <tr>
                    <td></td>
                    <td colSpan={8 + extraKeys.length} style={{ padding: '6px 6px 12px' }}>
                      {!g.end && (
                        <div style={{ color: '#9a3412', background: '#fff7ed', border: '1px solid #fdba74', padding: 8, borderRadius: 6, marginBottom: 8 }}>
                          This scope has not ended yet (incomplete).
                        </div>
                      )}
                      {g.samples.length === 0 ? (
                        <div style={{ fontSize: 12, color: '#666' }}>No scope samples. Showing scope begin{g.end ? ' and end' : ''} only.</div>
                      ) : (
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                          <thead>
                            <tr>
                              <th style={th}>Time</th>
                              <th style={th}>Type</th>
                              <th style={th}>Used MiB</th>
                              <th style={th}>GPU</th>
                            </tr>
                          </thead>
                          <tbody>
                            {g.samples.map((s, i) => {
                              const usedS = deriveUsedMiB(s);
                              const gpuS = Array.isArray(s.devices) && s.devices.length > 0
                                ? s.devices.map((d: any) => `${d.name || 'GPU'} (${d.uuid || 'unknown'})`).join('; ')
                                : `${s.gpuName || 'GPU'} (${s.uuid || 'unknown'})`;
                              return (
                                <tr key={`${g.id}-s-${i}`}>
                                  <td style={td}>{formatTimeMs(eventTimeMs(s))}</td>
                                  <td style={td}><span onClick={() => setDetail({ title: 'Scope Sample', ev: s })} style={{ cursor: 'pointer', textDecoration: 'underline' }}>scope_sample</span></td>
                                  <td style={tdNum}>{usedS ?? ''}</td>
                                  <td style={td}>{gpuS}</td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      )}
                    </td>
                  </tr>
                )}
              </React.Fragment>
            );
          })}
          {/* Kernel rows */}
          {kernelRows.map((e, idx) => {
            const t = eventTimeMs(e);
            const extra = (e as any).extra || {};
            const used = deriveUsedMiB(e);
            const gpuCol = Array.isArray((e as any).devices) && (e as any).devices.length > 0
              ? (e as any).devices.map((d: any) => `${d.name || 'GPU'} (${d.uuid || 'unknown'})`).join('; ')
              : `${e.gpuName || 'GPU'} (${e.uuid || 'unknown'})`;
            return (
              <tr key={`k-${idx}`} style={{ borderTop: '1px solid #f2f2f2' }}>
                <td style={td}></td>
                <td style={td}>{formatTimeMs(t)}</td>
                <td style={td}>Kernel (CUDA)</td>
                <td style={td}><span onClick={() => setDetail({ title: 'Kernel', ev: e })} style={{ cursor: 'pointer', textDecoration: 'underline' }}>{e.name || extra.kernel || extra.name || ''}</span></td>
                <td style={tdNum}>{e.pid}</td>
                <td style={td}>{e.tag || 'default'}</td>
                <td style={tdNum}>{nsToMs(e.durationNs)?.toFixed(3)}</td>
                <td style={tdNum}>{used ?? ''}</td>
                <td style={td}>{gpuCol}</td>
                {extraKeys.map((k) => (
                  <td key={`kc-${idx}-${k}`} style={td}>{compactVal(extra?.[k])}</td>
                ))}
              </tr>
            );
          })}
          {/* Sample rows (includes SystemMonitor system_sample) */}
          {samples.map((s, idx) => {
            const t = eventTimeMs(s);
            const used = deriveUsedMiB(s);
            const gpuCol = Array.isArray((s as any).devices) && (s as any).devices.length > 0
              ? (s as any).devices.map((d: any) => `${d.name || 'GPU'} (${d.uuid || 'unknown'})`).join('; ')
              : `${(s as any).gpuName || 'GPU'} (${(s as any).uuid || 'unknown'})`;
            return (
              <tr key={`s-${idx}`} style={{ borderTop: '1px solid #f2f2f2' }}>
                <td style={td}></td>
                <td style={td}>{formatTimeMs(t)}</td>
                <td style={td}>{s.type}</td>
                <td style={td}><span onClick={() => setDetail({ title: s.type === 'system_sample' ? 'System Sample' : 'Process Sample', ev: s })} style={{ cursor: 'pointer', textDecoration: 'underline' }}>{s.name || '-'}</span></td>
                <td style={tdNum}>{s.pid}</td>
                <td style={td}>{s.tag || 'default'}</td>
                <td style={tdNum}></td>
                <td style={tdNum}>{used ?? ''}</td>
                <td style={td}>{gpuCol}</td>
                {extraKeys.map((k) => (
                  <td key={`sc-${idx}-${k}`} style={td}></td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>

      {/* Detail modal for device info */}
      {detail && (
        <div style={modalBackdrop} onClick={() => setDetail(null)}>
          <div style={modal} onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <div style={{ fontWeight: 600 }}>{detail.title} Details</div>
              <button onClick={() => setDetail(null)}>Close</button>
            </div>
            <div style={{ maxHeight: 420, overflowY: 'auto' }}>
              <div style={{ marginBottom: 8 }}>Name: <b>{detail.ev?.name || detail.ev?.extra?.kernel || detail.ev?.extra?.name || '-'}</b></div>
              <div style={{ marginBottom: 8 }}>App: {detail.ev?.appName} (PID {detail.ev?.pid}) | Tag: {detail.ev?.tag || 'default'}</div>
              <div style={{ marginBottom: 8 }}>Time: {formatTimeMs(eventTimeMs(detail.ev))}</div>
              <div style={{ margin: '8px 0', fontWeight: 600 }}>Devices</div>
              {Array.isArray(detail.ev?.devices) && detail.ev.devices.length > 0 ? (
                detail.ev.devices.map((d: any, i: number) => (
                  <div key={i} style={{ marginBottom: 10, paddingBottom: 8, borderBottom: '1px solid #eee' }}>
                    <div style={{ marginBottom: 4 }}>{d.name || 'GPU'} ({d.uuid || 'unknown'})</div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(140px, 1fr))', gap: 8, fontSize: 12 }}>
                      <div>used_mib: {d.used_mib ?? '-'}</div>
                      <div>free_mib: {d.free_mib ?? '-'}</div>
                      <div>total_mib: {d.total_mib ?? '-'}</div>
                      <div>util_gpu: {d.util_gpu ?? '-'}</div>
                      <div>util_mem: {d.util_mem ?? '-'}</div>
                      <div>temp_c: {d.temp_c ?? '-'}</div>
                      <div>power_mw: {d.power_mw ?? '-'}</div>
                      <div>clk_gfx: {d.clk_gfx ?? '-'}</div>
                      <div>clk_sm: {d.clk_sm ?? '-'}</div>
                      <div>clk_mem: {d.clk_mem ?? '-'}</div>
                    </div>
                  </div>
                ))
              ) : (
                <div style={{ fontSize: 12, color: '#666' }}>No device information available.</div>
              )}
              {detail.ev?.extra && (
                <>
                  <div style={{ margin: '8px 0', fontWeight: 600 }}>Extra</div>
                  <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 12 }}>{compactVal(detail.ev.extra)}</pre>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const th: React.CSSProperties = { textAlign: 'left', padding: '8px 6px', borderBottom: '1px solid #e5e7eb', whiteSpace: 'nowrap' };
const td: React.CSSProperties = { padding: '6px', verticalAlign: 'top', whiteSpace: 'nowrap' };
const tdNum: React.CSSProperties = { ...td, textAlign: 'right' } as React.CSSProperties;

const modalBackdrop: React.CSSProperties = {
  position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
  background: 'rgba(0,0,0,0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
};
const modal: React.CSSProperties = {
  background: '#fff', padding: 16, borderRadius: 8, width: 720, maxWidth: '95%', boxShadow: '0 10px 30px rgba(0,0,0,0.2)'
};
