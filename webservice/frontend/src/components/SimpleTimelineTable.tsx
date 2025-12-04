import React, { useMemo, useState, useCallback } from 'react';
import { MetricEvent, TimelineData, TimelineBarItem } from '../types/metrics';
import { buildTimelineData } from '../utils/DataTransformer';
import { Drawer, Box, Typography, Divider, Button } from '@mui/material';

export interface SimpleTimelineTableProps {
  events: MetricEvent[];
  timeRangeMs?: number;
  nowMs?: number;
  onBarClick?: (bar: TimelineBarItem) => void;
}

function formatTimeMs(ms: number): string {
  const d = new Date(ms);
  const h = String(d.getHours()).padStart(2, '0');
  const m = String(d.getMinutes()).padStart(2, '0');
  const s = String(d.getSeconds()).padStart(2, '0');
  const S = String(d.getMilliseconds()).padStart(3, '0');
  return `${h}:${m}:${s}.${S}`;
}

function formatDateTimeMs(ms: number): string {
  const d = new Date(ms);
  const y = d.getFullYear();
  const M = String(d.getMonth() + 1).padStart(2, '0');
  const D = String(d.getDate()).padStart(2, '0');
  return `${y}-${M}-${D} ${formatTimeMs(ms)}`;
}

export default function SimpleTimelineTable({ events, timeRangeMs = 60 * 60 * 1000, nowMs, onBarClick }: SimpleTimelineTableProps) {
  const timeEnd = nowMs ?? Date.now();
  const timeStart = timeEnd - timeRangeMs;

  const data: TimelineData = useMemo(() => buildTimelineData(events, timeStart, timeEnd), [events, timeStart, timeEnd]);

  // Right-side drawer state for selected track (Azure-like slide-in)
  const [openTrackKey, setOpenTrackKey] = useState<string | null>(null);
  const selectedTrack = useMemo(() => data.tracks.find(t => t.key === openTrackKey) || null, [openTrackKey, data.tracks]);
  const openDrawer = useCallback((key: string) => setOpenTrackKey(key), []);
  const closeDrawer = useCallback(() => setOpenTrackKey(null), []);

  // Collect all bars within the window to compute a compact display window
  const allBars = useMemo(() => {
    const arr: TimelineBarItem[] = [];
    for (const t of data.tracks) {
      for (const b of t.bars) {
        // include bars that overlap [timeStart, timeEnd]
        if (b.endMs >= timeStart && b.startMs <= timeEnd) arr.push(b);
      }
    }
    return arr;
  }, [data.tracks, timeStart, timeEnd]);

  const { displayStart, displayEnd } = useMemo(() => {
    if (allBars.length === 0) return { displayStart: timeStart, displayEnd: timeEnd };
    let minS = Number.POSITIVE_INFINITY;
    let maxE = Number.NEGATIVE_INFINITY;
    for (const b of allBars) {
      if (b.startMs < minS) minS = b.startMs;
      if (b.endMs > maxE) maxE = b.endMs;
    }
    // clamp to requested window bounds
    minS = Math.max(minS, timeStart);
    maxE = Math.min(maxE, timeEnd);
    if (!(isFinite(minS) && isFinite(maxE)) || minS >= maxE) return { displayStart: timeStart, displayEnd: timeEnd };
    return { displayStart: minS, displayEnd: maxE };
  }, [allBars, timeStart, timeEnd]);

  const rangeMs = Math.max(1, displayEnd - displayStart);

  const rowHeight = 28;
  const barHeight = 12;

  // Determine if all tracks belong to the same application
  const uniqueApps = useMemo(() => Array.from(new Set(data.tracks.map(t => t.appName))), [data.tracks]);

  if (allBars.length === 0) {
    return (
      <div>
        <div style={{ fontSize: 12, color: '#666', marginBottom: 8 }}>
          No scopes found in the selected time range.
        </div>
      </div>
    );
  }

  return (
    <div style={{ width: '100%', overflowX: 'auto' }}>
      {/* Single app header to avoid repeating program name per track */}
      {uniqueApps.length === 1 && (
        <div style={{ marginBottom: 6 }}>
          <Typography variant="subtitle1" component="div" sx={{ fontWeight: 600 }}>
            Application: {uniqueApps[0]}
          </Typography>
        </div>
      )}
      {/* Legend */}
      <div style={{ display: 'flex', gap: 16, alignItems: 'center', fontSize: 12, color: '#444', marginBottom: 6 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 16, height: 8, background: '#3b82f6', display: 'inline-block', borderRadius: 2 }} />
          <span>Scope</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 16, height: 8, background: '#d97706', display: 'inline-block', borderRadius: 2 }} />
          <span>Kernel <span style={{ marginLeft: 4, padding: '2px 6px', background: '#fff7ed', color: '#9a3412', border: '1px solid #fdba74', borderRadius: 8 }}>CUDA</span></span>
        </div>
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={{ textAlign: 'left', padding: '8px 6px', borderBottom: '1px solid #eee', width: 260 }}>Track</th>
            <th style={{ textAlign: 'left', padding: '8px 6px', borderBottom: '1px solid #eee' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 600 }}>
                <span>{formatDateTimeMs(displayStart)}</span>
                <span>{formatDateTimeMs(displayEnd)}</span>
              </div>
            </th>
          </tr>
        </thead>
        <tbody>
          {data.tracks.map((track) => {
            // Hide repeating app name and PID; show concise tag label only
            const label = track.tag || 'default';
            // Only show tracks that have at least one overlapping bar
            const tbars = track.bars.filter((b) => b.endMs >= timeStart && b.startMs <= timeEnd);
            if (tbars.length === 0) return null;
            return (
              <>
                <tr key={track.key}>
                  <td style={{ padding: '8px 6px', fontSize: 12, verticalAlign: 'top', borderBottom: '1px solid #fafafa', color: '#333' }}>
                    <span
                      role="button"
                      onClick={() => openDrawer(track.key)}
                      title={'Show details'}
                      style={{ cursor: 'pointer', textDecoration: 'underline', textUnderlineOffset: 2 }}
                    >
                      {label}
                    </span>
                  </td>
                  <td style={{ padding: '8px 6px', borderBottom: '1px solid #fafafa' }}>
                    <div style={{ position: 'relative', height: rowHeight }}>
                      {tbars.map((b) => {
                        const s = Math.max(b.startMs, displayStart);
                        const e = Math.min(b.endMs, displayEnd);
                        const leftPct = ((s - displayStart) / rangeMs) * 100;
                        const widthPct = Math.max(0.2, ((e - s) / rangeMs) * 100); // ensure visible
                        const cudaLabel = b.type === 'kernel' ? ' [CUDA]' : '';
                        const title = `${b.name}${cudaLabel}\n${formatDateTimeMs(b.startMs)} – ${formatDateTimeMs(b.endMs)}`;
                        const color = b.type === 'kernel' ? '#d97706' : '#3b82f6';
                        return (
                          <div
                            key={`${b.id}-${s}`}
                            title={title}
                            onClick={() => onBarClick && onBarClick(b)}
                            style={{
                              position: 'absolute',
                              left: `${leftPct}%`,
                              top: (rowHeight - barHeight) / 2,
                              width: `${widthPct}%`,
                              height: barHeight,
                              background: color,
                              borderRadius: 3,
                              opacity: 0.9,
                              minWidth: 2,
                              cursor: 'pointer',
                            }}
                          />
                        );
                      })}
                    </div>
                  </td>
                </tr>
              </>
            );
          })}
        </tbody>
      </table>

      {/* Right-side sliding Drawer for track details */}
      <Drawer anchor="right" open={!!openTrackKey} onClose={closeDrawer} PaperProps={{ sx: { width: 420 } }}>
        <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 1, height: '100%' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>Track Details</Typography>
            <Button size="small" onClick={closeDrawer}>Close</Button>
          </Box>
          <Divider />
          {selectedTrack ? (
            <Box sx={{ overflowY: 'auto', pr: 1 }}>
              {/* Header info */}
              <Box sx={{ mb: 1, color: '#444', fontSize: 13 }}>
                {/* Show app name once here if multiple apps exist */}
                <div><strong>Application:</strong> {selectedTrack.appName}</div>
                {/* Hide PID as requested; still show tag */}
                <div><strong>Tag:</strong> {selectedTrack.tag || 'default'}</div>
              </Box>
              <Divider sx={{ mb: 1 }} />
              <ul style={{ margin: 0, paddingLeft: 16, listStyle: 'disc', fontSize: 12, color: '#333' }}>
                {selectedTrack.bars.map((b) => (
                  <li key={`drawer-${selectedTrack.key}-${b.id}`} style={{ marginBottom: 6 }}>
                    <span style={{
                      display: 'inline-block',
                      padding: '2px 6px',
                      borderRadius: 8,
                      marginRight: 6,
                      background: b.type === 'kernel' ? '#fff7ed' : '#eff6ff',
                      color: b.type === 'kernel' ? '#9a3412' : '#1d4ed8',
                      border: `1px solid ${b.type === 'kernel' ? '#fdba74' : '#93c5fd'}`,
                    }}>
                      {b.type === 'kernel' ? 'CUDA' : 'Scope'}
                    </span>
                    <strong>{b.name}</strong>
                    <span
                      role="button"
                      onClick={() => onBarClick && onBarClick(b)}
                      title="View details"
                      style={{ color: '#0f62fe', marginLeft: 8, cursor: 'pointer', textDecoration: 'underline', textUnderlineOffset: 2 }}
                    >
                      {formatTimeMs(b.startMs)} – {formatTimeMs(b.endMs)}
                    </span>
                    <span style={{ color: '#999', marginLeft: 8 }}>
                      ({Math.max(1, Math.round(b.durationMs))} ms)
                    </span>
                  </li>
                ))}
              </ul>
            </Box>
          ) : (
            <Typography variant="body2" sx={{ color: '#666' }}>No track selected.</Typography>
          )}
        </Box>
      </Drawer>
    </div>
  );
}
