import React, { useMemo, useRef, useState } from 'react';

// Lightweight SVG line chart to avoid external dependencies
// Props:
// - datasets: [{ label: string, points: {x:number,y:number}[], color?: string }]
// - label: y-axis label
export default function MetricChart({ datasets = [], label = 'Value', timeStartMs, timeEndMs }) {
  const width = 900;
  const height = 360;
  const margin = { top: 20, right: 20, bottom: 30, left: 50 };

  const processed = useMemo(() => {
    const ds = (Array.isArray(datasets) && datasets.length > 0) ? datasets : [];
    // Flatten all points to compute domains
    const allPoints = ds.flatMap(d => d.points || []);
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const p of allPoints) {
      if (Number.isFinite(p.x)) { xMin = Math.min(xMin, p.x); xMax = Math.max(xMax, p.x); }
      if (Number.isFinite(p.y)) { yMin = Math.min(yMin, p.y); yMax = Math.max(yMax, p.y); }
    }
    if (!Number.isFinite(xMin) || !Number.isFinite(xMax)) { xMin = 0; xMax = 1; }
    if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) { yMin = 0; yMax = 1; }
    if (yMin === yMax) { yMin = 0; }
    // If explicit window provided, override x domain to match top menu selection
    if (typeof timeStartMs === 'number' && typeof timeEndMs === 'number' && timeEndMs > timeStartMs) {
      xMin = timeStartMs;
      xMax = timeEndMs;
    }
    return { ds, xMin, xMax, yMin: Math.min(0, yMin), yMax };
  }, [datasets, timeStartMs, timeEndMs]);

  const xScale = (x) => {
    const { xMin, xMax } = processed;
    if (xMax === xMin) return margin.left;
    return margin.left + ((x - xMin) / (xMax - xMin)) * (width - margin.left - margin.right);
  };
  const yScale = (y) => {
    const { yMin, yMax } = processed;
    if (yMax === yMin) return height - margin.bottom;
    // invert y for SVG (0 at top)
    return margin.top + (1 - (y - yMin) / (yMax - yMin)) * (height - margin.top - margin.bottom);
  };

  const formatTime = (t) => {
    const d = new Date(t);
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    return `${hh}:${mm}`;
  };

  // Build axes ticks
  const xTicks = useMemo(() => {
    const { xMin, xMax } = processed;
    const n = 6;
    const arr = [];
    for (let i = 0; i < n; i++) {
      const t = xMin + (i / (n - 1)) * (xMax - xMin);
      arr.push({ x: t, label: formatTime(t) });
    }
    return arr;
  }, [processed]);
  const yTicks = useMemo(() => {
    const { yMin, yMax } = processed;
    const n = 5;
    const arr = [];
    for (let i = 0; i < n; i++) {
      const v = yMin + (i / (n - 1)) * (yMax - yMin);
      arr.push({ y: v, label: Math.round(v).toString() });
    }
    return arr;
  }, [processed]);

  const palette = ['#3b82f6','#ea580c','#10b981','#8b5cf6','#f43f5e','#f59e0b'];

  return (
    <div style={{ width: '100%', overflowX: 'auto' }}>
      <svg width={width} height={height} style={{ display: 'block', maxWidth: '100%' }}>
        {/* Axes */}
        <line x1={margin.left} y1={height - margin.bottom} x2={width - margin.right} y2={height - margin.bottom} stroke="#ccc" />
        <line x1={margin.left} y1={margin.top} x2={margin.left} y2={height - margin.bottom} stroke="#ccc" />
        {/* X ticks */}
        {xTicks.map((t, i) => (
          <g key={`xt-${i}`}>
            <line x1={xScale(t.x)} y1={height - margin.bottom} x2={xScale(t.x)} y2={height - margin.bottom + 4} stroke="#999" />
            <text x={xScale(t.x)} y={height - margin.bottom + 16} fontSize={11} textAnchor="middle" fill="#555">{t.label}</text>
          </g>
        ))}
        {/* Y ticks */}
        {yTicks.map((t, i) => (
          <g key={`yt-${i}`}>
            <line x1={margin.left - 4} y1={yScale(t.y)} x2={margin.left} y2={yScale(t.y)} stroke="#999" />
            <text x={margin.left - 8} y={yScale(t.y) + 3} fontSize={11} textAnchor="end" fill="#555">{t.label}</text>
            <line x1={margin.left} y1={yScale(t.y)} x2={width - margin.right} y2={yScale(t.y)} stroke="#f0f0f0" />
          </g>
        ))}
        {/* Y label */}
        <text x={-(height/2)} y={16} transform={`rotate(-90)`} fontSize={12} textAnchor="middle" fill="#333">{label}</text>

        {/* Lines */}
        {processed.ds.map((ds, idx) => {
          const color = ds.color || palette[idx % palette.length];
          const sorted = (ds.points || []).filter(p => Number.isFinite(p.x) && Number.isFinite(p.y)).sort((a,b)=>a.x-b.x);
          if (sorted.length === 0) return null;
          const path = sorted.map((p, i) => `${i===0?'M':'L'} ${xScale(p.x)} ${yScale(p.y)}`).join(' ');
          return (
            <g key={`line-${idx}`}>
              <path d={path} fill="none" stroke={color} strokeWidth={2} />
            </g>
          );
        })}

        {/* Legend */}
        {processed.ds.map((ds, idx) => {
          const color = ds.color || palette[idx % palette.length];
          const y = margin.top + idx * 18;
          return (
            <g key={`leg-${idx}`}>
              <rect x={margin.left + 8} y={y - 10} width={12} height={2} fill={color} />
              <text x={margin.left + 24} y={y - 8} fontSize={12} fill="#333">{ds.label}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
