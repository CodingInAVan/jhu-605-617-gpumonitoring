import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import * as echarts from 'echarts';
import { MetricEvent, TimelineBarItem, TimelineData } from '../types/metrics';
import { buildTimelineData } from '../utils/DataTransformer';

export interface TimelineProfilerProps {
  events: MetricEvent[];
  // If provided, show [now - timeRangeMs, now]
  timeRangeMs?: number;
  nowMs?: number;
  maxTracks?: number; // soft cap for rendering grids
  onBarClick?: (bar: TimelineBarItem) => void;
}

function formatDuration(ms: number): string {
  if (ms < 1) return `${Math.round(ms * 1000)}µs`;
  if (ms < 1000) return `${ms.toFixed(2)}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(2)}s`;
  const m = Math.floor(s / 60);
  const ss = Math.round(s % 60);
  return `${m}m ${ss}s`;
}

export default function TimelineProfiler({ events, timeRangeMs = 60_000, nowMs, maxTracks = 8, onBarClick }: TimelineProfilerProps) {
  const timeEnd = nowMs ?? Date.now();
  const timeStart = timeEnd - timeRangeMs;

  const data: TimelineData = useMemo(() => buildTimelineData(events, timeStart, timeEnd), [events, timeStart, timeEnd]);

  const tracks = data.tracks.slice(0, maxTracks);

  // Simple layout: single grid per track, only execution bars
  const gridHeight = 48;
  const gridGap = 8;
  const topPadding = 10;

  const grids: any[] = [];
  const xAxes: any[] = [];
  const yAxes: any[] = [];
  const series: any[] = [];

  tracks.forEach((track, idx) => {
    const topBar = topPadding + idx * (gridHeight + gridGap);
    const gridIndexBar = grids.length;
    grids.push({ top: topBar, height: gridHeight, left: 80, right: 20 });

    // X axes share same time domain
    const xCommon = { type: 'time', min: data.timeMinMs, max: data.timeMaxMs, axisLabel: { formatter: (val: number) => new Date(val).toLocaleTimeString() } };
    xAxes.push({ ...xCommon, gridIndex: gridIndexBar });

    // Y for bars: category with single row label displaying app/pid/tag
    const label = `${track.appName} • ${track.pid} • ${track.tag}`;
    yAxes.push({
      type: 'category',
      gridIndex: gridIndexBar,
      data: [label],
      axisLabel: { interval: 0 },
    });

    // Build bar series as custom
    const barData = track.bars.map((b) => ({
      name: b.name,
      value: [b.startMs, 0, b.endMs, 0], // start/end; y category index 0
      itemStyle: {
        color: '#3b82f6', // single color for simplicity
        opacity: 0.85,
      },
      barRef: b,
    }));

    series.push({
      name: 'Execution',
      type: 'custom',
      xAxisIndex: gridIndexBar,
      yAxisIndex: gridIndexBar,
      renderItem: (params: any, api: any) => {
        const start = api.value(0);
        const end = api.value(2);
        const y = api.coord([start, 0])[1];
        const xStart = api.coord([start, 0])[0];
        const xEnd = api.coord([end, 0])[0];
        const height = 16;
        const cy = y - height / 2 + 8;
        return {
          type: 'rect',
          shape: { x: xStart, y: cy, width: Math.max(1, xEnd - xStart), height },
          style: api.style(),
        };
      },
      data: barData,
      tooltip: {
        trigger: 'item',
        formatter: (p: any) => {
          const b: TimelineBarItem = p.data.barRef;
          const dur = formatDuration(b.durationMs);
          return `${echarts.format.encodeHTML(b.name)}<br/>${new Date(b.startMs).toLocaleString()} – ${new Date(b.endMs).toLocaleString()}<br/>Duration: ${dur}`;
        },
      },
    });
  });

  const option = {
    grid: grids,
    xAxis: xAxes,
    yAxis: yAxes,
    tooltip: { confine: true },
    dataZoom: [
      { type: 'inside', xAxisIndex: xAxes.map((_, i) => i) },
      { type: 'slider', xAxisIndex: [0], top: tracks.length * (gridHeight + gridGap) + 10 },
    ],
    series,
  } as echarts.EChartsOption;

  const onEvents = {
    click: (p: any) => {
      const d = p?.data;
      if (d?.barRef && onBarClick) onBarClick(d.barRef as TimelineBarItem);
    },
  } as any;

  const totalHeight = tracks.length * (gridHeight + gridGap) + 60;

  return (
    <div style={{ width: '100%', height: totalHeight }}>
      <ReactECharts option={option} onEvents={onEvents} notMerge lazyUpdate style={{ height: '100%', width: '100%' }} />
    </div>
  );
}
