import { AggregatedOperationRow, MemorySamplePoint, MetricEvent, TimelineBarItem, TimelineData, TimelineTrack } from '../types/metrics';

function safeTag(tag?: string): string {
  return tag && tag.length > 0 ? tag : 'default';
}

function nsToMs(ns?: number): number {
  if (!ns || ns <= 0) return 0;
  return ns / 1_000_000;
}

export function buildTimelineData(
  events: MetricEvent[],
  timeStartMs?: number,
  timeEndMs?: number
): TimelineData {
  const tracksMap = new Map<string, TimelineTrack>();
  let minT = timeStartMs ?? Number.POSITIVE_INFINITY;
  let maxT = timeEndMs ?? Number.NEGATIVE_INFINITY;

  for (const ev of events) {
    const tEnd = new Date(ev.timestamp).getTime();
    if (!Number.isFinite(tEnd)) continue;

    if (ev.type === 'process_sample') {
      const key = `${ev.appName}|${ev.pid}|${safeTag(ev.tag)}`;
      let track = tracksMap.get(key);
      if (!track) {
        track = {
          appName: ev.appName,
          pid: ev.pid,
          tag: safeTag(ev.tag),
          key,
          bars: [],
          samples: [],
        };
        tracksMap.set(key, track);
      }
      const used = typeof ev.usedMemoryMiB === 'number' ? ev.usedMemoryMiB : null;
      if (used !== null) {
        track.samples.push({ t: tEnd, v: used });
      }
      if (!timeStartMs) minT = Math.min(minT, tEnd);
      if (!timeEndMs) maxT = Math.max(maxT, tEnd);
    } else if (ev.type === 'scope_end' || ev.type === 'kernel') {
      // For both scope_end and kernel, prefer explicit tsStart/tsEnd if provided
      const endFromNs = ev.tsEndNs ? (ev.tsEndNs / 1_000_000) : tEnd;
      const durMs = nsToMs(ev.durationNs);
      const startFromNs = ev.tsStartNs ? (ev.tsStartNs / 1_000_000) : (endFromNs - durMs);
      const start = Math.min(startFromNs, endFromNs);
      const end = Math.max(startFromNs, endFromNs);
      const key = `${ev.appName}|${ev.pid}|${safeTag(ev.tag)}`;
      let track = tracksMap.get(key);
      if (!track) {
        track = {
          appName: ev.appName,
          pid: ev.pid,
          tag: safeTag(ev.tag),
          key,
          bars: [],
          samples: [],
        };
        tracksMap.set(key, track);
      }
      const bar: TimelineBarItem = {
        id: `${ev.type}:${ev.name ?? ''}:${start}`,
        type: ev.type === 'kernel' ? 'kernel' : 'scope',
        name: ev.name || (ev.type === 'kernel' ? 'kernel' : 'scope'),
        startMs: start,
        endMs: end,
        durationMs: Math.max(0, end - start),
        source: ev,
      };
      track.bars.push(bar);
      if (!timeStartMs) minT = Math.min(minT, start);
      if (!timeEndMs) maxT = Math.max(maxT, end);
    }
  }

  const tracks = Array.from(tracksMap.values())
    .map((t) => ({
      ...t,
      // sort for stable rendering
      bars: t.bars.sort((a, b) => a.startMs - b.startMs),
      samples: t.samples.sort((a, b) => a.t - b.t),
    }))
    // sort tracks by appName, then pid, then tag
    .sort((a, b) => a.appName.localeCompare(b.appName) || a.pid - b.pid || a.tag.localeCompare(b.tag));

  if (!Number.isFinite(minT)) minT = Date.now() - 5 * 60_000;
  if (!Number.isFinite(maxT)) maxT = Date.now();

  if (timeStartMs != null) minT = timeStartMs;
  if (timeEndMs != null) maxT = timeEndMs;

  return { tracks, timeMinMs: minT, timeMaxMs: maxT };
}

export function buildAggregatedRows(
  events: MetricEvent[],
  windowStartMs?: number,
  windowEndMs?: number
): AggregatedOperationRow[] {
  // Pre-index samples by pid|tag for quick range queries
  const sampleMap = new Map<string, MemorySamplePoint[]>();

  const rowsKey = (app: string, tag: string, name: string) => `${app}|${tag}|${name}`;
  const agg = new Map<string, { appName: string; tag: string; operation: string; count: number; totalDurationNs: number; peaks: number[] }>();

  const startBound = windowStartMs ?? Number.NEGATIVE_INFINITY;
  const endBound = windowEndMs ?? Number.POSITIVE_INFINITY;

  for (const ev of events) {
    const tEnd = new Date(ev.timestamp).getTime();
    if (!Number.isFinite(tEnd)) continue;

    if (ev.type === 'process_sample') {
      if (tEnd < startBound || tEnd > endBound) continue;
      const key = `${ev.pid}|${safeTag(ev.tag)}`;
      const used = typeof ev.usedMemoryMiB === 'number' ? ev.usedMemoryMiB : null;
      if (used === null) continue;
      let arr = sampleMap.get(key);
      if (!arr) {
        arr = [];
        sampleMap.set(key, arr);
      }
      arr.push({ t: tEnd, v: used });
    }
  }
  // sort samples
  for (const arr of sampleMap.values()) arr.sort((a, b) => a.t - b.t);

  for (const ev of events) {
    if (ev.type !== 'scope_end' && ev.type !== 'kernel') continue;
    const tEnd = new Date(ev.timestamp).getTime();
    if (!Number.isFinite(tEnd)) continue;
    const durMs = nsToMs(ev.durationNs);
    const start = tEnd - durMs;

    // window intersection
    const s = Math.max(start, startBound);
    const e = Math.min(tEnd, endBound);
    if (s > e) continue;

    const app = ev.appName;
    const tag = safeTag(ev.tag);
    const op = ev.name || (ev.type === 'kernel' ? 'kernel' : 'scope');
    const key = rowsKey(app, tag, op);
    let rec = agg.get(key);
    if (!rec) {
      rec = { appName: app, tag, operation: op, count: 0, totalDurationNs: 0, peaks: [] };
      agg.set(key, rec);
    }
    rec.count += 1;
    // Duration in ns (keep original unit for totals/avg as per requirement)
    rec.totalDurationNs += ev.durationNs || 0;

    // compute peak memory within [start, end]
    const smKey = `${ev.pid}|${tag}`;
    const samples = sampleMap.get(smKey);
    if (samples && samples.length) {
      // binary search window indices
      const i0 = lowerBound(samples, s);
      const i1 = upperBound(samples, e);
      if (i0 < i1) {
        let peak = Number.NEGATIVE_INFINITY;
        for (let i = i0; i < i1; i++) {
          if (samples[i].v > peak) peak = samples[i].v;
        }
        if (Number.isFinite(peak)) rec.peaks.push(peak);
      }
    }
  }

  const rows: AggregatedOperationRow[] = [];
  agg.forEach((v) => {
    const peak = v.peaks.length ? Math.max(...v.peaks) : null;
    rows.push({
      appName: v.appName,
      tag: v.tag,
      operation: v.operation,
      count: v.count,
      totalDurationNs: v.totalDurationNs,
      avgDurationNs: v.count > 0 ? Math.round(v.totalDurationNs / v.count) : 0,
      peakMemoryMiB: peak,
    });
  });

  // Sort: appName, tag, operation
  rows.sort((a, b) => a.appName.localeCompare(b.appName) || a.tag.localeCompare(b.tag) || a.operation.localeCompare(b.operation));
  return rows;
}

function lowerBound(arr: MemorySamplePoint[], t: number): number {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (arr[mid].t < t) lo = mid + 1; else hi = mid;
  }
  return lo;
}

function upperBound(arr: MemorySamplePoint[], t: number): number {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (arr[mid].t <= t) lo = mid + 1; else hi = mid;
  }
  return lo;
}
