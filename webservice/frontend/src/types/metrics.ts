// Shared TypeScript types for gpumon frontend

export interface MetricEvent {
  timestamp: string; // ISO string (end time for scope_end)
  type: 'process_sample' | 'scope_begin' | 'scope_end' | 'kernel';
  pid: number;
  appName: string; // e.g. "heavy_cuda_demo"
  tag?: string; // e.g. "training_loop" or thread name
  name?: string; // Scope name or Kernel name

  // For Samples
  usedMemoryMiB?: number;
  tempC?: number;

  // For Scopes/Kernels
  durationNs?: number;
  tsStartNs?: number; // optional raw start (ns) when available
  tsEndNs?: number;   // optional raw end (ns) when available

  // Hardware Context (Denormalized)
  gpuName: string;
  uuid: string;
  // Optional additional metadata from backend may be present
  [k: string]: unknown;
}

export interface TimelineBarItem {
  id: string;
  type: 'scope' | 'kernel';
  name: string;
  startMs: number; // epoch millis
  endMs: number; // epoch millis
  durationMs: number;
  color?: string;
  source: MetricEvent;
}

export interface MemorySamplePoint {
  t: number; // epoch millis
  v: number; // MiB
}

export interface TimelineTrack {
  appName: string;
  pid: number;
  tag: string; // grouping label
  key: string; // unique key appName|pid|tag
  bars: TimelineBarItem[];
  samples: MemorySamplePoint[]; // per-pid/tag memory samples
}

export interface TimelineData {
  tracks: TimelineTrack[];
  timeMinMs: number;
  timeMaxMs: number;
}

export interface AggregatedOperationRow {
  appName: string;
  tag: string;
  operation: string; // scope/kernel name
  count: number;
  totalDurationNs: number;
  avgDurationNs: number;
  peakMemoryMiB: number | null;
}
