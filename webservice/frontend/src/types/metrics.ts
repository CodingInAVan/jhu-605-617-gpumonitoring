// Shared TypeScript types for gpumon frontend

export interface MetricDevice {
  id?: number | null;
  uuid?: string | null;
  name?: string | null; // gpuName
  pci_bus?: number | null;
  used_mib?: number | null;
  free_mib?: number | null;
  total_mib?: number | null;
  util_gpu?: number | null; // percent
  util_mem?: number | null; // percent
  temp_c?: number | null;   // Celsius
  power_mw?: number | null; // milliwatts
  clk_gfx?: number | null;  // MHz
  clk_sm?: number | null;   // MHz
  clk_mem?: number | null;  // MHz
}

export interface MetricEvent {
  timestamp: string; // ISO string (end time for scope_end)
  type: 'process_sample' | 'system_sample' | 'scope_sample' | 'scope_begin' | 'scope_end' | 'kernel';
  pid: number;
  appName: string; // e.g. "heavy_cuda_demo"
  tag?: string; // e.g. "training_loop" or thread name
  name?: string; // Scope name or Kernel name

  // For Samples
  usedMemoryMiB?: number; // derived from devices[].used_mib when available
  tempC?: number; // legacy aggregate; prefer per-device temp_c
  devices?: MetricDevice[]; // one or more devices assigned to this event

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
