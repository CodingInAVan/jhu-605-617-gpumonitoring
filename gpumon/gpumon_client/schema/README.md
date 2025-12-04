GPUmon NDJSON Event Schema

Overview
- Format: NDJSON (newline-delimited JSON) — exactly one JSON object per line.
- Producer: gpumon_client (header-only library) at runtime in the target process.
- Collection: A separate crawler (gpumon/crawler) tails these log files, validates lines against the schema, and then forwards them to backend systems.
- Purpose: Define the stable event contract used by the crawler and backend for storage, querying, and visualization.

Global conventions
- Common fields present in most events:
  - type: string — event type discriminator. One of: init, scope_begin, scope_sample, scope_end, kernel, shutdown
  - pid: integer — process id of the emitting process
  - app: string — application name as passed in InitOptions::appName
- Timestamps:
  - ts_ns: integer — event timestamp in nanoseconds since a monotonic clock epoch
  - ts_start_ns / ts_end_ns: integer — start/end timestamps in nanoseconds
  - duration_ns: integer — derived as ts_end_ns - ts_start_ns (nanoseconds)
- Memory units:
  - Memory quantities are reported in mebibytes (MiB).
  - Snapshots are arrays of device-level records: { device, used_mib, free_mib, total_mib }

Event types
1) init
   - Emitted once after gpumon::init() succeeds.
   - Fields:
     - type: "init"
     - pid: integer
     - app: string
     - logPath: string — absolute or relative path of the current log file (may be empty if silent)
     - ts_ns: integer

   Example:
   {"type":"init","pid":1234,"app":"trainer","logPath":"gpumon.log","ts_ns":1731958400123456}

2) scope_begin
   - Emitted when a scope starts (GPUMON_SCOPE / ScopedMonitor construction).
   - Fields:
     - type: "scope_begin"
     - pid: integer
     - app: string
     - name: string — scope name
     - tag: string (optional) — user-provided tag
     - ts_ns: integer
     - memory: array<MemorySnapshot> (optional if backend cannot query)

   Example:
   {"type":"scope_begin","pid":1234,"app":"trainer","name":"epoch_1","ts_ns":1731958400123456,"memory":[{"device":0,"used_mib":1024,"free_mib":8192,"total_mib":9216}]}

3) scope_sample
   - Emitted periodically inside an open scope when InitOptions::sampleIntervalMs > 0.
   - Fields: same as scope_begin (includes ts_ns), plus optional tag and memory.

   Example:
   {"type":"scope_sample","pid":1234,"app":"trainer","name":"epoch_1","ts_ns":1731958401123456,"memory":[{"device":0,"used_mib":1100,"free_mib":8116,"total_mib":9216}]}

4) scope_end
   - Emitted when a scope finishes (ScopedMonitor destructor). A backend synchronize() is issued prior to logging to ensure the scope captures in-flight GPU work.
   - Fields:
     - type: "scope_end"
     - pid: integer
     - app: string
     - name: string
     - tag: string (optional)
     - ts_start_ns: integer
     - ts_end_ns: integer
     - duration_ns: integer
     - memory: array<MemorySnapshot> (optional)

   Example:
   {"type":"scope_end","pid":1234,"app":"trainer","name":"epoch_1","ts_start_ns":1731958400123456,"ts_end_ns":1731958403123456,"duration_ns":3000000,"memory":[{"device":0,"used_mib":1110,"free_mib":8106,"total_mib":9216}]}

5) kernel
   - Emitted by the CUDA backend macros GPUMON_LAUNCH/GPUMON_LAUNCH_TAGGED.
   - Fields:
     - type: "kernel"
     - pid: integer
     - app: string
     - kernel: string — kernel symbol/name as provided to the macro
     - ts_start_ns: integer
     - ts_end_ns: integer
     - duration_ns: integer
     - grid: [int, int, int] — grid dimensions (x, y, z)
     - block: [int, int, int] — block dimensions (x, y, z)
     - shared_mem_bytes: integer
     - cuda_error: string — CUDA error string from cudaGetLastError() after launch
     - tag: string (optional) — present when using GPUMON_LAUNCH_TAGGED

   Example:
   {"type":"kernel","pid":1234,"app":"trainer","kernel":"vectorAdd","ts_start_ns":1731958400123456,"ts_end_ns":1731958400126789,"duration_ns":3333,"grid":[128,1,1],"block":[256,1,1],"shared_mem_bytes":0,"cuda_error":"cudaSuccess"}

6) shutdown
   - Emitted once during gpumon::shutdown().
   - Fields:
     - type: "shutdown"
     - pid: integer
     - app: string
     - ts_ns: integer

   Example:
   {"type":"shutdown","pid":1234,"app":"trainer","ts_ns":1731958404123456}

MemorySnapshot shape
- device: integer — CUDA device index
- used_mib: integer — used memory in MiB (total_mib - free_mib)
- free_mib: integer — free memory in MiB
- total_mib: integer — total memory in MiB

JSON Schema
- A machine-readable JSON Schema (draft-07) is provided alongside this document at: gpumon_client/schema/ndjson.schema.json
- The crawler and backends can use this schema to validate and route events by type. The schema uses oneOf on the "type" field.

Notes and guarantees
- Field names and units match the current gpumon_client implementation in include/gpumon/core/monitor.hpp and include/gpumon/backends/cuda.hpp.
- Additional fields may be added in future versions in a backward-compatible manner; consumers (crawler/backend) should ignore unknown fields.
- The presence of the memory array depends on device/driver permissions; treat it as optional on scope_* events.

Collection pipeline (summary)
- Client: writes NDJSON files to InitOptions::logFilePath, or to GPUMON_LOG_DIR when logFilePath is empty.
- Crawler: discovers NDJSON files, validates each line against ndjson.schema.json, enriches if needed, and forwards to your backend.
