# GPU Monitoring Frontend

A minimal React + Vite frontend to visualize GPU metrics from the Spring Boot backend.

## Prerequisites
- Node.js 18+ and npm
- The backend running locally (Spring Boot, see project root README). By default it listens on http://localhost:8080

## Development
Install dependencies and start the dev server:

```
cd frontend
npm install
npm run dev
```

Open the printed URL (default http://localhost:5173). The dev server proxies API requests from `/api` to `http://localhost:8080`.

## Usage
- Select GPU and metric. All available fields for the chosen metric are displayed simultaneously.
- Data comes from `GET /metrics`.
- Adjust the result limit and auto-refresh interval.
- The GPU list is derived from recent `/metrics` items (no separate `/gpus` endpoint in the Spring backend).

## Build
```
npm run build
npm run preview
```

## Notes
- Ensure the backend has data (run the ingestor or use tests) so the chart can display points.
- If your backend runs on a different host/port, update `vite.config.ts` proxy target accordingly.
