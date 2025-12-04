import React, { useMemo } from 'react';
import { AggregatedOperationRow, MetricEvent } from '../types/metrics';
import { buildAggregatedRows } from '../utils/DataTransformer';
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';

export interface MetricsTableProps {
  events: MetricEvent[];
  timeRangeMs?: number;
  nowMs?: number;
}

function formatNs(ns: number): string {
  if (!ns) return '0 ns';
  if (ns < 1_000) return `${ns} ns`;
  if (ns < 1_000_000) return `${(ns / 1_000).toFixed(2)} µs`;
  if (ns < 1_000_000_000) return `${(ns / 1_000_000).toFixed(2)} ms`;
  return `${(ns / 1_000_000_000).toFixed(2)} s`;
}

export default function MetricsTable({ events, timeRangeMs = 5 * 60_000, nowMs }: MetricsTableProps) {
  const end = nowMs ?? Date.now();
  const start = end - timeRangeMs;
  const rows = useMemo<AggregatedOperationRow[]>(() => buildAggregatedRows(events, start, end), [events, start, end]);

  // Group rows by app -> tag
  const byApp = new Map<string, Map<string, AggregatedOperationRow[]>>();
  for (const r of rows) {
    let tagMap = byApp.get(r.appName);
    if (!tagMap) {
      tagMap = new Map();
      byApp.set(r.appName, tagMap);
    }
    const lst = tagMap.get(r.tag) ?? [];
    lst.push(r);
    tagMap.set(r.tag, lst);
  }

  return (
    <Box>
      {Array.from(byApp.entries()).map(([app, tagMap]) => (
        <Accordion key={app} defaultExpanded>
          <AccordionSummary>
            <Typography variant="h6">{app}</Typography>
          </AccordionSummary>
          <AccordionDetails>
            {Array.from(tagMap.entries()).map(([tag, list]) => (
              <Accordion key={`${app}-${tag}`} defaultExpanded sx={{ boxShadow: 'none' }}>
                <AccordionSummary>
                  <Typography variant="subtitle1">Tag: {tag}</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Operation</TableCell>
                        <TableCell align="right">Count</TableCell>
                        <TableCell align="right">Total Duration</TableCell>
                        <TableCell align="right">Avg Duration</TableCell>
                        <TableCell align="right">Peak Memory (MiB)</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {list.map((r) => (
                        <TableRow key={`${app}-${tag}-${r.operation}`} hover>
                          <TableCell>{r.operation}</TableCell>
                          <TableCell align="right">{r.count}</TableCell>
                          <TableCell align="right">{formatNs(r.totalDurationNs)}</TableCell>
                          <TableCell align="right">{formatNs(r.avgDurationNs)}</TableCell>
                          <TableCell align="right">{r.peakMemoryMiB ?? '—'}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </AccordionDetails>
              </Accordion>
            ))}
          </AccordionDetails>
        </Accordion>
      ))}
    </Box>
  );
}
