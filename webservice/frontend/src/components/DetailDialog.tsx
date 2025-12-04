import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Table,
  TableBody,
  TableCell,
  TableRow,
  Chip,
} from '@mui/material';
import { TimelineBarItem } from '../types/metrics';

function prettyJson(value: any): string {
  try {
    if (typeof value === 'string') {
      // try parse then pretty print; otherwise show as is
      const parsed = JSON.parse(value);
      return JSON.stringify(parsed, null, 2);
    }
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
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

export interface DetailDialogProps {
  open: boolean;
  bar: TimelineBarItem | null;
  onClose: () => void;
}

export default function DetailDialog({ open, bar, onClose }: DetailDialogProps) {
  const ev: any = bar?.source || {};
  const extra = (ev && (ev.extra || {})) as any;
  const isKernel = bar?.type === 'kernel';

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        {isKernel ? 'CUDA Kernel' : 'Scope'} Details
        {isKernel && <Chip size="small" label="CUDA" sx={{ ml: 1 }} color="warning" variant="outlined" />}
      </DialogTitle>
      <DialogContent dividers>
        {bar && (
          <Table size="small" sx={{ '& td': { verticalAlign: 'top' } }}>
            <TableBody>
              <TableRow>
                <TableCell width={180}>Name</TableCell>
                <TableCell>{bar.name || '—'}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Type</TableCell>
                <TableCell>{isKernel ? 'Kernel (CUDA)' : 'Scope'}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Application</TableCell>
                <TableCell>{ev?.appName} (PID {ev?.pid})</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Track Tag</TableCell>
                <TableCell>{ev?.tag || 'default'}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>GPU</TableCell>
                <TableCell>{ev?.gpuName || 'GPU'} ({ev?.uuid || 'unknown'})</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Start</TableCell>
                <TableCell>{formatTimeMs(bar.startMs)}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>End</TableCell>
                <TableCell>{formatTimeMs(bar.endMs)}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Duration</TableCell>
                <TableCell>{Math.max(0, bar.durationMs).toFixed(3)} ms</TableCell>
              </TableRow>
              {isKernel && (
                <TableRow>
                  <TableCell>Kernel Metadata</TableCell>
                  <TableCell>
                    <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{prettyJson(extra)}</pre>
                  </TableCell>
                </TableRow>
              )}
              {!isKernel && extra && Object.keys(extra).length > 0 && (
                <TableRow>
                  <TableCell>Extra</TableCell>
                  <TableCell>
                    <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{prettyJson(extra)}</pre>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}
