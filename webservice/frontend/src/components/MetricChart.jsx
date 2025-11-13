import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend);

export default function MetricChart({ points, label, color = 'rgb(75, 192, 192)', datasets }) {
  const effectiveDatasets = Array.isArray(datasets) && datasets.length > 0
    ? datasets
    : [{ label, points, color }];

  const normalizedDatasets = effectiveDatasets.map((ds) => {
    const c = ds.color || 'rgb(75, 192, 192)';
    return {
      label: ds.label,
      data: ds.points,
      borderColor: c,
      backgroundColor: c.replace('rgb', 'rgba').replace(')', ', 0.2)'),
      parsing: false,
      pointRadius: 0,
      tension: 0.2,
    };
  });

  const data = { datasets: normalizedDatasets };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        type: 'linear',
        ticks: {
          maxRotation: 0,
          autoSkip: true,
          callback: (value, idx, ticks) => {
            const v = ticks?.[idx]?.value ?? value;
            const d = new Date(v);
            const hh = String(d.getHours()).padStart(2, '0');
            const mm = String(d.getMinutes()).padStart(2, '0');
            return `${hh}:${mm}`;
          },
        },
        title: { display: true, text: 'Time' },
      },
      y: {
        beginAtZero: true,
        title: { display: true, text: label || 'Value' },
      },
    },
    plugins: {
      legend: { display: true },
      tooltip: {
        callbacks: {
          title: (items) => {
            if (!items || !items.length) return '';
            const v = items[0].parsed.x;
            return new Date(v).toLocaleString();
          },
          label: (ctx) => {
            const value = ctx.parsed.y;
            const dataPoint = ctx.dataset?.data?.[ctx.dataIndex];
            const processName = dataPoint?.processName;
            const label = ctx.dataset?.label ?? '';

            if (processName) {
              return `${label}: ${value} (Process: ${processName})`;
            }
            return `${label}: ${value}`;
          },
        },
      },
    },
  };

  return (
    <div style={{ height: '400px', width: '100%' }}>
      <Line data={data} options={options} />
    </div>
  );
}
