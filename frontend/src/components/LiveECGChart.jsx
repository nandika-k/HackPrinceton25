import { Line } from 'react-chartjs-2';
import { useEffect, useRef, useState } from 'react';
import { Chart, LineElement, CategoryScale, LinearScale, PointElement } from 'chart.js';
Chart.register(LineElement, CategoryScale, LinearScale, PointElement);

export default function LiveECGChart({ ecg }) {
  const [dataPoints, setDataPoints] = useState([]);
  const chartRef = useRef();

  // Update chart with new data every time `ecg` prop changes
  useEffect(() => {
    setDataPoints(prev => {
      const updated = [...prev, ecg];
      return updated.slice(-50); // keep only last 50 points
    });
  }, [ecg]);

  const data = {
    labels: dataPoints.map((_, i) => i),
    datasets: [
  {
    label: 'ECG Signal',
    data: dataPoints,
    borderColor: 'rgb(255, 64, 78)',
    borderWidth: 2,
    tension: 0,         // Sharp corners like real ECG
    pointRadius: 0,     // No dots
    fill: false,
  },
]
  };

  const options = {
  responsive: true,
  animation: false,
  scales: {
    y: {
      min: -2,
      max: 2,
      grid: {
        color: 'rgba(255, 64, 78, 0.1)', // subtle white gridlines
        lineWidth: 1,
      },
      ticks: {
        color: 'rgba(255,255,255,0.3)', // optional: make ticks visible
      },
    },
    x: {
  display: false,
  grid: {
    color: 'rgba(255, 255, 255, 0.05)', 
    lineWidth: 1,
  },
  ticks: {
    color: 'rgba(255,255,255,0.3)', 
  },
},
  },
  plugins: {
    legend: {
      display: false,
    },
  },
};


  return <Line ref={chartRef} data={data} options={options} />;
}
