// LiveAccelChart.jsx
import { Line } from "react-chartjs-2";
import { useEffect, useRef, useState } from "react";
import {
  Chart,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
} from "chart.js";

Chart.register(LineElement, CategoryScale, LinearScale, PointElement);

export default function LiveAccelChart({ ax, ay, az }) {
  const [dataX, setDataX] = useState([]);
  const [dataY, setDataY] = useState([]);
  const [dataZ, setDataZ] = useState([]);

  useEffect(() => {
    setDataX((prev) => [...prev, ax].slice(-50));
    setDataY((prev) => [...prev, ay].slice(-50));
    setDataZ((prev) => [...prev, az].slice(-50));
  }, [ax, ay, az]);

  const labels = Array.from({ length: dataX.length }, (_, i) => i);

  const data = {
    labels,
    datasets: [
      {
        label: "Accel X",
        data: dataX,
        borderColor: "#FFA500", // orange
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0,
      },
      {
        label: "Accel Y",
        data: dataY,
        borderColor: "#00CED1", // turquoise
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0,
      },
      {
        label: "Accel Z",
        data: dataZ,
        borderColor: "#ADFF2F", // green-yellow
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0,
      },
    ],
  };

  const options = {
    responsive: true,
    animation: false,
    scales: {
      y: {
        min: -0.1,
        max: 0.1,
        grid: {
          color: "rgba(255,255,255,0.05)",
        },
        ticks: {
          color: "rgba(255,255,255,0.4)",
        },
      },
      x: {
        display: false,
        grid: {
            color: "rgba(255,255,255,0.05)",
          display: false,
        },
      },
    },
    plugins: {
      legend: {
        labels: {
          color: "rgba(255,255,255,0.7)",
        },
      },
    },
  };

  return <Line data={data} options={options} />;
}
