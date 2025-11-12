import React from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";
import zoomPlugin from "chartjs-plugin-zoom";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  zoomPlugin
);

const MonteCarloChart = ({ dates, pctBands }) => {
  const tension = 0.2; // line smoothness
  const borderWidth = 2;
  const pointRadius = 0;
  const pointHoverRadius = 2;

  const data = {
    labels: dates,
    datasets: [
      // 95–5% shaded band
      {
        label: "5th–95th Percentile",
        data: pctBands.p95,
        borderColor: "rgba(0,0,0,0)", // invisible line
        backgroundColor: "rgba(132,194,37,0.1)", // soft green fill
        fill: "+1", // fill to next dataset (p5)
      },
      {
        label: "5th Percentile",
        data: pctBands.p5,
        borderColor: "rgba(0,0,0,0)",
        backgroundColor: "rgba(132,194,37,0.1)",
        fill: "-1",
      },
      // 75–25% shaded band
      {
        label: "25th–75th Percentile",
        data: pctBands.p75,
        borderColor: "rgba(0,0,0,0)",
        backgroundColor: "rgba(132,194,37,0.25)",
        fill: "+1",
      },
      {
        label: "25th Percentile",
        data: pctBands.p25,
        borderColor: "rgba(0,0,0,0)",
        backgroundColor: "rgba(132,194,37,0.25)",
        fill: "-1",
      },
      // Median line (p50)
      {
        label: "Median (p50)",
        data: pctBands.p50,
        borderColor: "#84c225",
        borderWidth,
        tension,
        pointRadius,
        pointHoverRadius,
        fill: false,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "bottom",
        labels: {
          boxWidth: 15,
          usePointStyle: true,
          font: { size: 12 },
        },
      },
      tooltip: {
        mode: "index",
        intersect: false,
      },
      zoom: {
        limits: {
        x: { minRange: 5 },
        y: { min: -0.2, max: 2 },
        },
        pan: { enabled: true, mode: "x" },
        zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: "x" },
      },
    },
    scales: {
      x: {
        title: { display: true, text: "Date" },
        ticks: { autoSkip: true, maxRotation: 0, minRotation: 0 },
      },
      y: {
        title: { display: true, text: "Portfolio Growth (%)" },
        beginAtZero: false,
      },
    },
  };


  return <Line data={data} options={options} />;
};

export default MonteCarloChart;
