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
} from "chart.js";

import zoomPlugin from "chartjs-plugin-zoom";

// Register required Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  zoomPlugin
);

const Chart = ({ portfolio, spy, dates }) => {
  
//     //autogenerate labels
//   const generateLabels = (startDate, numPoints, intervalDays) => {
//     const labels = [];
//     const start = new Date(startDate);

//     for (let i = 0; i < numPoints; i++) {
//       const newDate = new Date(start);
//       newDate.setDate(start.getDate() + i * intervalDays);
//       labels.push(
//         `${
//           newDate.getMonth() + 1
//         }/${newDate.getDate()}/${newDate.getFullYear()}`
//       );
//     }

//     return labels;
//   };

//   const labels = generateLabels("2013-01-01", portfolio.length, 7);

  const tension = 0.2; // line smoothness
  const borderWidth = 2;
  const pointRadius = 0; // No circles displayed
  const pointHoverRadius = 2; // But still show hover effect

  const data = {
    labels: dates,
    datasets: [
      {
        label: "Portfolio",
        data: portfolio,
        borderColor: "#84c225", // Line color
        tension,
        borderWidth,
        pointRadius,
        pointHoverRadius,
      },
      {
        label: "S&P 500",
        data: spy,
        borderColor: "#9ca3af", // Line color
        tension,
        borderWidth,
        pointRadius,
        pointHoverRadius,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
      },
      tooltip: {
        mode: "index",
        intersect: false,
      },
      zoom: {
        pan: {
          enabled: true,
          mode: "x", // Allow horizontal panning
        },
        zoom: {
          wheel: {
            enabled: true, // Zoom with the mouse wheel
          },
          pinch: {
            enabled: true, // Zoom with touch gestures
          },
          mode: "x", // Allow horizontal zooming
        },
      },
    },
    scales: {
      x: {
        ticks: {
          autoSkip: true, // Automatically skip labels to avoid overcrowding
          maxRotation: 0, // Prevent label rotation
          minRotation: 0,
        },
        title: {
          display: true,
          text: "Date",
        },
      },
      y: {
        title: {
          display: true,
          text: "Growth of 100%",
        },
      },
    },
  };

  return <Line data={data} options={options} />;
};

export default Chart;
