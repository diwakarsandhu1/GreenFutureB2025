import React from "react";
import { Pie } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";

// Register required Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend);

const PieChart = ({ weights }) => {
  // Sort weights in descending order
  const sortedWeights = weights.sort((a, b) => b - a);

  const largestWeight = sortedWeights[0];

  // Find the drop-off point
  let dropIndex = sortedWeights.length;
  for (let i = 1; i < sortedWeights.length; i++) {
    // Drop-off threshold: < 1/3 of highest weight
    if (sortedWeights[i] < largestWeight / 3.0) {
      dropIndex = i;
      break;
    }
  }

  // Divide into significant and "other" groups
  const significant = sortedWeights.slice(0, dropIndex);
  const others = sortedWeights.slice(dropIndex);

  // Calculate sums
  const top_stocks_total = significant.reduce((sum, item) => sum + item, 0);
  const other_stocks_total = others.reduce((sum, item) => sum + item, 0);
  const not_allocated = 100 - top_stocks_total - other_stocks_total;

  // Prepare labels and values for the pie chart
  const labels = [
    `Top ${dropIndex} stocks`, // Label for the significant group
    `Other ${others.length} stocks`, // Label for the rest
    "Cash Position",
  ];
  const values = [top_stocks_total, other_stocks_total, not_allocated];

  // Define the background colors for the slices
  const backgroundColors = [
    "#3e7738",
    "#84c225",
    "#F1E5A9", // Brighter yellow for Rest (Vibrant)
  ];

  // Function to darken a hex color by a certain factor
const darkenColor = (hex, factor = 0.1) => {
  // Ensure the factor is between 0 and 1
  factor = Math.min(Math.max(factor, 0), 1);

  // Remove the '#' if it's there
  hex = hex.replace(/^#/, "");

  // Parse the hex string into RGB components
  let r = parseInt(hex.substring(0, 2), 16);
  let g = parseInt(hex.substring(2, 4), 16);
  let b = parseInt(hex.substring(4, 6), 16);

  // Darken each RGB component
  r = Math.floor(r * (1 - factor));
  g = Math.floor(g * (1 - factor));
  b = Math.floor(b * (1 - factor));

  // Convert back to hex
  const toHex = (x) => x.toString(16).padStart(2, "0");

  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
};


  const chartData = {
    labels: labels,
    datasets: [
      {
        label: "Percent of portfolio",
        data: values,
        backgroundColor: backgroundColors,
        hoverBackgroundColor: backgroundColors.map(color => darkenColor(color, 0.2)),
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
        enabled: true,
      },
    },
    layout: {
      padding: 0, // Reduce extra padding around the chart
    },
  };

  return <Pie data={chartData} options={options} />;
};

export default PieChart;
