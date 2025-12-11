import React, { useState, useMemo } from "react";
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

const LegendItem = ({ color, label }) => (
  <div className="flex items-center gap-2 text-gray-700 text-sm whitespace-nowrap">
    <span
      className="inline-block w-3 h-3 rounded-full"
      style={{ backgroundColor: color }}
    ></span>
    {label}
  </div>
);

const CustomLegend = () => (
  <div className="mt-4 flex flex-col items-center gap-2 text-sm">
    
    {/* Row 1: Portfolio */}
    <div className="flex flex-wrap gap-6 justify-center">
      <LegendItem color="#84c2251A" label="Portfolio 5th–95th Percentile" />
      <LegendItem color="#84c22540" label="Portfolio 25th–75th Percentile" />
      <LegendItem color="#84c225" label="Portfolio Median" />
    </div>

    {/* Row 2: S&P 500 */}
    <div className="flex flex-wrap gap-6 justify-center">
      <LegendItem color="#1f78ff0D" label="S&P 500 5th–95th Percentile" />
      <LegendItem color="#1f78ff26" label="S&P 500 25th–75th Percentile" />
      <LegendItem color="#1b3a70" label="S&P 500 Median" />
    </div>
  </div>
);

// --------------------------
// SCALING FIX HERE
// --------------------------
const scaleBands = (bands, factor = 100) => ({
  p5:  bands.p5.map(v => v * factor),
  p25: bands.p25.map(v => v * factor),
  p50: bands.p50.map(v => v * factor),
  p75: bands.p75.map(v => v * factor),
  p95: bands.p95.map(v => v * factor)
});

// smoothing helpers
const smoothSeries = (arr, window = 5) => {
  if (!arr) return arr;
  const n = arr.length;
  const half = Math.floor(window / 2);
  const out = new Array(n);

  for (let i = 0; i < n; i++) {
    let sum = 0;
    let count = 0;
    for (let j = i - half; j <= i + half; j++) {
      if (j >= 0 && j < n) {
        sum += arr[j];
        count++;
      }
    }
    out[i] = sum / count;
  }
  return out;
};

const smoothBands = (bands, window = 5) => ({
  p5:  smoothSeries(bands.p5,  window),
  p25: smoothSeries(bands.p25, window),
  p50: smoothSeries(bands.p50, window),                
  p75: smoothSeries(bands.p75, window),
  p95: smoothSeries(bands.p95, window),
});


const MonteCarloChart = ({ dates, pctBandsPortfolio, pctBandsSPY }) => {
  const [showPortfolio, setShowPortfolio] = useState(true);
  const [showSPY, setShowSPY] = useState(true);

  const smoothedPortfolioBands = useMemo(
  () => (pctBandsPortfolio ? smoothBands(pctBandsPortfolio, 5) : null),
  [pctBandsPortfolio]
  );

  const smoothedSPYBands = useMemo(
    () => (pctBandsSPY ? smoothBands(pctBandsSPY, 5) : null),
    [pctBandsSPY]
  );

  // --------------------------
  // APPLY SCALE TO 100 HERE
  // --------------------------
  const scaledPortfolio = smoothedPortfolioBands
    ? scaleBands(smoothedPortfolioBands, 100)
    : null;

  const scaledSPY = smoothedSPYBands
    ? scaleBands(smoothedSPYBands, 100)
    : null;

  // visual settings
  const borderWidth = 2;
  const pointRadius = 0;
  const pointHoverRadius = 2;

  // Build dataset groups in original visual style
  const buildBands = (pct, color, label) => {
    const isPortfolio = label === "Portfolio";
    const order = isPortfolio ? 10 : 1;
    
    return [
      {
        label: `${label} 5th–95th Percentile`,
        data: pct.p95,
        borderColor: "rgba(0,0,0,0)",
        backgroundColor: label === "Portfolio" ? `${color}33` : `${color}0D`,
        fill: "+1",
        tension: 0,
        order
      },
      {
        label: `${label} 5th Percentile`,
        data: pct.p5,
        borderColor: "rgba(0,0,0,0)",
        backgroundColor: label === "Portfolio" ? `${color}33` : `${color}0D`,
        fill: "-1",
        tension: 0,
        order
      },
      {
        label: `${label} 25th–75th Percentile`,
        data: pct.p75,
        borderColor: "rgba(0,0,0,0)",
        backgroundColor: label === "Portfolio" ? `${color}55` : `${color}26`,
        fill: "+1",
        tension: 0,
        order
      },
      {
        label: `${label} 25th Percentile`,
        data: pct.p25,
        borderColor: "rgba(0,0,0,0)",
        backgroundColor: label === "Portfolio" ? `${color}55` : `${color}26`,
        fill: "-1",
        tension: 0,
        order
      },
      {
        label: `${label} Median`,
        data: pct.p50,
        borderColor: color,
        borderWidth,
        pointRadius,
        pointHoverRadius,
        fill: false,
        tension: 1,
        order
      }
    ];
  };

  const datasets = useMemo(() => {
    let d = [];

    if (showSPY && scaledSPY)
      d = d.concat(buildBands(scaledSPY, "#1b3a70", "S&P 500"));

    if (showPortfolio && scaledPortfolio)
      d = d.concat(buildBands(scaledPortfolio, "#84c225", "Portfolio"));

    return d;
  }, [showPortfolio, showSPY, scaledPortfolio, scaledSPY]);

  const data = { labels: dates, datasets };

  const options = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        mode: "index",
        intersect: false,

        callbacks: {
          title: (items) => items[0].label,

          label: (ctx) => {
            const label = ctx.dataset.label || "";

            // Only show tooltip for median dataset
            if (!label.includes("Median")) return "";

            const isPortfolio = label.includes("Portfolio");
            const p = isPortfolio ? pctBandsPortfolio : pctBandsSPY;
            const key = isPortfolio ? "Portfolio" : "S&P 500";

            const i = ctx.dataIndex;

            return [
              `${key} Median: ${(p.p50[i] * 100).toFixed(2)}`,
              `${key} 25th–75th: ${(p.p25[i] * 100).toFixed(2)} – ${(p.p75[i] * 100).toFixed(2)}`,
              `${key} 5th–95th: ${(p.p5[i] * 100).toFixed(2)} – ${(p.p95[i] * 100).toFixed(2)}`
            ];
          },

          // Ensure the color box matches the median line
          labelColor: (ctx) => ({
            borderColor: ctx.dataset.borderColor,
            backgroundColor: ctx.dataset.borderColor
          }),

          labelPointStyle: () => ({
            pointStyle: "rect",
            rotation: 0
          }),

          labelTextColor: () => "#fff"
        }
      },
      zoom: {
        limits: { x: { minRange: 5 } },
        pan: { enabled: true, mode: "x" },
        zoom: {
          wheel: { enabled: true },
          pinch: { enabled: true },
          mode: "x"
        }
      }
    },
    scales: {
      x: {
        title: { display: true, text: "Date" },
        ticks: {
          autoSkip: true,
          maxRotation: 0,
          minRotation: 0
        }
      },
      y: {
        title: { display: true, text: "Growth of 100%" },
        beginAtZero: false
      }
    },
    elements: {
      line: {
        tension: 0.6,
      },
      point: {
        radius: 0,
        hoverRadius: 0,
        hitRadius: 0
      },
    }
  };

  return (
    <div className="w-full h-auto">
      <div className="flex gap-6 mb-2 justify-end items-center">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={showPortfolio}
            onChange={() => setShowPortfolio(!showPortfolio)}
            className="h-4 w-4 text-gfwmDarkGreen border-gray-300 rounded focus:ring-gfwmLightGreen"
          />
          Show Portfolio Bands
        </label>

        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={showSPY}
            onChange={() => setShowSPY(!showSPY)}
            className="h-4 w-4 text-gfwmDarkGreen border-gray-300 rounded focus:ring-gfwmLightGreen"
          />
          Show S&P 500 Bands
        </label>
      </div>

      <Line data={data} options={options} />

      <CustomLegend />

    </div>
  );
};

export default MonteCarloChart;
