import React, {useState, useEffect} from "react";

import CircularProgress from "@mui/material/CircularProgress";
import MonteCarloChart from "./MonteCarloChart";
import ComparisonTable from "./ComparisonTable";


const MonteCarlo = ({ portfolio, riskAppetite, portfolio_weighing_scheme }) => {
    const [loading, setLoading] = useState(false);
    const [simulationResponse, setSimulationResponse] = useState({});
    const [showResults, setShowResults] = useState(false);
    const [hasUserSimulated, setHasUserSimulated] = useState(false);
    const [userIsDragging, setUserIsDragging] = useState(false);


    // User Inputs
    const [simData, setSimData] = useState({
        portfolio: portfolio|| {},
        riskAppetite: riskAppetite || 0.1,
        portfolio_weighing_scheme: portfolio_weighing_scheme,
        horizon: 120,
        num_paths: 5000,
        rebalancing_rule: "Annually",
        advisor_fee: 0.01,
    });

    // Input handlers
    const handleSliderChange = (key, value) => {
        setSimData((prev) => ({ ...prev, [key]: value }));
    };

    const comparisonTableData = simulationResponse?.portfolio && simulationResponse?.sp500
        ? [
            {
                field: "Average Annual Return",
                portfolio: `${(simulationResponse.portfolio.stats.mean_return * 100).toFixed(2)}%`,
                sp500: `${(simulationResponse.sp500.stats.mean_return * 100).toFixed(2)}%`,
            },
            {
                field: "Average Return Compared to Benchmark",
                portfolio: `${((simulationResponse.portfolio.stats.mean_return - simulationResponse.sp500.stats.mean_return) * 100).toFixed(2)}%`,
                sp500: "",
            },
            {
                field: "Average Standard Deviation",
                portfolio: `${(simulationResponse.portfolio.stats.volatility * 100).toFixed(2)}%`,
                sp500: `${(simulationResponse.sp500.stats.volatility * 100).toFixed(2)}%`,
            },
            {
                field: "Estimated Sharpe Ratio",
                portfolio: simulationResponse.portfolio.stats.sharpe_ratio.toFixed(2),
                sp500: simulationResponse.sp500.stats.sharpe_ratio.toFixed(2),
            },
            {
                field: "1-Month VaR (95%)",
                portfolio: `${(simulationResponse.portfolio.stats.VaR['1m']['var95'] * 100).toFixed(2)}%`,
                sp500: `${(simulationResponse.sp500.stats.VaR['1m']['var95'] * 100).toFixed(2)}%`,
            },
            {
                field: "1-Month VaR (99%)",
                portfolio: `${(simulationResponse.portfolio.stats.VaR['1m']['var99'] * 100).toFixed(2)}%`,
                sp500: `${(simulationResponse.sp500.stats.VaR['1m']['var99'] * 100).toFixed(2)}%`,
            },
            {
                field: "Horizon VaR (95%)",
                portfolio: `${(simulationResponse.portfolio.stats.VaR['horizon']['var95'] * 100).toFixed(2)}%`,
                sp500: `${(simulationResponse.sp500.stats.VaR['horizon']['var95'] * 100).toFixed(2)}%`,
            },
            {
                field: "Horizon VaR (99%)",
                portfolio: `${(simulationResponse.portfolio.stats.VaR['horizon']['var99'] * 100).toFixed(2)}%`,
                sp500: `${(simulationResponse.sp500.stats.VaR['horizon']['var99'] * 100).toFixed(2)}%`,
            },
            ]
        : null;


    // Handle Simulation
    const handleSimulate = async (e) => {
        // Setup
        if (e) e.preventDefault();
        setLoading(true)
        setShowResults(false)
        if (!hasUserSimulated) setHasUserSimulated(true);
        console.log("Simulate parameters:", simData)

        // API Request
        try {
            const res = await fetch(`http://${import.meta.env.VITE_API_DOMAIN}/simulate/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(simData),
            });
            const data = await res.json()
            console.log("Server response:", data);
            console.log(import.meta.env);
            setSimulationResponse(data);
        } catch (err) {
            console.error("Error:", err);
            alert("Could not connect to the server. Please try again later.");
        } finally {
            setLoading(false);
            setShowResults(true);
        }
    };

    // Update simData when props change
    useEffect(() => {
        if (!portfolio || Object.keys(portfolio).length === 0) return;

        setSimData(prev => ({
            ...prev,
            portfolio,
            riskAppetite,
            portfolio_weighing_scheme,
        }));
    }, [portfolio, riskAppetite, portfolio_weighing_scheme]);


    // --- Auto re-run simulation whenever the portfolio changes dynamically ---
    useEffect(() => {
        if (!hasUserSimulated) return;
        if (userIsDragging) return;

        console.log("(2) Detected portfolio change — re-running Monte Carlo simulation...");
        console.log("(3) Sending to resimulate: ", portfolio)
        handleSimulate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [simData]);

    return (
        <div className="mt-12 mb-8">
            <header className="mb-6">
                <h2 className="text-2xl font-bold text-gray-800">Monte Carlo Simulation</h2>
                <p className="text-lg text-gray-600">
                Run a simulation to see a range of potential future portfolio outcomes based on historical volatility.
                </p>
            </header>

        <div className='w-11/12'>
            <form onSubmit={handleSimulate} className="space-y-6 pl-4">
                {/* === Sliders Row (Horizon + Paths) === */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Horizon */}
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                    Simulation Horizon (months): {simData.horizon}
                    </label>
                    <input
                    type="range"
                    min="12"
                    max="240"
                    step="12"
                    value={simData.horizon}
                    onChange={(e) => handleSliderChange("horizon", Number(e.target.value))}
                    onMouseDown={() => setUserIsDragging(true)}
                    onTouchStart={() => setUserIsDragging(true)}
                    onMouseUp={() => {
                        setUserIsDragging(false);
                        if (hasUserSimulated) handleSimulate();
                    }}
                    onTouchEnd={() => {
                        setUserIsDragging(false);
                        if (hasUserSimulated) handleSimulate();
                    }}
                    className="mt-2 w-full h-2 appearance-none bg-gray-300 rounded-full slider-thumb"
                    />
                </div>

                {/* Number of Paths */}
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                    Number of Simulation Paths: {simData.num_paths.toLocaleString()}
                    </label>
                    <input
                    type="range"
                    min="1000"
                    max="10000"
                    step="500"
                    value={simData.num_paths}
                    onChange={(e) => handleSliderChange("num_paths", Number(e.target.value))}
                    onMouseDown={() => setUserIsDragging(true)}
                    onTouchStart={() => setUserIsDragging(true)}
                    onMouseUp={() => {
                        setUserIsDragging(false);
                        if (hasUserSimulated) handleSimulate();
                    }}
                    onTouchEnd={() => {
                        setUserIsDragging(false);
                        if (hasUserSimulated) handleSimulate();
                    }}
                    className="mt-2 w-full h-2 appearance-none bg-gray-300 rounded-full slider-thumb"
                    />
                </div>

                {/* === Rebalancing Frequency Buttons === */}
                <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                    Rebalancing Frequency
                </label>

                <div className="flex justify-left space-x-4 mx-4">
                    {["Quarterly", "Semi-Annually", "Annually"].map((option) => (
                    <label key={option} className="flex items-center space-x-2">
                        <input
                        type="radio"
                        name="rebalancingFrequency"
                        value={option}
                        checked={simData.rebalancing_rule === option}
                        onChange={() => handleSliderChange("rebalancing_rule", option)}
                        className="h-5 w-5 text-gfwmDarkGreen focus:ring-gfwmLightGreen border-gray-300 rounded-full bg-white"
                        />
                        <span>{option}</span>
                    </label>
                    ))}
                </div>
                </div>

                {/* === Advisor Fee Slider === */}
                <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                    Advisor Fee: {(simData.advisor_fee * 100).toFixed(2)}%
                </label>
                <input
                    type="range"
                    min={0.0001}
                    max={0.025}
                    step={0.0001}
                    value={simData.advisor_fee}
                    onChange={(e) => handleSliderChange("advisor_fee", Number(e.target.value))}
                    onMouseDown={() => setUserIsDragging(true)}
                    onTouchStart={() => setUserIsDragging(true)}
                    onMouseUp={() => {
                        setUserIsDragging(false);
                        if (hasUserSimulated) handleSimulate();
                    }}
                    onTouchEnd={() => {
                        setUserIsDragging(false);
                        if (hasUserSimulated) handleSimulate();
                    }}
                    className="mt-2 w-full h-2 appearance-none bg-gray-300 rounded-full slider-thumb"
                />
                </div>
                </div>

                {/* === Submit Button === */}
                <div className="mt-4">
                <button
                    type="submit"
                    disabled={loading}
                    className="hover:opacity-75 bg-gfwmDarkGreen text-white px-4 py-2 rounded mb-4"
                >
                    {loading ? "Simulating..." : "Simulate"}
                </button>
                </div>
            </form>
        </div>

        {/* --- Simulation Results --- */}
        {(showResults || loading) && (
            <div className="mt-10 bg-white rounded-lg shadow-md p-6">
                <header className="mb-4">
                <h3 className="text-xl font-bold text-gray-800">Monte Carlo Projection</h3>
                </header>

                <div className="grid grid-cols-12 gap-8 items-start">
                {/* === LEFT SIDE: Projection Statistics === */}
                {!loading && comparisonTableData ? (
                    <div className="col-span-12 md:col-span-4">
                        <ComparisonTable data={comparisonTableData} />
                    </div>
                ) : (
                    <div className="col-span-12 md:col-span-8 flex items-center justify-center text-gray-500">
                        {loading && "Running simulation..."}
                    </div>
                )}

                {/* === RIGHT SIDE: Monte Carlo Chart OR Spinner === */}
                <div className="col-span-12 md:col-span-8 flex items-center justify-center">
                    {loading ? (
                    <CircularProgress color="success" size={50} thickness={4} />
                    ) : simulationResponse?.times && simulationResponse?.portfolio?.pctBands ? (
                    <MonteCarloChart
                        dates={simulationResponse.times.map((d) =>
                        new Date(d).toLocaleDateString("en-US", {
                            month: "short",
                            year: "numeric",
                        })
                        )}
                        pctBandsPortfolio={simulationResponse.portfolio.pctBands}
                        pctBandsSPY={simulationResponse.sp500.pctBands}
                    />
                    ) : (
                    <p className="text-gray-500">Run a simulation to see results.</p>
                    )}
                </div>
                </div>
            </div>
        )}


        </div>

    );
};

export default MonteCarlo;