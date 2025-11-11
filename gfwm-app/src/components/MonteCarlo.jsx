import React, {useState, useEffect} from "react";

import CircularProgress from "@mui/material/CircularProgress";
import MonteCarloChart from "./MonteCarloChart";


const MonteCarlo = ({ portfolio, cash_percent, portfolio_weighing_scheme }) => {
    const [loading, setLoading] = useState(false);
    const [simulationResponse, setSimulationResponse] = useState({});
    const [showResults, setShowResults] = useState(false);
    const [hasUserSimulated, setHasUserSimulated] = useState(false);


    // User Inputs
    const [simData, setSimData] = useState({
        portfolio: portfolio|| {},
        cash_percent: cash_percent || 0.2,
        portfolio_weighing_scheme: portfolio_weighing_scheme,
        horizon: 120,
        num_paths: 5000,
        rebalancing_rule: "annual",
        compounding_type: "logarithmic",
    });

    // Input handlers
    const handleSliderChange = (key, value) => {
        setSimData((prev) => ({ ...prev, [key]: value }));
    };

    const handleSelectChange = (key, value) => {
        setSimData((prev) => ({ ...prev, [key]: value }));
    };


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

    setSimData((prev) => {
        const next = {
        ...prev,
        portfolio,
        cash_percent,
        portfolio_weighing_scheme,
        };
        // only update if something changed
        if (
        prev.portfolio === next.portfolio &&
        prev.cash_percent === next.cash_percent &&
        prev.portfolio_weighing_scheme === next.portfolio_weighing_scheme
        ) {
        return prev;
        }
        return next;
    });
    }, [portfolio, cash_percent, portfolio_weighing_scheme]);



    // --- Auto re-run simulation whenever the portfolio changes dynamically ---
    useEffect(() => {
        if (!hasUserSimulated) return;
        if (!portfolio || Object.keys(portfolio).length === 0) return;

        console.log("Detected portfolio or form setting change — re-running Monte Carlo simulation...");
        handleSimulate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [portfolio, cash_percent]);

    return (
        <div className="mt-12 mb-8">
            <header className="mb-6">
                <h2 className="text-2xl font-bold text-gray-800">Monte Carlo Simulation</h2>
                <p className="text-lg text-gray-600">
                Run a simulation to see a range of potential future portfolio outcomes based on historical volatility.
                </p>
            </header>

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
                className="w-full h-2 bg-gray-300 rounded-full appearance-none cursor-pointer accent-gfwmDarkGreen"
                />
            </div>

            {/* Number of Paths */}
            <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                Number of Simulation Paths: {simData.num_paths.toLocaleString()}
                </label>
                <input
                type="range"
                min="500"
                max="10000"
                step="500"
                value={simData.num_paths}
                onChange={(e) => handleSliderChange("num_paths", Number(e.target.value))}
                className="w-full h-2 bg-gray-300 rounded-full appearance-none cursor-pointer accent-gfwmDarkGreen"
                />
            </div>
            </div>

            {/* === Dropdowns Row (Rebalancing + Compounding) === */}
            <div className="flex flex-wrap gap-6 mt-2">
            <div className="flex-1 min-w-[160px]">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                Rebalancing Rule (Not Yet Implemented)
                </label>
                <select
                value={simData.rebalancing_rule}
                onChange={(e) => handleSelectChange("rebalancing_rule", e.target.value)}
                className="w-full border border-gray-300 rounded-md px-3 py-1.5 text-gray-700 text-sm focus:outline-none focus:ring-1 focus:ring-gfwmLightGreen"
                >
                <option value="monthly">Monthly</option>
                <option value="quarterly">Quarterly</option>
                <option value="annual">Annual</option>
                </select>
            </div>

            <div className="flex-1 min-w-[160px]">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                Compounding Type
                </label>
                <select
                value={simData.compounding_type}
                onChange={(e) => handleSelectChange("compounding_type", e.target.value)}
                className="w-full border border-gray-300 rounded-md px-3 py-1.5 text-gray-700 text-sm focus:outline-none focus:ring-1 focus:ring-gfwmLightGreen"
                >
                <option value="logarithmic">Logarithmic</option>
                <option value="arithmetic">Arithmetic</option>
                </select>
            </div>
            </div>

            {/* === Submit Button === */}
            <div className="mt-4">
            <button
                type="submit"
                disabled={loading}
                className="hover:opacity-75 bg-gfwmDarkGreen text-white px-6 py-2 rounded text-sm"
            >
                {loading ? "Simulating..." : "Simulate"}
            </button>
            </div>
        </form>

            {/* --- Simulation Results --- */}
            {(showResults || loading) && (
                <div className="mt-10 bg-white rounded-lg shadow-md p-6">
                    <header className="mb-4">
                    <h3 className="text-xl font-bold text-gray-800">Monte Carlo Projection</h3>
                    </header>

                    <div className="grid grid-cols-12 gap-8 items-start">
                    {/* === LEFT SIDE: Projection Statistics === */}
                    {!loading && simulationResponse?.stats ? (
                        <div className="col-span-12 md:col-span-3 space-y-2">
                        <p className="text-gray-700">
                            <span className="font-semibold">Mean Return:</span>{" "}
                            {`${(simulationResponse.stats.mean_return * 100).toFixed(2)}%`}
                        </p>
                        <p className="text-gray-700">
                            <span className="font-semibold">Volatility:</span>{" "}
                            {`${(simulationResponse.stats.volatility * 100).toFixed(2)}%`}
                        </p>
                        <p className="text-gray-700">
                            <span className="font-semibold">Sharpe Ratio:</span>{" "}
                            {simulationResponse.stats.sharpe_ratio?.toFixed(2)}
                        </p>
                        <p className="text-gray-700">
                            <span className="font-semibold">Max Drawdown:</span>{" "}
                            {`${(simulationResponse.stats.max_drawdown * 100).toFixed(2)}%`}
                        </p>
                        </div>
                    ) : (
                        <div className="col-span-12 md:col-span-3 flex items-center justify-center text-gray-500">
                        {loading && "Running simulation..."}
                        </div>
                    )}

                    {/* === RIGHT SIDE: Monte Carlo Chart OR Spinner === */}
                    <div className="col-span-12 md:col-span-9 h-[400px] flex items-center justify-center">
                        {loading ? (
                        <CircularProgress color="success" size={50} thickness={4} />
                        ) : simulationResponse?.times && simulationResponse?.pctBands ? (
                        <MonteCarloChart
                            dates={simulationResponse.times.map((d) =>
                            new Date(d).toLocaleDateString("en-US", {
                                month: "short",
                                year: "numeric",
                            })
                            )}
                            pctBands={simulationResponse.pctBands}
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