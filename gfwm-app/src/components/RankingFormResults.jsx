import React from "react";
import { useEffect, useState } from "react";

import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from "chart.js";

import Papa from "papaparse";
import StockSearchModal from "./StockSearchModal";

import ComparisonTable from "./ComparisonTable";
import PieChart from "./PieChart";
import TimeseriesChart from "./TimeseriesChart";

import MonteCarlo from "./MonteCarlo";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const RankingFormResults = ({ serverResponse, formResults }) => {
  //console.log('received props ', serverResponse, formResults);
  const [serverPortfolio, setServerPortfolio] = useState(serverResponse?.portfolio || {});
  const [compatibilityScores, setCompatibilityScores] = useState({});
  const [portfolioWeights, setPortfolioWeights] = useState({});
  const [initialized, setInitialized] = useState(false);


  // unpack server response
  const [portfolioSummaryStatistics, setPortfolioSummaryStatistics] = useState({
    ESGScore: 0,
    returnRange: [0, 0],
    averageReturn: 0,
    volatility: 0,
    sharpe: 0,
    maxDD: 0,
    timeseries: [],
  });
  const [sp500SummaryStatistics, setSP500SummaryStatistics] = useState({
    averageReturn: 0,
    volatility: 0,
    sharpe: 0,
    maxDD: 0,
    timeseries: [],
    timeseriesDates: [],
  });

  // unpack server response and update state variables
  useEffect(() => {

    if (!serverResponse || initialized) return;
    
    setCompatibilityScores(serverResponse.sp500_compatibility);

    // stores portfolio weights in {ticker: weight} object format
    // initialized by server response to form submission
    // modified when user adds and removes stocks in portfolio
    setPortfolioWeights(serverResponse.portfolio);

    // unpack server's summary statistics
    const {
      portfolio_esg_score: portfolioESGScore,
      portfolio_return_range: portfolioReturnRange,
      portfolio_volatility: portfolioVolatility,
      portfolio_sharpe: portfolioSharpe,
      portfolio_max_dd: portfolioMaxDD,
      portfolio_timeseries: portfolioTimeseries,
      sp500_average_return: sp500AverageReturn,
      sp500_average_volatility: sp500AverageVolatility,
      sp500_sharpe: sp500Sharpe,
      spy_max_dd: spyMaxDD,
      spy_timeseries: spyTimeseries,
      timeseries_dates: timeseriesDates,
    } = serverResponse.summary_statistics;

    console.log("portfolioSharpe ", portfolioSharpe);

    setPortfolioSummaryStatistics({
      ESGScore: portfolioESGScore,
      returnRange: portfolioReturnRange,
      averageReturn: (portfolioReturnRange[0] + portfolioReturnRange[1]) / 2.0,
      volatility: portfolioVolatility,
      sharpe: portfolioSharpe,
      maxDD: portfolioMaxDD,
      timeseries: portfolioTimeseries,
    });
    setSP500SummaryStatistics({
      averageReturn: sp500AverageReturn,
      volatility: sp500AverageVolatility,
      sharpe: sp500Sharpe,
      maxDD: spyMaxDD,
      timeseries: spyTimeseries,
      timeseriesDates,
    });
    
    setInitialized(true);
  }, [serverResponse]);

  // serialized version of esg preferences, just to check if changes occured
  const [esgPreferences, setESGPreferences] = useState("");

  const [riskAppetite, setRiskAppetite] = useState(0.1);
  const [weighingScheme, setWeighingScheme] = useState("");

  const [isPortfolioOutdated, setIsPortfolioOutdated] = useState(false);

  // unpack form results and update state variables
  useEffect(() => {
    if (formResults) {
      console.log("form results changed ", formResults);
      // unpack form results
      const { risk_appetite: riskAppetite, weighing_scheme: weighingScheme, ...rest } = formResults;

      console.log(rest);

      setRiskAppetite(riskAppetite);
      setWeighingScheme(weighingScheme);

      const sortedString = Object.keys(rest)
        .sort() // Sort keys alphabetically
        .map((key) => `${key}:${rest[key]}`) // Create key:value pairs
        .join("; "); // Join them into a single string

      // Update state only if the string changes
      if (sortedString !== esgPreferences) {
        setESGPreferences(sortedString);
        console.log("ESG preferences updated:", sortedString);
      }
    }
  }, [formResults]);

  // real time updates of data based off changes to risk slider
  useEffect(() => {
    //check if server response is truthy, ie some data sent back already
    // no point in updating risk if esg portfolio is outdated too
    if (serverResponse && !isPortfolioOutdated) {
      // package up ticker and weight columns to send to server
      const clientPortfolio = portfolioData.map((row) => {
        return {
          ticker: row.ticker,
          weight: row.weight,
        };
      });

      if (clientPortfolio.length > 0) {
        const body = JSON.stringify({
          client_portfolio: clientPortfolio,
          risk_appetite: riskAppetite,
        });

        // Send the necessary data to the server
        fetch(`http://${import.meta.env.VITE_API_DOMAIN}/updateRisk/`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: body,
        })
          .then((res) => {
            if (!res.ok) {
              console.log(res);
              throw new Error("Network response was not ok");
            }
            return res.json();
          })
          .then(({ updated_portfolio: updatedPortfolio, updated_summary_statistics: updatedSummaryStatistics }) => {
            //unpack response to get portfolio object (tickers and weights) and summary statistics

            console.log("Server response updated portfolio: ", updatedPortfolio);
            console.log("Server response updated summary stats: ", updatedSummaryStatistics);

            // update portfolio data with new weights via listener on portfolioWeights
            setPortfolioWeights(updatedPortfolio);
            setServerPortfolio(updatedPortfolio);

            const {
              portfolio_return_range: portfolioReturnRange,
              portfolio_volatility: portfolioVolatility,
              portfolio_sharpe: portfolioSharpe,
              portfolio_max_dd: portfolioMaxDD,
              portfolio_timeseries: portfolioTimeseries,
            } = updatedSummaryStatistics;

            setPortfolioSummaryStatistics({
              ESGScore: portfolioSummaryStatistics.ESGScore, //unchanged esg score when modifying risk
              returnRange: portfolioReturnRange,
              averageReturn: (portfolioReturnRange[0] + portfolioReturnRange[1]) / 2.0,
              volatility: portfolioVolatility,
              sharpe: portfolioSharpe,
              maxDD: portfolioMaxDD,
              timeseries: portfolioTimeseries,
            });
          })
          .catch((error) => {
            console.error("Error:", error);
          });
      }
    }
  }, [riskAppetite]);

  // real time updates of data based off weighing scheme changes
  useEffect(() => {
    //check if server response is truthy, ie some data sent back already
    if (serverResponse && !isPortfolioOutdated) {
      // package up tickers to update weights for
      const tickers = portfolioData.map((row) => row.ticker);
      if (tickers.length > 0) {
        console.log(tickers);
        updatePortfolioWeights(tickers);
      }
    }
  }, [weighingScheme]);

  useEffect(() => {
    if (esgPreferences) {
      setIsPortfolioOutdated(true);
      console.log("esg preferences changed ", esgPreferences);
    }
  }, [esgPreferences]);

  // MANAGE STOCK DATA OBJECT

  // contains all stocks the client might want to invest in along with compatibility scores
  // stocks removed by fossil fuels / weapons have compatibility 0 by default
  //    (as the server does not return compatibilities for those)
  const [stockData, setStockData] = useState([]);

  // runs when server's compatibility score changes
  // augments preprocessed csv with compatibility scores from server
  useEffect(() => {
    // stop if there are no compatibility scores to augment stockData with
    if (!compatibilityScores) return;

    const augmentStockData = async () => {
      try {
        const preprocessedCSV = await fetch("./preprocessed_refinitiv.csv"); //public version of preprocessed
        const preprocessedCSVText = await preprocessedCSV.text();
        const parseResult = Papa.parse(preprocessedCSVText, {
          header: true,
          skipEmptyLines: true,
        });
        // List of columns known to be numeric
        //TODO make this not hardcoded
        const numericColumns = [
          "esg_combined",
          "controversy",
          "environment",
          "social",
          "governance",
          "human_rights",
          "community",
          "workforce",
          "product_responsibility",
          "shareholders",
          "management",
          "annual_return",
          "volatility",
          "fossil_fuels",
          "weapons",
          "tobacco",
        ];

        //augment data with compatibility column pulled from server response
        const augmentedData = parseResult.data.map((row) => {
          // Convert specific numeric columns from strings to numbers
          const parsedRow = {
            ...row,
            ...Object.fromEntries(
              numericColumns.map((col) => [col, Number(row[col]) || 0]) // Convert or default to 0
            ),
          };

          // Match ticker to score or assign 0
          // 0 happens if the ticker was excluded for fossil fuels or weapons involvement
          const compatibility = compatibilityScores[row.ticker] || 0;

          return {
            ...parsedRow,
            compatibility, // Add compatibility score as a new property
          };
        });

        console.log("updated stock data wth compatibility", augmentedData);

        setStockData(augmentedData);
      } catch (error) {
        console.error("Error with augmenting preprocessed CSV with server's compatibility scores:", error);
      }
    };

    augmentStockData();
  }, [compatibilityScores]);
  //--------------------------------

  // MANAGE PORTFOLIO DATA OBJECT

  // copy of the selected stocks from the overall S&P 500 data
  // also includes weight column
  const [portfolioData, setPortfolioData] = useState([]);

  // runs when the server updates the portfolio weights object
  // copies correct rows (tickers) from stockData and augments with given weight
  useEffect(() => {
    if (!portfolioWeights) return;

    setIsPortfolioOutdated(false);

    const updatePortfolioData = async () => {
      console.log("updating portfolio data based off weights change: ", Object.keys(portfolioWeights).length);

      //TODO speed this up either by
      // 1. returning a proper object of {ticker1: weight1, ticker2: weight2 ...}
      // 2. making a hashmap on the client of the same format

      // first check if the ticker exists in the portfolio object
      const portfolioData = stockData
        .filter((row) => portfolioWeights.hasOwnProperty(row.ticker))
        .map((row) => {
          //then access the weight directly by ticker
          const weight = portfolioWeights[row.ticker] || 0;
          return {
            ...row,
            weight, // Add the weight property
          };
        });
      setPortfolioData(portfolioData);
    };

    updatePortfolioData();
  }, [portfolioWeights, stockData]);
  //-----------------------------------

  // HANDLE SORTING OF PORTFOLIO DATA

  const [sortConfig, setSortConfig] = useState({
    key: "compatibility", //default sort by compatibility
    direction: "descending",
  });

  // Sort portfolio data based on sortConfig
  const sortedPortfolioData = React.useMemo(() => {
    let sortableData = [...portfolioData]; //shallow copy of memoized data
    if (sortConfig !== null) {
      sortableData.sort((a, b) => {
        if (a[sortConfig.key] < b[sortConfig.key]) {
          return sortConfig.direction === "ascending" ? -1 : 1;
        }
        if (a[sortConfig.key] > b[sortConfig.key]) {
          return sortConfig.direction === "ascending" ? 1 : -1;
        }
        return 0;
      });
    }
    return sortableData;
  }, [portfolioData, sortConfig]);

  // Handle sorting
  const requestSort = (key) => {
    let direction = "descending";
    if (sortConfig.key === key && sortConfig.direction === "descending") {
      direction = "ascending";
    }
    setSortConfig({ key, direction });
  };
  //------------------------------

  const rowRefs = React.useRef([]);

  // VALUES FOR COMPARISON TABLE
  const comparisonTableData = [
    {

      field: "Annual Return",
      portfolio: `${(portfolioSummaryStatistics.averageReturn * 100).toFixed(2)}%`,

      sp500: `${(sp500SummaryStatistics.averageReturn * 100).toFixed(2)}%`,
    },
    {
      field: "Average Return Compared to Benchmark",
      portfolio: `${((portfolioSummaryStatistics.averageReturn - sp500SummaryStatistics.averageReturn) * 100).toFixed(
        2
      )}%`,
      sp500: "",
    },
    {
      field: "Average Standard Deviation",
      portfolio: `${(portfolioSummaryStatistics.volatility * 100).toFixed(2)}%`,
      sp500: `${(sp500SummaryStatistics.volatility * 100).toFixed(2)}%`,
    },
    // {
    //   field: "Growth of $10k in 10 years",
    //   portfolio: `$${growth_of_10k_10_years.toFixed(2)}`,
    //   sp500: `$${(10000 * Math.pow(1 + sp500_average_return, 10)).toFixed(2)}`,
    // },
    {
      field: "Estimated Sharpe Ratio",
      portfolio: `${portfolioSummaryStatistics.sharpe.toFixed(2)}`,
      sp500: `${sp500SummaryStatistics.sharpe.toFixed(2)}`,
    },
    {
      field: "Estimated Max Drawdown",
      portfolio: `${(portfolioSummaryStatistics.maxDD * 100).toFixed(2)}%`,
      sp500: `${(sp500SummaryStatistics.maxDD * 100).toFixed(2)}%`,
    },
    {
      field: "Portfolio ESG Score",
      portfolio: `${portfolioSummaryStatistics.ESGScore.toFixed(2)}`,
      sp500: "66.66",
    },
    {
      field: "Number of stocks",
      portfolio: `${portfolioData.length}`,
      sp500: "500",
    },
  ];

  // OPEN TABLEAU DASHBOARD FOR SELECTED TICKER
  const [selectedTicker, setSelectedTicker] = React.useState(null);

  const openTableauDashboard = (ticker) => {
    setSelectedTicker(ticker);

    const popup = window.open("", "_blank", "width=1300,height=800");
    const embedCode = `
    <div class='tableauPlaceholder' id='viz1731438480314' style='position: relative'>
    <head><title>Individual Stock View</title></head>
        <noscript>
          <a href='#'>
            <img alt='Dashboard 1' src='https://public.tableau.com/static/images/In/IndividualStock3_0/Dashboard1/1_rss.png' style='border: none' />
          </a>
        </noscript>
        <object class='tableauViz' style='display:none;'>
          <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
          <param name='embed_code_version' value='3' />
          <param name='site_root' value='' />
          <param name='name' value='IndividualStock3_0_17337605280590/PrimaryDashboard' />
          <param name='tabs' value='no' />
          <param name='toolbar' value='yes' />
          <param name='static_image' value='https://public.tableau.com/static/images/In/IndividualStock3_0/Dashboard1/1.png' />
          <param name='animate_transition' value='yes' />
          <param name='display_static_image' value='yes' />
          <param name='display_spinner' value='yes' />
          <param name='display_overlay' value='yes' />
          <param name='display_count' value='yes' />
          <param name='language' value='en-US' />
          <param name='filter' value='Symbol=${ticker}' />
        </object>
      </div>
      <script type='text/javascript'>
        var divElement = document.getElementById('viz1731438480314');
        var vizElement = divElement.getElementsByTagName('object')[0];
      
          vizElement.style.width = '100%';
          vizElement.style.height = '100%';
        
        var scriptElement = document.createElement('script');
        scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
        vizElement.parentNode.insertBefore(scriptElement, vizElement);
      </script>
      `;
    popup.document.open();
    popup.document.write(embedCode);
    popup.document.close();
  };
  //--------------

  //SCROLL TO TOP
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  //ADD and REMOVE from portfolio

  const updatePortfolioWeights = (tickers) => {
    // package up ticker and compatibility columns to send to server
    const clientPortfolio = tickers.map((ticker) => {
      // Find the object in portfolio that matches the ticker
      const match = stockData.find((item) => item.ticker === ticker);

      if (!match) {
        console.log("match not found for ", ticker);
      }

      return {
        ticker: ticker,
        compatibility: match.compatibility,
      };
    });

    const body = JSON.stringify({
      client_portfolio: clientPortfolio,
      risk_appetite: riskAppetite,
      weighing_scheme: weighingScheme,
    });

    // Send the necessary data to the server
    fetch(`http://${import.meta.env.VITE_API_DOMAIN}/updateWeights/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: body,
    })
      .then((res) => {
        if (!res.ok) {
          console.log(res);
          throw new Error("Network response was not ok");
        }
        return res.json();
      })
      .then(({ updated_portfolio: updatedPortfolio, updated_summary_statistics: updatedSummaryStatistics }) => {
        //unpack response to get portfolio object (tickers and weights) and summary statistics

        console.log("(1) Server response updated portfolio: ", updatedPortfolio);
        console.log("Server response updated summary stats: ", updatedSummaryStatistics);

        // update portfolio data with new weights via listener on portfolioWeights
        setPortfolioWeights(updatedPortfolio);
        setServerPortfolio(updatedPortfolio);

        const {
          portfolio_return_range: portfolioReturnRange,
          portfolio_volatility: portfolioVolatility,
          portfolio_sharpe: portfolioSharpe,
          portfolio_max_dd: portfolioMaxDD,
          portfolio_esg_score: portfolioESGScore,
          portfolio_timeseries: portfolioTimeseries,
        } = updatedSummaryStatistics;

        setPortfolioSummaryStatistics({
          ESGScore: portfolioESGScore, // TODO MUST CHANGE ESG SCORE
          returnRange: portfolioReturnRange,
          averageReturn: (portfolioReturnRange[0] + portfolioReturnRange[1]) / 2.0,
          volatility: portfolioVolatility,
          sharpe: portfolioSharpe,
          maxDD: portfolioMaxDD,
          timeseries: portfolioTimeseries,
        });
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  };

  const handleAddStocks = (addedStocks) => {
    const oldTickers = portfolioData.map((item) => item.ticker);
    var newTickers = addedStocks.map((item) => item.ticker);
    newTickers = newTickers.filter((ticker) => !oldTickers.includes(ticker));

    // updating weights triggers updating rest of portfolio data
    updatePortfolioWeights(oldTickers.concat(newTickers));

    setSelectedTicker(addedStocks[0].ticker);
  };

  const handleRemoveTickers = (removedTickers) => {
    var oldTickers = portfolioData.map((item) => item.ticker);
    oldTickers = oldTickers.filter((ticker) => !removedTickers.includes(ticker));

    // keep the rows whose ticker is not in the list of tickers to remove
    updatePortfolioWeights(oldTickers.filter((ticker) => !removedTickers.includes(ticker)));

    setSelectedTicker(removedTickers[0]);
  };

  const [isModalOpen, setIsModalOpen] = React.useState(false);
  const [selectedTickers, setSelectedTickers] = useState([]);

  const handleSelectStock = (ticker) => {
    // toggle selected ticker to or from list of selected tickers
    setSelectedTickers((prevSelected) =>
      prevSelected.includes(ticker) ? prevSelected.filter((item) => item !== ticker) : [...prevSelected, ticker]
    );
    setSelectedTicker(ticker);
  };

  return (
    // grey out portfolio data if outdated, blur when modal is open
    <div>
      <div className={`${isPortfolioOutdated ? "opacity-50" : ""} ${isModalOpen ? "blur-sm" : ""} transition`}>
        <h2 className="text-2xl font-bold text-gray-800">Portfolio Summary</h2>

        {/* portfolio summary statistics */}
        <div className="flex flex-wrap flex-col lg:flex-row gap-4 items-start w-full">
          {/* Comparison Table */}
          <div className="flex-none w-full sm:w-[35%] min-w-[200px]">
            <ComparisonTable data={comparisonTableData} />
          </div>

          {/* Timeseries Chart */}
          <div className="flex-1 w-[40%] min-w[200px]">
            <TimeseriesChart
              portfolio={portfolioSummaryStatistics.timeseries}
              spy={sp500SummaryStatistics.timeseries}
              dates={sp500SummaryStatistics.timeseriesDates}
            />
          </div>

          {/* Pie Chart */}
          <div className="flex-none w-[20%] min-w-[100px]">
            <PieChart weights={portfolioData.map((item) => item.weight * 100)} />
          </div>
        </div>

          { /* Monte Carlo Simulation */}
          <MonteCarlo portfolio={serverPortfolio} riskAppetite={riskAppetite} portfolio_weighing_scheme={weighingScheme}/>

        {/* portfolio results header and edit button */}
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold text-gray-800">Your Portfolio</h2>
          <button
            onClick={() => setIsModalOpen(true)}
            className="hover:opacity-75 bg-gfwmDarkGreen text-white px-4 py-2 rounded"
          >
            Edit Portfolio
          </button>
        </div>

        <div className="relative">
          <table className="min-w-full bg-white mb-2 text-sm">
            <thead className="sticky top-0 bg-white </tr>z-10">
              <tr title="Sort data">
                <th className="py-1 px-2 border-b-2 border-gray-300 text-left leading-4 text-gray-600 tracking-wider"></th>

                <th
                  className="py-1 px-2 border-b-2 border-gray-300 text-left leading-4 text-gray-600 tracking-wider cursor-pointer"
                  onClick={() => requestSort("ticker")}
                >
                  Symbol {sortConfig.key === "ticker" && (sortConfig.direction === "ascending" ? "▲" : "▼")}
                </th>

                <th
                  className="py-1 px-2 border-b-2 border-gray-300 text-left leading-4 text-gray-600 tracking-wider cursor-pointer"
                  onClick={() => requestSort("name")}
                >
                  Name {sortConfig.key === "name" && (sortConfig.direction === "ascending" ? "▲" : "▼")}
                </th>
                <th
                  className="py-1 px-2 border-b-2 border-gray-300 text-left leading-4 text-gray-600 tracking-wider cursor-pointer"
                  onClick={() => requestSort("annual_return")}
                >
                  Annualized Return{" "}
                  {sortConfig.key === "annual_return" && (sortConfig.direction === "ascending" ? "▲" : "▼")}
                </th>
                <th
                  className="py-1 px-2 border-b-2 border-gray-300 text-left leading-4 text-gray-600 tracking-wider cursor-pointer"
                  onClick={() => requestSort("volatility")}
                >
                  Standard Deviation{" "}
                  {sortConfig.key === "volatility" && (sortConfig.direction === "ascending" ? "▲" : "▼")}
                </th>
                <th
                  className="py-1 px-2 border-b-2 border-gray-300 text-left leading-4 text-gray-600 tracking-wider cursor-pointer"
                  onClick={() => requestSort("esg_combined")}
                >
                  Combined ESG {sortConfig.key === "esg_combined" && (sortConfig.direction === "ascending" ? "▲" : "▼")}
                </th>
                <th
                  className="py-1 px-2 border-b-2 border-gray-300 text-left leading-4 text-gray-600 tracking-wider cursor-pointer"
                  onClick={() => requestSort("environment")}
                >
                  Environment {sortConfig.key === "environment" && (sortConfig.direction === "ascending" ? "▲" : "▼")}
                </th>
                <th
                  className="py-1 px-2 border-b-2 border-gray-300 text-left leading-4 text-gray-600 tracking-wider cursor-pointer"
                  onClick={() => requestSort("social")}
                >
                  Social {sortConfig.key === "social" && (sortConfig.direction === "ascending" ? "▲" : "▼")}
                </th>
                <th
                  className="py-1 px-2 border-b-2 border-gray-300 text-left leading-4 text-gray-600 tracking-wider cursor-pointer"
                  onClick={() => requestSort("governance")}
                >
                  Governance {sortConfig.key === "governance" && (sortConfig.direction === "ascending" ? "▲" : "▼")}
                </th>
                <th
                  className="py-1 px-2 border-b-2 border-gray-300 text-left leading-4 text-gray-600 tracking-wider cursor-pointer"
                  onClick={() => requestSort("compatibility")}
                >
                  Compatibility{" "}
                  {sortConfig.key === "compatibility" && (sortConfig.direction === "ascending" ? "▲" : "▼")}
                </th>
                <th
                  className="py-1 px-2 border-b-2 border-gray-300 text-left leading-4 text-gray-600 tracking-wider cursor-pointer"
                  onClick={() => requestSort("weight")}
                >
                  Weight {sortConfig.key === "weight" && (sortConfig.direction === "ascending" ? "▲" : "▼")}
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedPortfolioData.map((item, index) => {
                const isSelected = selectedTickers.includes(item.ticker);
                return (
                  <tr
                    ref={(el) => (rowRefs.current[item.ticker] = el)}
                    key={item.ticker}
                    className={`hover:bg-gray-100 cursor-pointer ${
                      selectedTicker === item.ticker ? "bg-gray-100" : ""
                    }`}
                    onClick={() => openTableauDashboard(item.ticker)}
                    title="Show more"
                  >
                    <td
                      className="border-b text-right pl-2 border-gray-300 cursor-pointer "
                      onClick={(e) => {
                        e.stopPropagation();
                        handleSelectStock(item.ticker);
                      }}
                    >
                      <input
                        className="cursor-pointer h-5 w-5 text-gfwmDarkGreen focus:ring-gfwmLightGreen border-gray-300 rounded bg-white"
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => handleSelectStock(item.ticker)}
                        onClick={(e) => e.stopPropagation()}
                      />
                    </td>

                    <td className="py-1 px-2 border-b border-gray-300">{item.ticker}</td>
                    <td className="py-1 px-2 border-b border-gray-300">{item.name}</td>
                    <td className="py-1 px-2 border-b border-gray-300">{(item.annual_return * 100).toFixed(2)}%</td>
                    <td className="py-1 px-2 border-b border-gray-300">{(item.volatility * 100).toFixed(2)}%</td>
                    <td className="py-1 px-2 border-b border-gray-300">{item.esg_combined.toFixed(2)}</td>
                    <td className="py-1 px-2 border-b border-gray-300">{item.environment.toFixed(2)}</td>
                    <td className="py-1 px-2 border-b border-gray-300">{item.social.toFixed(2)}</td>
                    <td className="py-1 px-2 border-b border-gray-300">{item.governance.toFixed(2)}</td>
                    <td className="py-1 px-2 border-b border-gray-300 ">{item.compatibility.toFixed(0)}%</td>
                    <td className="py-1 px-2 border-b border-gray-300">{(item.weight * 100).toFixed(2)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <StockSearchModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        currentStocks={portfolioData}
        stocks={stockData}
        onAddStocks={handleAddStocks}
        onRemoveTickers={handleRemoveTickers}
        onClickStock={openTableauDashboard}
        selectedTickers={selectedTickers}
        setSelectedTickers={setSelectedTickers}
        handleSelectStock={handleSelectStock}
      />
    </div>
  );
};

export default RankingFormResults;
