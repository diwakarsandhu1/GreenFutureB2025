import React, { useEffect, useState } from "react";
import HoverPopup from "./HoverPopup";

const StockSearchModal = ({
  isOpen,
  onClose,
  stocks,
  currentStocks,
  onAddStocks,
  onRemoveTickers,
  onClickStock,
  selectedTickers,
  setSelectedTickers,
  handleSelectStock,
}) => {
  if (!isOpen) return null;

  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("all");

  useEffect(() => {
    setSearchTerm("");
    if (isOpen && selectedTickers.length > 0) {
      setSelectedCategory("selected");
    }
  }, [isOpen]);

  const categories = [
    { name: "All", key: "all" },
    { name: "Selected Stocks", key: "selected" },
    { name: "In Portfolio", key: "current_portfolio" },
    { name: "Not In Portfolio", key: "not_in_portfolio" },
    { name: "Top Overall ESG", key: "esg_combined" },
    { name: "Top Environment", key: "environment" },
    { name: "Top Governance", key: "governance" },
    { name: "Top Product Responsibility", key: "product_responsibility" },
    { name: "Top Social", key: "social" },
    { name: "Top Human Rights", key: "human_rights" },
  ];

  const filteredStocks = stocks.filter((stock) => {
    const matchesSearchTerm =
      stock.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      stock.ticker.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesSearchTerm;
  });

  const sortedStocks =
    selectedCategory === "all"
      ? filteredStocks
      : selectedCategory === "selected"
      ? filteredStocks.filter((stock) => selectedTickers.includes(stock.ticker))
      : selectedCategory === "current_portfolio"
      ? filteredStocks.filter((stock) => currentStocks.some((cs) => cs.ticker === stock.ticker))
      : selectedCategory === "not_in_portfolio"
      ? filteredStocks.filter((stock) => !currentStocks.some((cs) => cs.ticker === stock.ticker))
      : filteredStocks
          .filter((stock) => stock[selectedCategory] !== undefined)
          .sort((a, b) => (b[selectedCategory] || 0) - (a[selectedCategory] || 0))
          .slice(0, 25);

  const handleAddSelected = () => {
    const addStockNumber = selectedTickers.filter((ticker) => !currentStocks.some((cs) => cs.ticker === ticker)).length;
    var addStockMessage = `Are you sure you want to add the ${addStockNumber} selected stock(s) to your portfolio?`;
    if (addStockNumber === 0) {
      alert("No selected stocks are in your portfolio.");
      return;
    } else if (addStockNumber < selectedTickers.length) {
      addStockMessage = `Some of the selected stocks are already in your portfolio. Do you want to add the remaining ${addStockNumber} selected stock(s) to your portfolio?`;
    }
    if (!window.confirm(addStockMessage)) {
      return;
    }
    // handle adding list of tickers accounting for duplicates
    const stocksToAdd = stocks.filter(
      (stock) => selectedTickers.includes(stock.ticker) && !currentStocks.includes(stock.ticker)
    );
    onAddStocks(stocksToAdd);
    setSelectedTickers([]);
  };

  const handleRemoveSelected = () => {
    const removeStockNumber = selectedTickers.filter((ticker) =>
      currentStocks.some((cs) => cs.ticker === ticker)
    ).length;
    var removeStockMessage = `Are you sure you want to remove the ${removeStockNumber} selected stock(s) from your portfolio?`;
    if (removeStockNumber === 0) {
      alert("No selected stocks are in your portfolio.");
      return;
    } else if (removeStockNumber < selectedTickers.length) {
      removeStockMessage = `Some of the selected stocks are not in your portfolio. Do you want to remove the remaining ${removeStockNumber} selected stock(s) from your portfolio?`;
    }

    if (!window.confirm(removeStockMessage)) {
      return;
    }
    // handle removing list of selected tickers from portfolio
    onRemoveTickers(selectedTickers);
    setSelectedTickers([]);
  };

  return (
    // place modal at z 50 above rest of app and position on page
    <div className="fixed inset-0 flex items-center justify-center z-50">
      {/* click outside modal to close */}
      <div className="fixed inset-0 bg-black opacity-50" onClick={onClose}></div>

      {/* wrapper for modal content. p-6 padding applies to everything inside
      pt-2 limits top padding so top row is closer to top edge
       */}
      <div className="bg-white rounded-lg shadow-lg p-6 pt-2 z-50 max-w-3xl w-full">
        {/* top row of modal */}
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">Portfolio Edit</h2>
          <button onClick={onClose} className="text-4xl text-gray-500 hover:text-gray-700">
            &times;
          </button>
        </div>

        <div className="flex items-center mb-3">
          <input
            type="text"
            placeholder="Search by name or ticker"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded"
          />
          <button
            onClick={() => {
              setSearchTerm("");
            }}
            className="ml-2 px-4 py-2 bg-red-500 text-white rounded hover:opacity-75"
          >
            Clear
          </button>
        </div>
        <div className="flex items-center mb-3">
          <label className="mr-2">Filter by category:</label>
          <select
            value={selectedCategory}
            onChange={(e) => {
              setSelectedCategory(e.target.value);
              setSearchTerm("");
            }}
            className="p-2 border border-gray-300 rounded"
          >
            {categories.map((category) => (
              <option key={category.key} value={category.key}>
                {category.name}
              </option>
            ))}
          </select>
          <button
            onClick={() => setSelectedCategory("all")}
            className="ml-2 text-xs text-gray-500 hover:underline  hover:opacity-75"
          >
            Clear Filter
          </button>
        </div>
        <div className="max-h-96 min-h-96 overflow-y-auto">
          <table className="min-w-full bg-white mb-2 text-sm">
            <thead className="sticky top-0 bg-white z-10">
              <tr>
                <th className="pl-2 border-b-2 border-gray-300 text-left leading-4 text-gray-600 tracking-wider">
                  {" "}
                  <input
                    className="cursor-pointer h-5 w-5 text-gfwmDarkGreen focus:ring-gfwmLightGreen border-gray-300 rounded bg-white"
                    type="checkbox"
                    checked={selectedTickers.length === sortedStocks.length}
                    onChange={() => {
                      if (selectedTickers.length === sortedStocks.length) {
                        setSelectedTickers([]);
                      } // if all stocked in the current filter is in the selectedStocks, then clear the selectedStocks
                      else {
                        setSelectedTickers((prevSelected) => [
                          ...prevSelected,
                          ...sortedStocks
                            .filter((stock) => !prevSelected.includes(stock.ticker))
                            .map((stock) => stock.ticker),
                        ]);
                      }
                    }}
                  ></input>{" "}
                </th>
                <th className="py-0 px-0 border-b-2 border-gray-300 text-left leading-4 text-gray-600"></th>
                <th className="py-1 px-2 border-b-2 border-gray-300 text-left leading-4 text-gray-600 tracking-wider">
                  Symbol
                </th>
                <th className="py-1 px-2 border-b-2 border-gray-300 text-left leading-4 text-gray-600 tracking-wider">
                  Name
                </th>
                <th className="py-1 px-2 border-b-2 border-gray-300 text-left leading-4 text-gray-600 tracking-wider">
                  Combined ESG
                </th>
                <th className="py-1 px-2 border-b-2 border-gray-300 text-left leading-4 text-gray-600 tracking-wider">
                  Compatibility
                </th>
                <th className="py-1 px-2 border-b-2 border-gray-300 text-left leading-4 text-gray-600 tracking-wider">
                  Status
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedStocks.map((stock, index) => {
                const isInPortfolio = currentStocks.find((pI) => pI.ticker === stock.ticker);
                const isSelected = selectedTickers.includes(stock.ticker);
                return (
                  <tr key={stock.ticker} className="hover:bg-gray-100  ">
                    <td
                      className="border-b text-right pl-2 border-gray-300 cursor-pointer "
                      onClick={() => handleSelectStock(stock.ticker)}
                    >
                      <input
                        className="cursor-pointer h-5 w-5 text-gfwmDarkGreen focus:ring-gfwmLightGreen border-gray-300 rounded bg-white"
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => handleSelectStock(stock.ticker)}
                        onClick={(e) => e.stopPropagation()}
                      />
                    </td>
                    <td
                      className="py-0 px-0 border-b border-gray-300 text-gray-400 text-right cursor-pointer"
                      onClick={() => handleSelectStock(stock.ticker)}
                    >
                      {index + 1}
                    </td>
                    <td
                      className="py-1 px-2 border-b border-gray-300 cursor-pointer"
                      onClick={() => onClickStock(stock.ticker)}
                    >
                      {stock.ticker}
                    </td>

                    <td
                      className="py-1 px-2 border-b border-gray-300 cursor-pointer"
                      onClick={() => onClickStock(stock.ticker)}
                    >
                      {stock.name}
                    </td>
                    <td
                      className="py-1 px-2 border-b border-gray-300 cursor-pointer"
                      onClick={() => onClickStock(stock.ticker)}
                    >
                      {stock.esg_combined.toFixed(2)}
                    </td>
                    <td
                      className="py-1 px-2 border-b border-gray-300 cursor-pointer"
                      onClick={() => onClickStock(stock.ticker)}
                    >
                      {stock.compatibility.toFixed(0)}%
                    </td>
                    <td
                      className="py-1 px-2 border-b border-gray-300 cursor-pointer"
                      onClick={() => onClickStock(stock.ticker)}
                    >
                      {isInPortfolio ? (
                        <span className="text-gfwmDarkGreen">In Portfolio</span>
                      ) : (
                        <span className="text-gray-500 text-xs">Not In Portfolio</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="flex justify-end space-x-4 mt-4">
          <div className="flex flex-col text-gray-700">
            <HoverPopup
              content={
                <div>
                  <p className="text-sm text-gray-700">Selected stocks:</p>
                  <ul className="list-disc list-inside">
                    {selectedTickers.map((ticker, index) => (
                      <li key={index} className="text-sm text-gray-700">
                        {ticker}
                      </li>
                    ))}
                  </ul>
                </div>
              }
            >
              <span
                className="cursor-pointer hover:underline"
                onClick={() => {
                  setSelectedCategory("selected");
                  setSearchTerm("");
                }}
              >
                {selectedTickers.length} stock(s) selected
              </span>
            </HoverPopup>
            <button
              onClick={() => setSelectedTickers([])}
              className=" text-xs hover:underline text-gray-500 hover:opacity-75"
            >
              Clear Selected Stocks
            </button>
          </div>
          <HoverPopup
            content={
              <div>
                <p className="text-sm text-gray-700">
                  {selectedTickers.filter((ticker) => currentStocks.some((cs) => cs.ticker === ticker)).length} stock(s)
                  to be removed:
                </p>
                <ul className="list-disc list-inside">
                  {selectedTickers
                    .filter((ticker) => currentStocks.some((cs) => cs.ticker === ticker))
                    .map((ticker, index) => (
                      <li key={index} className="list-none text-sm text-gray-700">
                        {ticker}
                      </li>
                    ))}
                </ul>
              </div>
            }
          >
            <button onClick={handleRemoveSelected} className="hover:opacity-75 px-4 py-2 bg-red-500 text-white rounded">
              Remove Selected
            </button>
          </HoverPopup>

          <HoverPopup
            content={
              <div>
                <p className="text-sm text-gray-700">
                  {selectedTickers.filter((ticker) => !currentStocks.some((cs) => cs.ticker === ticker)).length}{" "}
                  stock(s) to be added:
                </p>
                <ul className="list-disc list-inside">
                  {selectedTickers
                    .filter((ticker) => !currentStocks.some((cs) => cs.ticker === ticker))
                    .map((ticker, index) => (
                      <li key={index} className="list-none text-sm text-gray-700">
                        {ticker}
                      </li>
                    ))}
                </ul>
              </div>
            }
          >
            <button
              onClick={handleAddSelected}
              className="hover:opacity-75 px-4 py-2 bg-gfwmDarkGreen text-white rounded"
            >
              Add Selected
            </button>
          </HoverPopup>
        </div>
      </div>
    </div>
  );
};

export default StockSearchModal;
