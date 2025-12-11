import React from "react";

const ComparisonTable = ({ data }) => {
  return (
    <div>
      <table className="w-full border-collapse table-auto text-left">
        <thead>
          <tr>
            <th className="px-4 py-2 font-semibold text-gray-700">Field</th>
            <th className="px-4 py-2 font-semibold text-gray-700">Portfolio</th>
            <th className="px-4 py-2 font-semibold text-gray-700">Composite Benchmark</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row, index) => (
            <tr key={index} className="border-b">
              <td className="w-[50%] px-2">{row.field}</td>
              <td className="w-[30%] px-4">
                <span className="text-green-700 text-xl font-bold">
                  {row.portfolio}
                </span>
              </td>
              <td className="w-[20%] px-4">
                <span className="text-light-green-700 text-xl font-bold">
                  {row.sp500}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ComparisonTable;
