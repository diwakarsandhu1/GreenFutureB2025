import React from "react";
import RankingForm from "./RankingForm.jsx";

const LandingPage = () => {
  return (
    <div className="landing-page">
      <header className="bg-gfwmDarkGreen text-white p-2 text-left">
        <h2 className="text-2xl font-bold">Green Future Wealth Management ESG Questionnaire</h2>
      </header>
      <div className="flex-1 p-5 text-left">
        <RankingForm />
      </div>

      <footer className="bg-gray-600 text-white p-4 flex justify-center items-center">
        <div className="space-y-2">

          <p className="w-[1200px] text-center">
            The content is developed from sources believed to be providing accurate information. The information in this
            material is not intended as tax or legal advice. Please consult legal or tax professionals for specific
            information regarding your individual situation. Some of this material was developed and produced by FMG
            Suite to provide information on a topic that may be of interest. FMG Suite is not affiliated with the named
            representative, broker - dealer, state - or SEC - registered investment advisory firm. The opinions
            expressed and material provided are for general information, and should not be considered a solicitation for
            the purchase or sale of any security.
          </p>

          <p className="w-[1200px] text-center">
            Securities offered through Registered Representatives of Cambridge Investment Research, Inc., a broker
            dealer, member FINRA/SIPC. Advisory services offered through Cambridge Investment Research Advisors, Inc, a
            Registered Investment Adviser. Green Future Wealth Management and Cambridge are not affiliated.
          </p>

        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
