# main.py

import json

from text_preprocessing import TextPreprocessor
from clause_extraction import ClauseExtractor
from risk import RiskDetector
from SummaryGenerator import SummaryGenerator


class ContractAnalyzer:

    def __init__(
        self,
        clause_json_path="clauses_.json",
        risk_json_path="important_risk_patterns.json"
    ):
        """
        Initialize all modules.
        """

        self.preprocessor = TextPreprocessor()

        self.clause_extractor = ClauseExtractor(
            clause_json_path
        )

        self.risk_detector = RiskDetector(
            risk_json_path
        )

        self.summary_generator = SummaryGenerator()

    def analyze_contract(self, contract_file_path):
        """
        Complete contract analysis pipeline:

        1. Read contract
        2. Preprocess text
        3. Extract clauses
        4. Detect risks clause-wise
        5. Generate summary
        """

        # ==================================
        # STEP 1: READ CONTRACT
        # ==================================
        try:
            with open(
                contract_file_path,
                "r",
                encoding="utf-8"
            ) as file:
                raw_text = file.read()

        except Exception as e:
            raise Exception(
                f"Failed to read contract: {e}"
            )

        # ==================================
        # STEP 2: PREPROCESS TEXT
        # ==================================
        cleaned_text = self.preprocessor.preprocess(
            raw_text
        )

        # ==================================
        # STEP 3: EXTRACT & CLASSIFY CLAUSES
        # ==================================
        clauses = self.clause_extractor.process(
            cleaned_text
        )

        # ==================================
        # STEP 4: RISK ANALYSIS
        # ==================================
        for clause in clauses:

            detected_risks = (
                self.risk_detector.detect_risks(
                    clause["text"]
                )
            )

            clause["risks"] = detected_risks

        # ==================================
        # STEP 5: GENERATE SUMMARY
        # ==================================
        summary = (
            self.summary_generator
            .generate_executive_summary(
                clauses
            )
        )

        # ==================================
        # FINAL OUTPUT
        # ==================================
        return {
            "cleaned_text": cleaned_text,
            "clauses": clauses,
            "summary": summary
        }


# ==================================================
# RUN APPLICATION
# ==================================================

if __name__ == "__main__":

    CONTRACT_FILE = "contract.txt"

    analyzer = ContractAnalyzer(
        clause_json_path="clauses_.json",
        risk_json_path="important_risk_patterns.json"
    )

    result = analyzer.analyze_contract(
        CONTRACT_FILE
    )

    print("\n")
    print("=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    print(result["summary"])

    print("\n")
    print("=" * 80)
    print("CLAUSE ANALYSIS")
    print("=" * 80)

    for clause in result["clauses"]:

        print("\n-----------------------------------")
        print(
            f"Clause Number : "
            f"{clause['clause_number']}"
        )

        print(
            f"Clause Title  : "
            f"{clause['clause_title']}"
        )

        print(
            f"Clause Type   : "
            f"{clause['clause_type']}"
        )

        print("\nClause Text:")
        print(clause["text"])

        print("\nRisks:")

        if clause["risks"]:

            for risk in clause["risks"]:

                print(
                    f"- [{risk['level']}] "
                    f"{risk['risk']}"
                )

        else:
            print("No risks detected.")

    print("\n")
    print("=" * 80)
    print("JSON OUTPUT")
    print("=" * 80)

    print(
        json.dumps(
            result,
            indent=4
        )
    )
