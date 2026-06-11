"""
Clause Extraction Module
"""

import json
import re
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClauseExtractor:
    """
    Extract and classify clauses from legal contracts.
    """

    def __init__(self, clause_json_path: str, threshold: int = 1):
        self.threshold = threshold

        try:
            with open(clause_json_path, "r", encoding="utf-8") as f:
                self.clause_dict = json.load(f)

        except Exception as e:
            logger.error(f"Unable to load JSON file: {e}")
            raise

    def extract_clauses(self, text: str) -> List[Dict]:
        """
        Extract numbered clauses.
        """

        clauses = []

        try:

            pattern = re.compile(
                r'(?=^\s*\d+(?:\.\d+)*\s+)',
                re.MULTILINE
            )

            positions = [m.start() for m in pattern.finditer(text)]

            if not positions:
                return []

            positions.append(len(text))

            for i in range(len(positions) - 1):

                block = text[
                    positions[i]:positions[i + 1]
                ].strip()

                lines = block.splitlines()

                if not lines:
                    continue

                heading = lines[0].strip()

                number_match = re.match(
                    r'^(\d+(?:\.\d+)*)',
                    heading
                )

                clause_number = (
                    number_match.group(1)
                    if number_match
                    else ""
                )

                title_match = re.match(
                    r'^\d+(?:\.\d+)*\s+(.*)',
                    heading
                )

                clause_title = (
                    title_match.group(1)
                    if title_match
                    else heading
                )

                clauses.append(
                    {
                        "clause_number": clause_number,
                        "clause_title": clause_title,
                        "text": block
                    }
                )

            return clauses

        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return []

    def classify_clause(self, clause_text: str) -> str:
        """
        Classify clause using keywords.
        """

        clause_text = clause_text.lower()

        scores = {}

        for category, keywords in self.clause_dict.items():

            score = 0

            for keyword in keywords:
                score += clause_text.count(
                    keyword.lower()
                )

            scores[category] = score

        best_category = max(
            scores,
            key=scores.get,
            default="Unknown"
        )

        if scores[best_category] < self.threshold:
            return "Unknown"

        return best_category

    def process(self, text: str) -> List[Dict]:
        """
        Complete pipeline.
        """

        clauses = self.extract_clauses(text)

        for clause in clauses:

            clause["clause_type"] = (
                self.classify_clause(
                    clause["text"]
                )
            )

        return clauses


# -----------------------------------
# TESTING
# -----------------------------------

if __name__ == "__main__":

    sample_text = """
1 TERMINATION

Either party may terminate this agreement with notice.

2 CONFIDENTIALITY

All confidential information must remain protected.

3 PAYMENT

Payment shall be made within 30 days.
"""

    extractor = ClauseExtractor(
        "clauses.json"
    )

    result = extractor.process(
        sample_text
    )

    print("\nEXTRACTED CLAUSES:\n")

    for clause in result:
        print(clause)