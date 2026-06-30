
"""
clause_extraction.py

Clause Extraction and Classification Module
"""

import json
import logging
import re
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClauseExtractor:
    """
    Extracts and classifies clauses from legal contracts.
    """

    def __init__(
        self,
        clause_json_path: str,
        threshold: int = 1
    ):
        """
        Load clause categories from JSON.
        """

        self.threshold = threshold

        try:
            with open(
                clause_json_path,
                "r",
                encoding="utf-8"
            ) as file:

                self.clause_dictionary = json.load(file)

        except Exception as e:

            logger.error(
                f"Unable to load JSON file: {e}"
            )

            raise

    def extract_clauses(
        self,
        text: str
    ) -> List[Dict]:
        """
        Extract clauses using common legal numbering patterns.
        """

        try:

            clause_pattern = re.compile(
                r"""
                (?=^(
                    \d+(?:\.\d+)*\.?\s+.*      |
                    section\s+\d+.*            |
                    article\s+[ivxlcdm]+.*     |
                    [A-Z]\.\s+.*               |
                    \([a-z]\)\s+.*
                ))
                """,
                re.IGNORECASE
                | re.MULTILINE
                | re.VERBOSE
            )

            matches = list(
                clause_pattern.finditer(text)
            )

            clauses = []

            if not matches:
                return clauses

            for i in range(len(matches)):

                start = matches[i].start()

                if i + 1 < len(matches):
                    end = matches[i + 1].start()
                else:
                    end = len(text)

                block = text[start:end].strip()

                lines = block.splitlines()

                if not lines:
                    continue

                heading = lines[0].strip()

                clause_number = ""
                clause_title = heading

                number_match = re.match(
                    r'^(\d+(?:\.\d+)*\.?)',
                    heading
                )

                section_match = re.match(
                    r'^(section\s+\d+)',
                    heading,
                    re.IGNORECASE
                )

                article_match = re.match(
                    r'^(article\s+[ivxlcdm]+)',
                    heading,
                    re.IGNORECASE
                )

                alpha_match = re.match(
                    r'^([A-Z]\.)',
                    heading
                )

                bracket_match = re.match(
                    r'^(\([a-z]\))',
                    heading
                )

                if number_match:

                    clause_number = (
                        number_match.group(1)
                    )

                    clause_title = heading[
                        len(clause_number):
                    ].strip()

                elif section_match:

                    clause_number = (
                        section_match.group(1)
                    )

                    clause_title = heading

                elif article_match:

                    clause_number = (
                        article_match.group(1)
                    )

                    clause_title = heading

                elif alpha_match:

                    clause_number = (
                        alpha_match.group(1)
                    )

                    clause_title = heading[
                        len(clause_number):
                    ].strip()

                elif bracket_match:

                    clause_number = (
                        bracket_match.group(1)
                    )

                    clause_title = heading[
                        len(clause_number):
                    ].strip()

                clauses.append(
                    {
                        "clause_number":
                            clause_number,
                        "clause_title":
                            clause_title,
                        "text":
                            block
                    }
                )

            return clauses

        except Exception as e:

            logger.error(
                f"Clause extraction failed: {e}"
            )

            return []

    def classify_clause(
        self,
        clause_text: str
    ) -> str:
        """
        Classify clause using keyword scoring.
        """

        try:

            clause_text = clause_text.lower()

            scores = {}

            for (
                category,
                keywords
            ) in self.clause_dictionary.items():

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

            if (
                scores[best_category]
                < self.threshold
            ):
                return "Unknown"

            return best_category

        except Exception as e:

            logger.error(
                f"Classification failed: {e}"
            )

            return "Unknown"

    def process(
        self,
        text: str
    ) -> List[Dict]:
        """
        Full pipeline:
        Extract + Classify
        """

        clauses = self.extract_clauses(
            text
        )

        for clause in clauses:

            clause["clause_type"] = (
                self.classify_clause(
                    clause["text"]
                )
            )

        return clauses


# --------------------------------------------------
# TESTING
# --------------------------------------------------

if __name__ == "__main__":

    sample_text = """
1. TERMINATION

Either party may terminate this agreement.

1.1 NOTICE

Written notice shall be provided.

SECTION 2

Confidential information shall remain protected.

ARTICLE III

Payment terms are listed below.

A. LIABILITY

The vendor shall be liable.

(a) DAMAGES

Damages may be recovered.
"""

    extractor = ClauseExtractor(
        "clauses.json"
    )

    result = extractor.process(
        sample_text
    )

    import json

    print(
        json.dumps(
            result,
            indent=4
        )
    )

