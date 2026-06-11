import json
import logging
import re
from rapidfuzz import fuzz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskDetector:
    """
    Rule-based + fuzzy risk detector.
    """

    def __init__(self, risk_json_path, fuzzy_threshold=85):
        self.threshold = fuzzy_threshold

        with open(risk_json_path, "r", encoding="utf-8") as f:
            self.rules = json.load(f)

    def _extract_snippet(self, text, phrase, window=120):

        idx = text.lower().find(phrase.lower())

        if idx == -1:
            return ""

        start = max(0, idx - window)
        end = min(len(text), idx + len(phrase) + window)

        return text[start:end]

    def detect_risks(self, text):

        findings = []
        seen = set()

        for rule in self.rules:

            pattern = rule["pattern"]
            risk = rule["risk"]
            level = rule["level"]

            # Exact Match
            if pattern.lower() in text.lower():

                key = (pattern, risk)

                if key not in seen:

                    findings.append(
                        {
                            "pattern": pattern,
                            "risk": risk,
                            "level": level,
                            "match_type": "exact",
                            "matched_text": self._extract_snippet(
                                text,
                                pattern
                            ),
                        }
                    )

                    seen.add(key)

                continue

            # Fuzzy Match
            sentences = re.split(r"[.\n]", text)

            for sentence in sentences:

                score = fuzz.partial_ratio(
                    pattern.lower(),
                    sentence.lower()
                )

                if score >= self.threshold:

                    key = (pattern, risk)

                    if key not in seen:

                        findings.append(
                            {
                                "pattern": pattern,
                                "risk": risk,
                                "level": level,
                                "match_type": "fuzzy",
                                "matched_text": sentence.strip(),
                            }
                        )

                        seen.add(key)

                    break

        return findings


# ==========================
# TEST CODE
# ==========================

if __name__ == "__main__":

    contract_text = """
    Either party may terminate this agreement without notice.
    The vendor shall have unlimited liability for damages.
    The company may act at its sole discretion.
    """

    detector = RiskDetector(
        "important_risk_patterns.json"
    )

    risks = detector.detect_risks(contract_text)

    print("\nDetected Risks:\n")

    for risk in risks:
        print(risk)