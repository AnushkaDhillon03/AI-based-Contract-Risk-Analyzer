from collections import Counter


class SummaryGenerator:

    def generate_executive_summary(self, clauses):

        clause_types = set()
        risk_counter = Counter()

        high_risks = []
        medium_risks = []
        low_risks = []

        for clause in clauses:

            clause_types.add(
                clause["clause_type"]
            )

            for risk in clause.get("risks", []):

                level = risk["level"]

                risk_counter[level] += 1

                if level == "HIGH":
                    high_risks.append(
                        risk["risk"]
                    )

                elif level == "MEDIUM":
                    medium_risks.append(
                        risk["risk"]
                    )

                elif level == "LOW":
                    low_risks.append(
                        risk["risk"]
                    )

        # Overall Risk
        if risk_counter["HIGH"] > 0:
            overall_risk = "HIGH"

        elif risk_counter["MEDIUM"] > 0:
            overall_risk = "MEDIUM"

        else:
            overall_risk = "LOW"

        summary = []

        summary.append("EXECUTIVE SUMMARY\n")

        summary.append(
            f"The agreement contains "
            f"{len(clause_types)} clause categories: "
            f"{', '.join(sorted(clause_types))}.\n"
        )

        if high_risks:

            summary.append(
                "High-Risk Provisions:"
            )

            for risk in set(high_risks):

                summary.append(
                    f"• {risk}"
                )

            summary.append("")

        if medium_risks:

            summary.append(
                "Medium-Risk Provisions:"
            )

            for risk in set(medium_risks):

                summary.append(
                    f"• {risk}"
                )

            summary.append("")

        summary.append("Risk Distribution:")

        summary.append(
            f"• High Risks: {risk_counter['HIGH']}"
        )

        summary.append(
            f"• Medium Risks: {risk_counter['MEDIUM']}"
        )

        summary.append(
            f"• Low Risks: {risk_counter['LOW']}"
        )

        summary.append("")

        summary.append(
            f"Overall Contract Risk Level: "
            f"{overall_risk}"
        )

        return "\n".join(summary)
