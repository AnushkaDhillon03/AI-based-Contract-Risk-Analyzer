"""
app.py — Contract Analysis API

Endpoints:
  POST /analyze        → full pipeline (preprocess → extract → risk detect → summary)
  POST /risks/filter   → filter already-analyzed risks by level + top-N
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from text_preprocessing import TextPreprocessor
from clause_extraction import ClauseExtractor
from risk import RiskDetector
from SummaryGenerator import SummaryGenerator

app = Flask(__name__)
CORS(app)  # allow browser fetch from the frontend HTML

# --------------------------------------------------
# Boot all modules once at startup
# --------------------------------------------------
preprocessor      = TextPreprocessor()
clause_extractor  = ClauseExtractor("clauses.json")
risk_detector     = RiskDetector("important_risk_patterns.json")
summary_generator = SummaryGenerator()


# --------------------------------------------------
# Helper
# --------------------------------------------------
def run_pipeline(raw_text: str):
    """Run the full 4-step pipeline and return structured result."""

    # Step 1 – clean text
    cleaned = preprocessor.preprocess(raw_text)

    # Step 2 – extract + classify clauses
    clauses = clause_extractor.process(cleaned)

    # Step 3 – detect risks per clause
    for clause in clauses:
        clause["risks"] = risk_detector.detect_risks(clause["text"])

    # Step 4 – executive summary
    summary = summary_generator.generate_executive_summary(clauses)

    return {
        "cleaned_text": cleaned,
        "clauses":      clauses,
        "summary":      summary,
    }


# --------------------------------------------------
# Route 1: Full analysis
# --------------------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Body (JSON):
        { "text": "<raw contract text>" }

    Optional filters applied AFTER pipeline:
        { "risk_level": "HIGH" | "MEDIUM" | "LOW",   (default: all)
          "top_n": 5 }                                 (default: no limit)

    Returns full pipeline output + optionally filtered risk list.
    """
    body = request.get_json(force=True, silent=True) or {}

    raw_text = body.get("text", "").strip()
    if not raw_text:
        return jsonify({"error": "No contract text provided."}), 400

    risk_level = (body.get("risk_level") or "").upper() or None
    top_n      = body.get("top_n", None)

    # Validate
    valid_levels = {"HIGH", "MEDIUM", "LOW"}
    if risk_level and risk_level not in valid_levels:
        return jsonify({"error": f"Invalid risk_level. Choose from {valid_levels}"}), 400

    try:
        top_n = int(top_n) if top_n is not None else None
        if top_n is not None and top_n < 1:
            return jsonify({"error": "top_n must be >= 1"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "top_n must be an integer"}), 400

    try:
        result = run_pipeline(raw_text)
    except Exception as e:
        return jsonify({"error": f"Pipeline failed: {e}"}), 500

    # --------------------------------------------------
    # Optional: build a flat filtered risk list
    # --------------------------------------------------
    all_risks = []
    for clause in result["clauses"]:
        for risk in clause.get("risks", []):
            all_risks.append({
                "clause_number": clause["clause_number"],
                "clause_title":  clause["clause_title"],
                "clause_type":   clause["clause_type"],
                **risk,
            })

    # Default behaviour: if nothing specified → top 5 HIGH
    if risk_level is None and top_n is None:
        filtered = [r for r in all_risks if r["level"] == "HIGH"][:5]
        filter_applied = "default (top 5 HIGH)"
    else:
        filtered = all_risks
        if risk_level:
            filtered = [r for r in filtered if r["level"] == risk_level]
        if top_n:
            filtered = filtered[:top_n]
        filter_applied = f"level={risk_level or 'ALL'}, top_n={top_n or 'ALL'}"

    result["filtered_risks"]  = filtered
    result["filter_applied"]  = filter_applied

    return jsonify(result), 200


# --------------------------------------------------
# Route 2: Filter risks from a previous analysis
# --------------------------------------------------
@app.route("/risks/filter", methods=["POST"])
def filter_risks():
    """
    Re-filter risks from an already-analyzed result without re-running the pipeline.

    Body (JSON):
        {
          "risks":       [ ...list from /analyze... ],
          "risk_level":  "HIGH" | "MEDIUM" | "LOW",   (optional, default: HIGH)
          "top_n":       5                             (optional, default: 5)
        }
    """
    body = request.get_json(force=True, silent=True) or {}

    risks      = body.get("risks", [])
    risk_level = (body.get("risk_level") or "HIGH").upper()
    top_n      = body.get("top_n", 5)

    valid_levels = {"HIGH", "MEDIUM", "LOW"}
    if risk_level not in valid_levels:
        return jsonify({"error": f"Invalid risk_level. Choose from {valid_levels}"}), 400

    try:
        top_n = int(top_n)
        if top_n < 1:
            return jsonify({"error": "top_n must be >= 1"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "top_n must be an integer"}), 400

    filtered = [r for r in risks if r.get("level") == risk_level][:top_n]

    return jsonify({
        "risk_level":    risk_level,
        "top_n":         top_n,
        "count":         len(filtered),
        "filtered_risks": filtered,
    }), 200


# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Contract Analysis API is running"}), 200

from flask import send_from_directory

@app.route("/")
def serve_frontend():
    return send_from_directory(".", "index.html")
# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    print("\n Contract Analysis API")
    print(" ─────────────────────────────────────")
    print("  POST /analyze       → full pipeline")
    print("  POST /risks/filter  → filter risks")
    print("  GET  /health        → status check")
    print(" ─────────────────────────────────────\n")
    app.run(debug=True, port=8080)
