"""
app.py — Contract Analysis API (FastAPI version)

Run with:
    python3 app.py
or:
    uvicorn app:app --reload --port 8080

Endpoints:
  GET  /                → serves frontend (index.html)
  GET  /version         → API version info
  GET  /health           → status check
  POST /analyze          → full pipeline (preprocess → extract → risk detect → summary)
  POST /risks/filter     → filter already-analyzed risks by level + top-N

Interactive docs auto-generated at:
  http://127.0.0.1:8080/docs
"""

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from text_preprocessing import TextPreprocessor
from clause_extraction import ClauseExtractor
from risk import RiskDetector
from SummaryGenerator import SummaryGenerator


# --------------------------------------------------
# App setup
# --------------------------------------------------
app = FastAPI(
    title="Contract Risk Analyzer API",
    description="NLP pipeline for legal contract clause extraction, risk detection, and summarization.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Boot all pipeline modules once at startup
# --------------------------------------------------
preprocessor      = TextPreprocessor()
clause_extractor  = ClauseExtractor("clauses.json")
risk_detector     = RiskDetector("important_risk_patterns.json")
summary_generator = SummaryGenerator()

VALID_LEVELS = {"HIGH", "MEDIUM", "LOW"}


# --------------------------------------------------
# Request / response schemas
# --------------------------------------------------
class AnalyzeRequest(BaseModel):
    text: str
    risk_level: Optional[str] = None   # "HIGH" | "MEDIUM" | "LOW"
    top_n: Optional[int] = None


class FilterRequest(BaseModel):
    risks: List[dict]
    risk_level: Optional[str] = "HIGH"
    top_n: Optional[int] = 5


# --------------------------------------------------
# Helper: run the full pipeline
# --------------------------------------------------
def run_pipeline(raw_text: str) -> dict:
    cleaned = preprocessor.preprocess(raw_text)

    clauses = clause_extractor.process(cleaned)

    for clause in clauses:
        clause["risks"] = risk_detector.detect_risks(clause["text"])

    summary = summary_generator.generate_executive_summary(clauses)

    return {
        "cleaned_text": cleaned,
        "clauses": clauses,
        "summary": summary,
    }


# --------------------------------------------------
# Route: serve frontend
# --------------------------------------------------
@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


# --------------------------------------------------
# Route: version info
# --------------------------------------------------
@app.get("/version")
def version():
    return {
        "name": "Contract Risk Analyzer API",
        "version": "1.0.0",
        "framework": "FastAPI",
    }


# --------------------------------------------------
# Route: health check
# --------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "message": "Contract Analysis API is running"}


# --------------------------------------------------
# Route: full analysis
# --------------------------------------------------
@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    raw_text = req.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="No contract text provided.")

    risk_level = req.risk_level.upper() if req.risk_level else None
    top_n = req.top_n

    if risk_level and risk_level not in VALID_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid risk_level. Choose from {VALID_LEVELS}",
        )

    if top_n is not None and top_n < 1:
        raise HTTPException(status_code=400, detail="top_n must be >= 1")

    try:
        result = run_pipeline(raw_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

    # Flatten risks across all clauses
    all_risks = []
    for clause in result["clauses"]:
        for risk in clause.get("risks", []):
            all_risks.append({
                "clause_number": clause["clause_number"],
                "clause_title": clause["clause_title"],
                "clause_type": clause["clause_type"],
                **risk,
            })

    # Default: nothing specified → top 5 HIGH
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

    result["filtered_risks"] = filtered
    result["filter_applied"] = filter_applied

    return result


# --------------------------------------------------
# Route: filter risks from a previous analysis
# --------------------------------------------------
@app.post("/risks/filter")
def filter_risks(req: FilterRequest):
    risk_level = (req.risk_level or "HIGH").upper()
    top_n = req.top_n or 5

    if risk_level not in VALID_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid risk_level. Choose from {VALID_LEVELS}",
        )

    if top_n < 1:
        raise HTTPException(status_code=400, detail="top_n must be >= 1")

    filtered = [r for r in req.risks if r.get("level") == risk_level][:top_n]

    return {
        "risk_level": risk_level,
        "top_n": top_n,
        "count": len(filtered),
        "filtered_risks": filtered,
    }


# --------------------------------------------------
# Run directly with: python3 app.py
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    print("\n Contract Analysis API (FastAPI)")
    print(" ─────────────────────────────────────")
    print("  GET  /              → frontend")
    print("  GET  /version       → version info")
    print("  GET  /health        → status check")
    print("  POST /analyze       → full pipeline")
    print("  POST /risks/filter  → filter risks")
    print("  GET  /docs          → interactive API docs")
    print(" ─────────────────────────────────────\n")

    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)
