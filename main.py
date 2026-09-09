"""
FastAPI Server for Intelligent Progress Tracking (IPT) Model.
Ready for deployment on Render, Railway, or Docker.
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure src can be resolved
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.matcher import IPTMatcher

# Global matcher instance
matcher: Optional[IPTMatcher] = None

# Optional secret key for securing the API
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "").strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: loads the ML model into memory once on startup."""
    global matcher
    models_dir = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
    print(f"[*] Starting IPT API service...")
    print(f"[*] Loading model artifacts from: {models_dir}")
    try:
        matcher = IPTMatcher(models_dir=models_dir)
        print("[+] Model and FAISS index loaded successfully into RAM.")
    except Exception as e:
        print(f"[-] Failed to load model artifacts: {e}", file=sys.stderr)
        raise e
    yield
    print("[*] Shutting down IPT API service...")


app = FastAPI(
    title="Intelligent Progress Tracking (IPT) Model API",
    description="High-precision semantic activity matching and progress tracking API.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for requests from Vercel or localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace with your specific Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_api_key(authorization: Optional[str] = None, x_api_key: Optional[str] = None):
    """Optional authentication check if API_SECRET_KEY environment variable is defined."""
    if not API_SECRET_KEY:
        return True  # No auth configured, open access

    # Check Authorization header (Bearer <token>)
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1] == API_SECRET_KEY:
            return True

    # Check X-API-KEY header
    if x_api_key and x_api_key == API_SECRET_KEY:
        return True

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized: Invalid or missing API key."
    )


# Request & Response Schemas
class MatchRequest(BaseModel):
    query_text: str = Field(
        ...,
        description="Daily site report or progress update text",
        example="RCC footing reinforcement completed at Block A"
    )
    project_id: Optional[str] = Field(
        None,
        description="Optional Project ID filter (e.g., P00010)",
        example="P00010"
    )
    top_k: int = Field(
        5,
        ge=1,
        le=50,
        description="Number of candidate matches to return"
    )


class BatchMatchRequest(BaseModel):
    items: List[MatchRequest] = Field(..., description="List of match queries to process")


# Endpoints
@app.get("/")
def root():
    return {
        "service": "Intelligent Progress Tracking (IPT) API",
        "status": "online",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Health check endpoint used by Render / Vercel uptime monitors."""
    if matcher is None:
        raise HTTPException(status_code=503, detail="Model is initializing")
    return {
        "status": "healthy",
        "model_loaded": True,
        "indexed_activities": matcher.index.ntotal if matcher and hasattr(matcher, "index") else 0
    }


@app.post("/api/v1/match", tags=["Matching"])
def match_activity(
    payload: MatchRequest,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY"),
):
    """
    Match a natural language site report to scheduled project activities.
    Returns the top match with progress tracking metrics and top-k candidates.
    """
    verify_api_key(authorization, x_api_key)

    if matcher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still initializing. Please retry in a few seconds."
        )

    try:
        result = matcher.match(
            query_text=payload.query_text,
            top_k=payload.top_k,
            project_id=payload.project_id
        )
        return {
            "success": True,
            "data": result
        }
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.post("/api/v1/batch-match", tags=["Matching"])
def batch_match(
    payload: BatchMatchRequest,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY"),
):
    """Process multiple site reports in a single batch call."""
    verify_api_key(authorization, x_api_key)

    if matcher is None:
        raise HTTPException(status_code=503, detail="Model is initializing.")

    results = []
    for item in payload.items:
        try:
            res = matcher.match(
                query_text=item.query_text,
                top_k=item.top_k,
                project_id=item.project_id
            )
            results.append({"success": True, "query": item.query_text, "data": res})
        except Exception as err:
            results.append({"success": False, "query": item.query_text, "error": str(err)})

    return {"success": True, "count": len(results), "results": results}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
