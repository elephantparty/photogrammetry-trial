#!/usr/bin/env python3
"""
Run the Photogrammetry web application.
"""
import uvicorn
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

if __name__ == "__main__":
    print("\n🎨 Photogrammetry Web App")
    print("=" * 40)
    print("Starting server at http://localhost:8000")
    print("Press Ctrl+C to stop\n")

    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
