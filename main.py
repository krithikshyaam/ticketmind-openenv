"""Entry point – run with: python main.py  or  uvicorn main:app"""
import uvicorn
from app.main import app  # noqa: F401  re-export for uvicorn
 
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
    )