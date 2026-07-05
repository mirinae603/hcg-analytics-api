import os
import uvicorn
from app.main import app

if __name__ == "__main__":
    # Cloud hosts (Render / Railway / Fly / Cloud Run / HF Spaces) inject $PORT.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
