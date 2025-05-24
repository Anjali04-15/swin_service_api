# Swin Service API

This is a FastAPI-based microservice that hosts a Swin Transformer model for fruit freshness classification.

---

## Setup & Run Locally

1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd swin_service
2. Create and activate a Python virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. pip install -r requirements.txt
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8080
5. Open your browser and go to:
   ```bash
   http://localhost:8080/docs
Use the interactive Swagger UI to test the /predict endpoint by uploading an image.
