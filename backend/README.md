# AgroDashRD Backend API

This is the Python/FastAPI backend for AgroDashRD.

## Setup

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate     # Windows
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Variables:**
    Create a `.env` file in the `backend/` directory:
    ```env
    DATABASE_URL=sqlite:///./agrodash.db  # Use postgresql://... in production
    SECRET_KEY=your_secret_key_here
    ```

4.  **Run the server:**
    ```bash
    uvicorn backend.app.main:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.

## API Documentation
Once running, go to `http://127.0.0.1:8000/docs` to see the interactive Swagger UI.

## Database
The system uses SQLite by default for development. To use PostgreSQL (recommended for production):
1.  Install PostgreSQL locally or use a cloud service (Hostinger/Render).
2.  Update `DATABASE_URL` in `.env`.
3.  The tables will be created automatically on startup (for dev), or use `schema.sql` for manual setup.
