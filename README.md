# Activities Dashboard – v3.1 (No secrets, local SQLite only)

This build removes Streamlit secrets entirely and **always uses local SQLite** (a file-based SQL database) stored at `./data/prioritized_tasks.db`. No Postgres, no cloud DB needed.

**Auth** is **off by default**. If you want a simple one-password gate later, set environment variables (see below).

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## (Optional) Simple password (no secrets file)
Set a single environment variable before starting Streamlit:

**Windows PowerShell**
```powershell
$env:AUTH_MODE="simple"
$env:APP_PASSWORD="yourpassword"
streamlit run streamlit_app.py
```

**macOS/Linux**
```bash
export AUTH_MODE=simple
export APP_PASSWORD=yourpassword
streamlit run streamlit_app.py
```

## CSV format
Supports your headers and new fields:
- Required: `Task` (or `Title`), `Group_Name`, `Status`
- Optional: `Depends_on`, `Priority`, `Notes`, `Due Date`, `id`, `description`

## Where data lives
- SQLite file at `./data/tasks.db` (created automatically). SQLite is an embedded SQL database that requires no server.

## Notes
- This version includes a light migration that adds **priority** and **notes** columns if not present.
- You can copy the whole folder to another machine; your data is in `data/tasks.db`.
