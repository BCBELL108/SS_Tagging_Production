# Production Tracking App (Streamlit + Neon Postgres)

This is a simple production tracking app with:
- **Submissions** (log daily work)
- **Analytics** (KPIs + charts)
- **Employees** (add/remove employees)
- **Customers** (add/remove customers)

## 1) Local run
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## 2) Add your Neon DB URL (Streamlit Secrets)
In Streamlit Cloud:
- App → Settings → Secrets
Add:
```toml
db_url = "postgresql://USER:PASSWORD@HOST/DB?sslmode=require"
```

Locally, you can create `.streamlit/secrets.toml` (do NOT commit it):
```toml
db_url = "postgresql://USER:PASSWORD@HOST/DB?sslmode=require"
```

## 3) First run
On first run, the app will automatically:
- create tables (if missing)
- seed customers + employees

## 4) Logo
Replace `assets/silverscreen_logo.png` with your real logo file (same name) to keep the layout intact.

## Notes
- Expected production is calculated from your rates:
  - Tags (8hr day): Del Sol = 800, Cariloha = 500, Purpose Built = 600, all other customers default to 800
  - Stickers (8hr day): Del Sol + Cariloha = 2400
- Expected is prorated by hours worked (Expected = DailyRate * Hours/8).
