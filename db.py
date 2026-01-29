\
import os
from dataclasses import dataclass
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

@dataclass(frozen=True)
class DBConfig:
    db_url: str

def get_db_url_from_streamlit_secrets() -> str:
    """
    Tries Streamlit secrets first, then env var DATABASE_URL / DB_URL.
    Keeps the app flexible across local + Streamlit Cloud.
    """
    try:
        import streamlit as st
        if "db_url" in st.secrets:
            return str(st.secrets["db_url"]).strip()
    except Exception:
        pass

    for k in ("DATABASE_URL", "DB_URL", "NEON_DATABASE_URL"):
        v = os.getenv(k)
        if v:
            return v.strip()

    raise RuntimeError(
        "Database URL not found. Add st.secrets['db_url'] in Streamlit (or set DATABASE_URL env var)."
    )

def get_engine() -> Engine:
    db_url = get_db_url_from_streamlit_secrets()
    # SQLAlchemy uses psycopg2-binary
    return create_engine(db_url, pool_pre_ping=True)

def init_db(engine: Engine) -> None:
    """
    Create tables if missing.
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS employees (
        employee_id UUID PRIMARY KEY,
        first_name  TEXT NOT NULL,
        last_name   TEXT NOT NULL,
        is_active   BOOLEAN NOT NULL DEFAULT TRUE,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS customers (
        customer_id UUID PRIMARY KEY,
        customer_name TEXT NOT NULL UNIQUE,
        is_active   BOOLEAN NOT NULL DEFAULT TRUE,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS production_entries (
        entry_id UUID PRIMARY KEY,
        entry_date DATE NOT NULL,
        employee_id UUID NOT NULL REFERENCES employees(employee_id),
        customer_name TEXT NOT NULL,
        work_type TEXT NOT NULL CHECK (work_type IN ('Tags','Stickers')),
        hours_worked NUMERIC(5,2) NOT NULL CHECK (hours_worked > 0),
        actual_qty INTEGER NOT NULL CHECK (actual_qty >= 0),
        expected_qty INTEGER NOT NULL CHECK (expected_qty >= 0),
        notes TEXT,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_entries_date ON production_entries(entry_date);
    CREATE INDEX IF NOT EXISTS idx_entries_emp ON production_entries(employee_id);
    CREATE INDEX IF NOT EXISTS idx_entries_customer ON production_entries(customer_name);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))

def table_has_rows(engine: Engine, table: str) -> bool:
    with engine.begin() as conn:
        res = conn.execute(text(f"SELECT EXISTS (SELECT 1 FROM {table} LIMIT 1)"))
        return bool(res.scalar())

def seed_employees(engine: Engine, employees: list[dict]) -> None:
    import uuid
    with engine.begin() as conn:
        for e in employees:
            conn.execute(
                text("""
                    INSERT INTO employees (employee_id, first_name, last_name, is_active)
                    VALUES (:id, :fn, :ln, TRUE)
                    ON CONFLICT DO NOTHING
                """),
                {"id": str(uuid.uuid4()), "fn": e["first_name"], "ln": e["last_name"]},
            )

def seed_customers(engine: Engine, customers: list[str]) -> None:
    import uuid
    with engine.begin() as conn:
        for c in customers:
            conn.execute(
                text("""
                    INSERT INTO customers (customer_id, customer_name, is_active)
                    VALUES (:id, :name, TRUE)
                    ON CONFLICT (customer_name) DO NOTHING
                """),
                {"id": str(uuid.uuid4()), "name": c},
            )
