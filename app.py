import os
import uuid
from pathlib import Path
from datetime import date

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
import plotly.express as px

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(
    page_title="Production Tracker",
    page_icon="🏷️",
    layout="wide",
)

APP_TITLE = "Production Tracker"

# Try these logo locations (works whether you keep it in repo root or /assets)
LOGO_CANDIDATES = [
    "assets/silverscreen_logo.png",
    "assets/silverscreen_logo.PNG",
    "silverscreen_logo.png",
    "silverscreen_logo.PNG",
]

# Rates are per 8-hour shift
TAG_RATES_PER_DAY = {
    "Del Sol": 800,
    "Cariloha": 500,
    "Purpose Built": 600,          # fallback spelling
    "Purpose-Built PRO": 600,
    "Purpose-Built Retail": 600,
}
DEFAULT_TAG_RATE_PER_DAY = 800

STICKER_ALLOWED_CUSTOMERS = {"Del Sol", "Cariloha"}
STICKER_RATE_PER_DAY = 2400

# Seed data
EMPLOYEE_SEED = [
    {"first_name": "Yesenia", "last_name": "Alcala villa"},
    {"first_name": "Brandon", "last_name": "Bell"},
    {"first_name": "Andie", "last_name": "Dunsmore"},
    {"first_name": "Scott", "last_name": "Frank"},
    {"first_name": "Robin", "last_name": "Hranac"},
    {"first_name": "Kirstin", "last_name": "Hurley"},
    {"first_name": "John", "last_name": "Kneiblher"},
    {"first_name": "Jay", "last_name": "Lobos Virula"},
    {"first_name": "Montana", "last_name": "Marsh"},
    {"first_name": "Izzy", "last_name": "Price"},
    {"first_name": "Joseph", "last_name": "Qualye"},
    {"first_name": "Randi", "last_name": "Robertson"},
    {"first_name": "Steve", "last_name": "Zenz"},
]

# (kept identical to your earlier list; trimmed here only for readability if needed)
CUSTOMER_SEED = [
    "Del Sol",
    "Cariloha",
    "Purpose-Built PRO",
    "Purpose-Built Retail",
    "2469 - The UPS Store",
    "33.Black, LLC",
    "4M Promotions",
    "503 Network LLC",
    "714 Creative",
    "A4 Promotions",
    "Abacus Products, Inc.",
    "ACI Printing Services, Inc.",
    "Adaptive Branding",
    # ... keep the rest of your list as-is ...
    "Zazzle",
]

# =============================================================================
# DB (Neon / Postgres)
# =============================================================================
def _get_database_url() -> str:
    """
    Supports your existing Streamlit secrets pattern:
      db_url = "postgresql://..."

    Also supports:
      DATABASE_URL = "postgresql://..."
    And env var:
      DATABASE_URL
    """

    # 1) Your standard key (from your screenshot)
    if "db_url" in st.secrets and str(st.secrets["db_url"]).strip():
        return str(st.secrets["db_url"]).strip()

    # 2) Common alternative key
    if "DATABASE_URL" in st.secrets and str(st.secrets["DATABASE_URL"]).strip():
        return str(st.secrets["DATABASE_URL"]).strip()

    # 3) Environment variable fallback
    env = os.getenv("DATABASE_URL", "").strip()
    if env:
        return env

    raise RuntimeError(
        "Database URL not set. Add `db_url` (preferred) or `DATABASE_URL` to Streamlit secrets, "
        "or set env var DATABASE_URL."
    )

@st.cache_resource
def get_engine():
    db_url = _get_database_url()

    # Neon/psycopg can use sslmode in the URL as you have it. SQLAlchemy will pass it through.
    # pool_pre_ping helps keep pooled connections healthy.
    return create_engine(db_url, pool_pre_ping=True)

def init_db():
    sql = """
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

    CREATE TABLE IF NOT EXISTS employees (
      employee_id uuid PRIMARY KEY,
      first_name text NOT NULL,
      last_name text NOT NULL,
      is_active boolean NOT NULL DEFAULT TRUE,
      created_at timestamptz NOT NULL DEFAULT now(),
      updated_at timestamptz NOT NULL DEFAULT now()
    );

    CREATE UNIQUE INDEX IF NOT EXISTS employees_name_uniq
      ON employees (lower(first_name), lower(last_name));

    CREATE TABLE IF NOT EXISTS customers (
      customer_id uuid PRIMARY KEY,
      customer_name text NOT NULL UNIQUE,
      is_active boolean NOT NULL DEFAULT TRUE,
      created_at timestamptz NOT NULL DEFAULT now(),
      updated_at timestamptz NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS production_entries (
      entry_id uuid PRIMARY KEY,
      entry_date date NOT NULL,
      employee_id uuid NOT NULL REFERENCES employees(employee_id),
      customer_name text NOT NULL,
      work_type text NOT NULL CHECK (work_type IN ('Tags','Stickers')),
      hours_worked numeric(5,2) NOT NULL CHECK (hours_worked > 0),
      actual_qty integer NOT NULL CHECK (actual_qty >= 0),
      expected_qty integer NOT NULL CHECK (expected_qty >= 0),
      notes text NULL,
      created_at timestamptz NOT NULL DEFAULT now()
    );

    CREATE INDEX IF NOT EXISTS production_entries_date_idx
      ON production_entries(entry_date);

    CREATE INDEX IF NOT EXISTS production_entries_employee_idx
      ON production_entries(employee_id);
    """
    with get_engine().begin() as conn:
        conn.execute(text(sql))

def fetch_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    with get_engine().begin() as conn:
        res = conn.execute(text(sql), params or {})
        rows = res.mappings().all()
    return pd.DataFrame(rows)

def exec_sql(sql: str, params: dict | None = None) -> None:
    with get_engine().begin() as conn:
        conn.execute(text(sql), params or {})

def table_has_rows(table: str) -> bool:
    df = fetch_df(f"SELECT 1 AS one FROM {table} LIMIT 1;")
    return not df.empty

def seed_employees():
    if table_has_rows("employees"):
        return
    for r in EMPLOYEE_SEED:
        exec_sql(
            """
            INSERT INTO employees (employee_id, first_name, last_name, is_active)
            VALUES (:id, :fn, :ln, TRUE)
            ON CONFLICT DO NOTHING
            """,
            {"id": str(uuid.uuid4()), "fn": r["first_name"], "ln": r["last_name"]},
        )

def seed_customers():
    if table_has_rows("customers"):
        return
    for name in CUSTOMER_SEED:
        exec_sql(
            """
            INSERT INTO customers (customer_id, customer_name, is_active)
            VALUES (:id, :name, TRUE)
            ON CONFLICT (customer_name) DO NOTHING
            """,
            {"id": str(uuid.uuid4()), "name": name},
        )

def boot():
    init_db()
    seed_employees()
    seed_customers()

# =============================================================================
# UI helpers
# =============================================================================
def find_logo_path() -> str | None:
    for p in LOGO_CANDIDATES:
        if Path(p).exists():
            return p
    return None

def sidebar():
    with st.sidebar:
        logo = find_logo_path()
        if logo:
            st.image(logo, use_container_width=True)
        else:
            st.caption("Logo not found. Put it at repo root as `silverscreen_logo.png` or in `assets/`.")
        st.markdown("----")
        page = st.radio(
            "Navigate",
            ["Submissions", "Analytics", "Employees", "Customers"],
            label_visibility="collapsed",
        )
        st.markdown("----")
        st.caption("🏷️ Tags target (team): 800/day • Stickers: 2400/day (Del Sol & Cariloha)")
    return page

def expected_qty(customer: str, work_type: str, hours: float) -> int:
    if work_type == "Tags":
        rate = TAG_RATES_PER_DAY.get(customer, DEFAULT_TAG_RATE_PER_DAY)
    else:
        rate = STICKER_RATE_PER_DAY
    exp = rate * (hours / 8.0)
    return int(round(exp))

def format_employee(row) -> str:
    return f"{row['first_name']} {row['last_name']}"

# =============================================================================
# Pages
# =============================================================================
def page_submissions():
    st.title("Submissions")
    st.write("Log a shift (or partial shift) for **Tags** or **Stickers**. Add multiple lines for split shifts.")

    emp_df = fetch_df("""
        SELECT employee_id::text AS employee_id, first_name, last_name
        FROM employees
        WHERE is_active = TRUE
        ORDER BY first_name, last_name
    """)
    cust_df = fetch_df("""
        SELECT customer_name
        FROM customers
        WHERE is_active = TRUE
        ORDER BY
          CASE
            WHEN customer_name = 'Del Sol' THEN 1
            WHEN customer_name = 'Cariloha' THEN 2
            WHEN customer_name LIKE 'Purpose%' THEN 3
            ELSE 99
          END,
          customer_name
    """)

    if emp_df.empty:
        st.error("No active employees. Add employees in the Employees page.")
        return
    if cust_df.empty:
        st.error("No customers found in DB.")
        return

    emp_label_to_id = {format_employee(r): r["employee_id"] for _, r in emp_df.iterrows()}

    if "draft_rows" not in st.session_state:
        st.session_state.draft_rows = [
            {
                "entry_date": date.today(),
                "employee_label": list(emp_label_to_id.keys())[0],
                "work_type": "Tags",
                "customer": cust_df["customer_name"].iloc[0],
                "hours": 8.0,
                "actual": None,
                "notes": "",
            }
        ]

    def add_row():
        last = st.session_state.draft_rows[-1]
        st.session_state.draft_rows.append(
            {
                "entry_date": last["entry_date"],
                "employee_label": last["employee_label"],
                "work_type": last["work_type"],
                "customer": last["customer"],
                "hours": 4.0,
                "actual": None,
                "notes": "",
            }
        )

    def remove_row(i: int):
        if len(st.session_state.draft_rows) <= 1:
            return
        st.session_state.draft_rows.pop(i)

    st.subheader("New entries")
    for i, row in enumerate(st.session_state.draft_rows):
        st.markdown(f"**Entry {i+1}**")
        c1, c2, c3, c4, c5 = st.columns([1.1, 1.6, 1.2, 1.6, 1.2])

        row["entry_date"] = c1.date_input("Date", value=row["entry_date"], key=f"d_{i}")
        row["employee_label"] = c2.selectbox(
            "Employee",
            list(emp_label_to_id.keys()),
            index=list(emp_label_to_id.keys()).index(row["employee_label"]),
            key=f"e_{i}",
        )
        row["work_type"] = c3.radio(
            "Work Type",
            ["Tags", "Stickers"],
            horizontal=True,
            index=0 if row["work_type"] == "Tags" else 1,
            key=f"wt_{i}",
        )
        row["customer"] = c4.selectbox(
            "Customer",
            cust_df["customer_name"].tolist(),
            index=cust_df["customer_name"].tolist().index(row["customer"]),
            key=f"c_{i}",
        )
        row["hours"] = c5.number_input(
            "Hours",
            min_value=0.25,
            max_value=12.0,
            value=float(row["hours"]),
            step=0.25,
            key=f"h_{i}",
        )

        exp = expected_qty(row["customer"], row["work_type"], float(row["hours"]))
        default_actual = exp if row["actual"] is None else int(row["actual"])
        row["actual"] = st.number_input(
            "Actual pieces completed",
            min_value=0,
            value=int(default_actual),
            step=10,
            key=f"a_{i}",
        )
        row["notes"] = st.text_input(
            "Notes (optional)",
            value=row.get("notes", ""),
            placeholder="e.g., helped on setup, downtime, etc.",
            key=f"n_{i}",
        )

        if row["work_type"] == "Stickers" and row["customer"] not in STICKER_ALLOWED_CUSTOMERS:
            st.warning("Stickers are only allowed for **Del Sol** and **Cariloha**.")

        eff = (row["actual"] / exp * 100.0) if exp > 0 else 0.0
        k1, k2, k3 = st.columns(3)
        k1.metric("Expected (prorated)", f"{exp:,}")
        k2.metric("Actual", f"{int(row['actual']):,}")
        k3.metric("Efficiency", f"{eff:.0f}%")

        if len(st.session_state.draft_rows) > 1:
            if st.button("🗑️ Remove this entry", key=f"rm_{i}"):
                remove_row(i)
                st.rerun()

        st.markdown("---")

    b1, b2 = st.columns([1, 2])
    if b1.button("➕ Add another submission", use_container_width=True):
        add_row()
        st.rerun()

    save_all = b2.button("✅ Save ALL entries", use_container_width=True)

    if save_all:
        errors = []
        for idx, r in enumerate(st.session_state.draft_rows, start=1):
            if r["work_type"] == "Stickers" and r["customer"] not in STICKER_ALLOWED_CUSTOMERS:
                errors.append(f"Entry {idx}: Stickers only allowed for Del Sol and Cariloha.")
            if float(r["hours"]) <= 0:
                errors.append(f"Entry {idx}: Hours must be > 0.")
        if errors:
            st.error("Fix these before saving:\n\n- " + "\n- ".join(errors))
            return

        for r in st.session_state.draft_rows:
            exp = expected_qty(r["customer"], r["work_type"], float(r["hours"]))
            exec_sql(
                """
                INSERT INTO production_entries
                  (entry_id, entry_date, employee_id, customer_name, work_type,
                   hours_worked, actual_qty, expected_qty, notes)
                VALUES
                  (:id, :d, :eid::uuid, :c, :wt, :h, :a, :e, :n)
                """,
                {
                    "id": str(uuid.uuid4()),
                    "d": r["entry_date"],
                    "eid": emp_label_to_id[r["employee_label"]],
                    "c": r["customer"],
                    "wt": r["work_type"],
                    "h": float(r["hours"]),
                    "a": int(r["actual"]),
                    "e": int(exp),
                    "n": r["notes"].strip() if r.get("notes") else None,
                },
            )

        st.success("Saved ✅")

        st.session_state.draft_rows = [
            {
                "entry_date": date.today(),
                "employee_label": list(emp_label_to_id.keys())[0],
                "work_type": "Tags",
                "customer": cust_df["customer_name"].iloc[0],
                "hours": 8.0,
                "actual": None,
                "notes": "",
            }
        ]
        st.rerun()

    st.markdown("### Recent entries (last 14 days)")
    recent = fetch_df("""
        SELECT
          pe.entry_date,
          e.first_name || ' ' || e.last_name AS employee,
          pe.customer_name,
          pe.work_type,
          pe.hours_worked,
          pe.actual_qty,
          pe.expected_qty,
          ROUND(CASE WHEN pe.expected_qty > 0 THEN (pe.actual_qty::numeric / pe.expected_qty) * 100 ELSE 0 END, 1) AS efficiency_pct,
          pe.notes
        FROM production_entries pe
        JOIN employees e ON e.employee_id = pe.employee_id
        WHERE pe.entry_date >= (CURRENT_DATE - INTERVAL '14 days')
        ORDER BY pe.entry_date DESC, pe.created_at DESC
        LIMIT 250
    """)
    if recent.empty:
        st.info("No entries yet.")
    else:
        st.dataframe(recent, use_container_width=True, hide_index=True)

def page_analytics():
    st.title("Analytics")
    st.write("Quick KPI view of production vs expected, by day, employee, and customer.")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        start = st.date_input("Start", value=date.today().replace(day=1))
    with col2:
        end = st.date_input("End", value=date.today())
    with col3:
        work_filter = st.multiselect("Work Type", ["Tags", "Stickers"], default=["Tags", "Stickers"])

    if start > end:
        st.error("Start date must be <= End date.")
        return

    df = fetch_df(
        """
        SELECT
          pe.entry_date,
          e.first_name || ' ' || e.last_name AS employee,
          pe.customer_name,
          pe.work_type,
          pe.hours_worked,
          pe.actual_qty,
          pe.expected_qty
        FROM production_entries pe
        JOIN employees e ON e.employee_id = pe.employee_id
        WHERE pe.entry_date BETWEEN :s AND :e
          AND pe.work_type = ANY(:work_types)
        """,
        {"s": start, "e": end, "work_types": work_filter},
    )

    if df.empty:
        st.info("No data in the selected range.")
        return

    df["efficiency_pct"] = (df["actual_qty"] / df["expected_qty"]).replace([pd.NA, float("inf")], 0) * 100

    total_actual = int(df["actual_qty"].sum())
    total_expected = int(df["expected_qty"].sum())
    eff = (total_actual / total_expected * 100.0) if total_expected else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Actual", f"{total_actual:,}")
    k2.metric("Total Expected", f"{total_expected:,}")
    k3.metric("Efficiency", f"{eff:.0f}%")
    k4.metric("Entries", f"{len(df):,}")

    st.markdown("----")

    daily = df.groupby(["entry_date", "work_type"], as_index=False)[["actual_qty", "expected_qty"]].sum()
    daily["efficiency_pct"] = (daily["actual_qty"] / daily["expected_qty"]).replace([pd.NA, float("inf")], 0) * 100

    st.subheader("Daily totals")
    st.plotly_chart(px.line(daily, x="entry_date", y="actual_qty", color="work_type", markers=True), use_container_width=True)
    st.plotly_chart(px.line(daily, x="entry_date", y="expected_qty", color="work_type", markers=True), use_container_width=True)

    st.subheader("Employee performance (Efficiency %)")
    emp = df.groupby("employee", as_index=False)[["actual_qty", "expected_qty"]].sum()
    emp["efficiency_pct"] = (emp["actual_qty"] / emp["expected_qty"]).replace([pd.NA, float("inf")], 0) * 100
    emp = emp.sort_values("efficiency_pct", ascending=False)

    fig3 = px.bar(emp, x="employee", y="efficiency_pct")
    fig3.update_layout(yaxis_title="Efficiency (%)", xaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(emp, use_container_width=True, hide_index=True)

    st.subheader("Customer mix (Actual pieces)")
    cust = (
        df.groupby("customer_name", as_index=False)["actual_qty"]
        .sum()
        .sort_values("actual_qty", ascending=False)
        .head(25)
    )
    fig4 = px.bar(cust, x="customer_name", y="actual_qty")
    fig4.update_layout(xaxis_title="", yaxis_title="Actual pieces")
    st.plotly_chart(fig4, use_container_width=True)

def page_employees():
    st.title("Employees")
    st.write("Add or deactivate employees. (Deactivated employees will disappear from Submissions.)")

    emp = fetch_df("""
        SELECT employee_id::text AS employee_id, first_name, last_name, is_active
        FROM employees
        ORDER BY is_active DESC, first_name, last_name
    """)
    st.dataframe(emp[["first_name", "last_name", "is_active"]], use_container_width=True, hide_index=True)

    st.markdown("----")
    colA, colB = st.columns([1, 1])

    with colA:
        st.subheader("Add employee")
        fn = st.text_input("First name", key="add_fn")
        ln = st.text_input("Last name", key="add_ln")
        if st.button("➕ Add", use_container_width=True):
            fn2, ln2 = fn.strip(), ln.strip()
            if not fn2 or not ln2:
                st.error("Enter both first and last name.")
            else:
                exec_sql(
                    """
                    INSERT INTO employees (employee_id, first_name, last_name, is_active)
                    VALUES (:id, :fn, :ln, TRUE)
                    ON CONFLICT DO NOTHING
                    """,
                    {"id": str(uuid.uuid4()), "fn": fn2, "ln": ln2},
                )
                st.success("Added ✅")
                st.rerun()

    with colB:
        st.subheader("Deactivate / Reactivate")
        labels = [
            f"{r.first_name} {r.last_name} ({'Active' if r.is_active else 'Inactive'})"
            for r in emp.itertuples(index=False)
        ]
        pick = st.selectbox("Select employee", labels)
        selected = emp.iloc[labels.index(pick)]
        new_state = not bool(selected["is_active"])
        btn_label = "Deactivate" if selected["is_active"] else "Reactivate"

        if st.button(f"🔁 {btn_label}", use_container_width=True):
            exec_sql(
                """
                UPDATE employees
                SET is_active = :s, updated_at = NOW()
                WHERE employee_id = :id::uuid
                """,
                {"s": new_state, "id": selected["employee_id"]},
            )
            st.success("Updated ✅")
            st.rerun()

    st.markdown("----")
    st.subheader("Danger zone (delete entries)")
    st.caption("Use with caution. This only deletes **production entries** (not employees/customers).")
    if st.button("🧨 Delete ALL production entries", use_container_width=True):
        exec_sql("DELETE FROM production_entries;")
        st.success("All entries deleted.")
        st.rerun()

def page_customers():
    st.title("Customers")
    st.write("Add or deactivate customers. (Inactive customers will disappear from Submissions.)")

    cust = fetch_df("""
        SELECT customer_id::text AS customer_id, customer_name, is_active
        FROM customers
        ORDER BY is_active DESC,
          CASE
            WHEN customer_name = 'Del Sol' THEN 1
            WHEN customer_name = 'Cariloha' THEN 2
            WHEN customer_name LIKE 'Purpose%' THEN 3
            ELSE 99
          END,
          customer_name
    """)
    if cust.empty:
        st.info("No customers yet.")
    else:
        st.dataframe(cust[["customer_name", "is_active"]], use_container_width=True, hide_index=True)

    st.markdown("----")
    colA, colB = st.columns([1, 1])

    with colA:
        st.subheader("Add customer")
        name = st.text_input("Customer name", key="add_customer_name", placeholder="e.g., New Customer Inc.")
        if st.button("➕ Add customer", use_container_width=True):
            nm = name.strip()
            if not nm:
                st.error("Enter a customer name.")
            else:
                exec_sql(
                    """
                    INSERT INTO customers (customer_id, customer_name, is_active)
                    VALUES (:id, :name, TRUE)
                    ON CONFLICT (customer_name) DO NOTHING
                    """,
                    {"id": str(uuid.uuid4()), "name": nm},
                )
                st.success("Saved ✅")
                st.rerun()

    with colB:
        st.subheader("Deactivate / Reactivate")
        if cust.empty:
            st.info("Add a customer first.")
        else:
            labels = [f"{r.customer_name} ({'Active' if r.is_active else 'Inactive'})" for r in cust.itertuples(index=False)]
            pick = st.selectbox("Select customer", labels)
            selected = cust.iloc[labels.index(pick)]
            new_state = not bool(selected["is_active"])
            btn_label = "Deactivate" if selected["is_active"] else "Reactivate"

            if st.button(f"🔁 {btn_label}", use_container_width=True):
                exec_sql(
                    """
                    UPDATE customers
                    SET is_active = :s, updated_at = NOW()
                    WHERE customer_id = :id::uuid
                    """,
                    {"s": new_state, "id": selected["customer_id"]},
                )
                st.success("Updated ✅")
                st.rerun()

# =============================================================================
# Main
# =============================================================================
st.title(APP_TITLE)

try:
    boot()
except Exception as e:
    st.error(f"Database init failed: {e}")
    st.stop()

page = sidebar()

if page == "Submissions":
    page_submissions()
elif page == "Analytics":
    page_analytics()
elif page == "Employees":
    page_employees()
else:
    page_customers()
