# app.py
# =============================================================================
# SilverScreen Tagging Production Tracker (Streamlit + Neon/Postgres)
#
# FIXED: Reads Neon URL from Streamlit Secrets in THIS format:
#   [connections.db]
#   url = "postgresql://..."
#
# Also supports db_url / DATABASE_URL if present.
# =============================================================================

import os
import uuid
from datetime import date, timedelta
from typing import Any, Dict, List
from collections.abc import Mapping

import pandas as pd
import streamlit as st
import plotly.express as px

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


# =============================================================================
# CONFIG
# =============================================================================
APP_TITLE = "Tagging Production"
TEAM_DAILY_TARGET = 800

WORK_TYPES = ["Pick", "Tag/Sticker", "Hang Tags", "VAS"]

DEFAULT_CUSTOMERS = ["Del Sol", "Cariloha", "Purpose Built"]
DEFAULT_EMPLOYEES = [
    "Yesinia Alcala",
    "Brandon Bell",
    "Andie Dunsmore",
    "Scott Frank",
    "Kirstin Hurley",
    "John Kneiblher",
    "Jay lobos Virula",
    "Montana Marsh",
    "Izzy Price",
    "Joey Q",
    "Randi Robertison",
    "Steve Zenz",
]

SIDEBAR_LOGO_PATH = "silverscreen_logo.png"


# =============================================================================
# PAGE SETUP
# =============================================================================
st.set_page_config(page_title=APP_TITLE, layout="wide")

if os.path.exists(SIDEBAR_LOGO_PATH):
    st.sidebar.image(SIDEBAR_LOGO_PATH, use_container_width=True)
st.sidebar.title(APP_TITLE)


# =============================================================================
# DB URL (ROBUST + MATCHES YOUR SECRETS)
# =============================================================================
def _looks_like_postgres_url(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    return s.startswith("postgresql://") or s.startswith("postgres://")


def _get_nested(obj: Any, keys: List[str]) -> Any:
    """
    Safely get nested values from dict-like OR AttrDict-like objects.
    """
    cur = obj
    for k in keys:
        if cur is None:
            return None
        # Mapping (dict/AttrDict)
        if isinstance(cur, Mapping) and k in cur:
            cur = cur[k]
            continue
        # attribute-style
        if hasattr(cur, k):
            cur = getattr(cur, k)
            continue
        return None
    return cur


def get_db_url() -> str:
    # 1) Your exact Streamlit secret structure: [connections.db] url="..."
    url = _get_nested(st.secrets, ["connections", "db", "url"])
    if isinstance(url, str) and _looks_like_postgres_url(url):
        return url.strip()

    # 2) Common alternates (if you ever change later)
    for key in ["db_url", "DATABASE_URL", "NEON_DATABASE_URL", "POSTGRES_URL", "POSTGRESQL_URL"]:
        v = st.secrets.get(key, None)
        if isinstance(v, str) and _looks_like_postgres_url(v):
            return v.strip()

    # 3) Environment variables (local / backup)
    for key in ["DATABASE_URL", "NEON_DATABASE_URL", "POSTGRES_URL", "POSTGRESQL_URL"]:
        v = os.getenv(key)
        if isinstance(v, str) and _looks_like_postgres_url(v):
            return v.strip()

    # If we got here, the secrets are present but not in a usable place
    raise RuntimeError(
        "Could not find a Postgres URL. Expected Streamlit secrets like:\n"
        "[connections.db]\nurl = \"postgresql://...\""
    )


@st.cache_resource(show_spinner=False)
def get_engine():
    db_url = get_db_url()
    return create_engine(db_url, pool_pre_ping=True, pool_size=5, max_overflow=5)


# Create engine with a clear UI error if it fails
try:
    eng = get_engine()
except Exception as e:
    st.error("Database connection is not configured correctly.")
    st.write("Error:", str(e))
    st.info(
        "Your secrets should look like:\n\n"
        "[connections.db]\n"
        "url = \"postgresql://USER:PASSWORD@HOST/DB?sslmode=require\""
    )
    st.stop()


def exec_sql(sql: str, params: Dict[str, Any] | None = None, *, fetch: bool = False):
    """
    Executes SQL with auto-commit. If it errors, show the real SQL + params.
    """
    sql = sql.strip()
    try:
        with eng.begin() as conn:
            res = conn.execute(text(sql), params or {})
            if fetch:
                return res.mappings().all()
            return None
    except SQLAlchemyError as e:
        st.error("Database error while executing SQL.")
        st.code(sql, language="sql")
        if params:
            st.write("Params:", params)
        st.write("Error:", str(e))
        raise


# =============================================================================
# SCHEMA / INIT
# =============================================================================
def init_db():
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS employees (
            employee_id UUID PRIMARY KEY,
            employee_name TEXT NOT NULL UNIQUE,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS customers (
            customer_id UUID PRIMARY KEY,
            customer_name TEXT NOT NULL UNIQUE,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS submissions (
            submission_id UUID PRIMARY KEY,
            batch_id UUID NOT NULL,
            work_date DATE NOT NULL,
            employee_id UUID NOT NULL REFERENCES employees(employee_id) ON DELETE CASCADE,
            customer_id UUID NOT NULL REFERENCES customers(customer_id) ON DELETE RESTRICT,
            work_type TEXT NOT NULL,
            pieces INTEGER NOT NULL CHECK (pieces >= 0),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_submissions_work_date ON submissions(work_date);
        CREATE INDEX IF NOT EXISTS idx_submissions_employee ON submissions(employee_id);
        CREATE INDEX IF NOT EXISTS idx_submissions_batch ON submissions(batch_id);
        """
    )

    n_emp = int(exec_sql("SELECT COUNT(*) AS n FROM employees;", fetch=True)[0]["n"])
    if n_emp == 0:
        for nm in DEFAULT_EMPLOYEES:
            exec_sql(
                """
                INSERT INTO employees (employee_id, employee_name, is_active)
                VALUES (:id, :name, TRUE)
                ON CONFLICT (employee_name) DO NOTHING;
                """,
                {"id": str(uuid.uuid4()), "name": nm},
            )

    n_cust = int(exec_sql("SELECT COUNT(*) AS n FROM customers;", fetch=True)[0]["n"])
    if n_cust == 0:
        for nm in DEFAULT_CUSTOMERS:
            exec_sql(
                """
                INSERT INTO customers (customer_id, customer_name, is_active)
                VALUES (:id, :name, TRUE)
                ON CONFLICT (customer_name) DO NOTHING;
                """,
                {"id": str(uuid.uuid4()), "name": nm},
            )


init_db()


# =============================================================================
# CACHED LOOKUPS (speed)
# =============================================================================
@st.cache_data(show_spinner=False, ttl=30)
def get_active_employees() -> pd.DataFrame:
    rows = exec_sql(
        """
        SELECT employee_id::text AS employee_id, employee_name
        FROM employees
        WHERE is_active = TRUE
        ORDER BY employee_name;
        """,
        fetch=True,
    )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=30)
def get_active_customers() -> pd.DataFrame:
    rows = exec_sql(
        """
        SELECT customer_id::text AS customer_id, customer_name
        FROM customers
        WHERE is_active = TRUE
        ORDER BY
            CASE
                WHEN customer_name ILIKE 'Del Sol%' THEN 0
                WHEN customer_name ILIKE 'Cariloha%' THEN 1
                WHEN customer_name ILIKE 'Purpose Built%' THEN 2
                ELSE 99
            END,
            customer_name;
        """,
        fetch=True,
    )
    return pd.DataFrame(rows)


def clear_lookup_caches():
    get_active_employees.clear()
    get_active_customers.clear()


# =============================================================================
# BUSINESS RULES
# =============================================================================
def allowed_work_types_for_customer(customer_name: str) -> List[str]:
    cn = (customer_name or "").strip().lower()
    is_del_sol = ("del sol" in cn) or ("delsol" in cn)
    is_cariloha = "cariloha" in cn
    is_purpose_built = "purpose built" in cn

    allowed = ["VAS"]  # all customers
    if is_del_sol:
        allowed.append("Pick")  # ONLY Del Sol
    if is_del_sol or is_cariloha:
        allowed.append("Tag/Sticker")
    if is_del_sol or is_cariloha or is_purpose_built:
        allowed.append("Hang Tags")

    return [wt for wt in WORK_TYPES if wt in allowed]


# =============================================================================
# SESSION STATE
# =============================================================================
if "selected_employee_id" not in st.session_state:
    st.session_state.selected_employee_id = None
if "selected_employee_name" not in st.session_state:
    st.session_state.selected_employee_name = None
if "daily_work_date" not in st.session_state:
    st.session_state.daily_work_date = date.today()
if "daily_batch_lines" not in st.session_state:
    st.session_state.daily_batch_lines = []


# =============================================================================
# SIDEBAR NAV + EMPLOYEE SELECT (ONCE)
# =============================================================================
emps_df = get_active_employees()
if emps_df.empty:
    st.sidebar.warning("No active employees. Add employees in Manage Employees.")
else:
    emp_names = emps_df["employee_name"].tolist()
    idx = 0
    if st.session_state.selected_employee_name in emp_names:
        idx = emp_names.index(st.session_state.selected_employee_name)

    chosen_name = st.sidebar.selectbox("Employee", emp_names, index=idx, key="sb_employee")
    chosen_row = emps_df.loc[emps_df["employee_name"] == chosen_name].iloc[0]
    st.session_state.selected_employee_name = chosen_name
    st.session_state.selected_employee_id = chosen_row["employee_id"]

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Submissions", "Analytics", "Manage Employees", "Manage Customers"],
    index=0,
)


# =============================================================================
# HELPERS
# =============================================================================
def fmt_int(x: Any) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)


def business_days_back(n: int, from_day: date) -> List[date]:
    days = []
    d = from_day
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return sorted(days)


def delete_submission(submission_id: str):
    exec_sql("DELETE FROM submissions WHERE submission_id = :id;", {"id": submission_id})


def delete_batch(batch_id: str):
    exec_sql("DELETE FROM submissions WHERE batch_id = :bid;", {"bid": batch_id})


# =============================================================================
# PAGE: SUBMISSIONS
# =============================================================================
def page_submissions():
    st.header("Submissions")

    if not st.session_state.selected_employee_id:
        st.warning("Select an employee in the sidebar.")
        return

    col1, col2, col3 = st.columns([1.2, 1.0, 1.8])
    with col1:
        work_date = st.date_input("Work date", value=st.session_state.daily_work_date, key="sub_work_date")
        st.session_state.daily_work_date = work_date
    with col2:
        st.metric("Daily team target", fmt_int(TEAM_DAILY_TARGET))
    with col3:
        st.info("Build a daily batch (multiple line items) then click Submit Day once.")

    cust_df = get_active_customers()
    if cust_df.empty:
        st.warning("No active customers. Add customers in Manage Customers.")
        return

    st.subheader("Add line item")
    a1, a2, a3, a4 = st.columns([1.6, 1.1, 0.9, 0.7])

    with a1:
        cust_name = st.selectbox("Customer", cust_df["customer_name"].tolist(), key="sub_customer")
        cust_id = cust_df.loc[cust_df["customer_name"] == cust_name].iloc[0]["customer_id"]

    with a2:
        work_type = st.selectbox("Work type", allowed_work_types_for_customer(cust_name), key="sub_work_type")

    with a3:
        pieces = st.number_input("Pieces", min_value=0, step=1, value=0, key="sub_pieces")

    with a4:
        if st.button("Add", use_container_width=True):
            if int(pieces) <= 0:
                st.warning("Pieces must be greater than 0.")
            else:
                st.session_state.daily_batch_lines.append(
                    {"customer_id": cust_id, "customer_name": cust_name, "work_type": work_type, "pieces": int(pieces)}
                )
                st.success("Added.")

    st.divider()
    st.subheader("Current daily batch")

    if not st.session_state.daily_batch_lines:
        st.info("No line items yet.")
    else:
        lines_df = pd.DataFrame(st.session_state.daily_batch_lines)
        display_df = lines_df.groupby(["customer_name", "work_type"], as_index=False)["pieces"].sum()
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        total_pieces = int(display_df["pieces"].sum())
        st.metric("Total pieces in this daily batch", fmt_int(total_pieces))

        b1, b2, _ = st.columns([1, 1, 2])
        with b1:
            if st.button("Clear batch", use_container_width=True):
                st.session_state.daily_batch_lines = []
                st.success("Cleared.")
                st.rerun()

        with b2:
            if st.button("Submit Day", type="primary", use_container_width=True):
                batch_id = str(uuid.uuid4())
                employee_id = st.session_state.selected_employee_id

                for line in st.session_state.daily_batch_lines:
                    exec_sql(
                        """
                        INSERT INTO submissions (
                            submission_id, batch_id, work_date, employee_id, customer_id, work_type, pieces
                        )
                        VALUES (
                            :submission_id, :batch_id, :work_date, :employee_id, :customer_id, :work_type, :pieces
                        );
                        """,
                        {
                            "submission_id": str(uuid.uuid4()),
                            "batch_id": batch_id,
                            "work_date": work_date,
                            "employee_id": employee_id,
                            "customer_id": line["customer_id"],
                            "work_type": line["work_type"],
                            "pieces": int(line["pieces"]),
                        },
                    )

                st.session_state.daily_batch_lines = []
                st.success("Submitted!")
                st.rerun()

    st.divider()
    st.subheader("Review / delete records")

    r1, r2 = st.columns([1.2, 1.2])
    with r1:
        review_date = st.date_input("Review date", value=work_date, key="review_date")
    with r2:
        emp_names = emps_df["employee_name"].tolist()
        idx = emp_names.index(st.session_state.selected_employee_name) if st.session_state.selected_employee_name in emp_names else 0
        review_emp_name = st.selectbox("Employee", emp_names, index=idx, key="review_employee")
        review_emp_id = emps_df.loc[emps_df["employee_name"] == review_emp_name].iloc[0]["employee_id"]

    rows = exec_sql(
        """
        SELECT
            s.submission_id::text AS submission_id,
            s.batch_id::text AS batch_id,
            s.work_date,
            e.employee_name,
            c.customer_name,
            s.work_type,
            s.pieces,
            s.created_at
        FROM submissions s
        JOIN employees e ON e.employee_id = s.employee_id
        JOIN customers c ON c.customer_id = s.customer_id
        WHERE s.work_date = :d
          AND s.employee_id::text = :eid
        ORDER BY s.created_at DESC;
        """,
        {"d": review_date, "eid": review_emp_id},
        fetch=True,
    )
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No records found for that date/employee.")
        return

    st.dataframe(
        df[["work_date", "employee_name", "customer_name", "work_type", "pieces", "batch_id", "submission_id", "created_at"]],
        use_container_width=True,
        hide_index=True,
    )

    d1, d2 = st.columns([1, 1])
    with d1:
        sub_ids = df["submission_id"].tolist()
        sub_to_delete = st.selectbox("Delete one line (submission_id)", sub_ids, key="del_one_sub")
        if st.button("Delete line", use_container_width=True):
            delete_submission(sub_to_delete)
            st.success("Deleted line.")
            st.rerun()

    with d2:
        batch_ids = sorted(df["batch_id"].unique().tolist())
        batch_to_delete = st.selectbox("Delete whole batch (batch_id)", batch_ids, key="del_batch")
        if st.button("Delete batch", use_container_width=True):
            delete_batch(batch_to_delete)
            st.success("Deleted batch.")
            st.rerun()


# =============================================================================
# PAGE: ANALYTICS
# =============================================================================
def page_analytics():
    st.header("Analytics")

    today = date.today()
    default_start = today - timedelta(days=60)

    c1, c2, c3 = st.columns([1, 1, 1.6])
    with c1:
        start_date = st.date_input("Start date", value=default_start, key="ana_start")
    with c2:
        end_date = st.date_input("End date", value=today, key="ana_end")
    with c3:
        st.caption("Keep ranges tight for speed.")

    daily_rows = exec_sql(
        """
        SELECT s.work_date, SUM(s.pieces) AS total_pieces
        FROM submissions s
        WHERE s.work_date BETWEEN :start AND :end
        GROUP BY s.work_date
        ORDER BY s.work_date;
        """,
        {"start": start_date, "end": end_date},
        fetch=True,
    )
    daily_df = pd.DataFrame(daily_rows)
    if daily_df.empty:
        st.info("No data for that date range.")
        return

    daily_df["work_date"] = pd.to_datetime(daily_df["work_date"])
    daily_df["total_pieces"] = daily_df["total_pieces"].astype(int)

    st.subheader("Daily totals")
    st.plotly_chart(px.line(daily_df, x="work_date", y="total_pieces", markers=True), use_container_width=True)

    st.subheader("Last 5 business days (team total)")
    last5 = business_days_back(5, today)
    l5_start, l5_end = min(last5), max(last5)

    l5_rows = exec_sql(
        """
        SELECT s.work_date, SUM(s.pieces) AS total_pieces
        FROM submissions s
        WHERE s.work_date BETWEEN :start AND :end
        GROUP BY s.work_date
        ORDER BY s.work_date;
        """,
        {"start": l5_start, "end": l5_end},
        fetch=True,
    )
    l5_df = pd.DataFrame(l5_rows)
    l5_df["work_date"] = pd.to_datetime(l5_df["work_date"])
    l5_df["total_pieces"] = l5_df["total_pieces"].astype(int)

    idx = pd.to_datetime(pd.Series(last5))
    l5_df = l5_df.set_index("work_date").reindex(idx, fill_value=0).reset_index()
    l5_df.columns = ["work_date", "total_pieces"]

    st.plotly_chart(px.line(l5_df, x="work_date", y="total_pieces", markers=True), use_container_width=True)

    st.subheader("Monthly totals")
    tmp = daily_df.copy()
    tmp["month"] = tmp["work_date"].dt.to_period("M").dt.to_timestamp()
    monthly = tmp.groupby("month", as_index=False)["total_pieces"].sum()
    st.plotly_chart(px.line(monthly, x="month", y="total_pieces", markers=True), use_container_width=True)

    st.subheader("Work type breakdown")
    wt_rows = exec_sql(
        """
        SELECT s.work_type, SUM(s.pieces) AS total_pieces
        FROM submissions s
        WHERE s.work_date BETWEEN :start AND :end
        GROUP BY s.work_type
        ORDER BY total_pieces DESC;
        """,
        {"start": start_date, "end": end_date},
        fetch=True,
    )
    wt_df = pd.DataFrame(wt_rows)
    if not wt_df.empty:
        wt_df["total_pieces"] = wt_df["total_pieces"].astype(int)
        st.plotly_chart(px.bar(wt_df, x="work_type", y="total_pieces"), use_container_width=True)


# =============================================================================
# PAGE: MANAGE EMPLOYEES
# =============================================================================
def page_manage_employees():
    st.header("Manage Employees")

    st.subheader("Add employee")
    new_name = st.text_input("Employee name", key="add_emp_name").strip()
    if st.button("Add employee", type="primary"):
        if not new_name:
            st.warning("Enter a name.")
        else:
            exec_sql(
                """
                INSERT INTO employees (employee_id, employee_name, is_active)
                VALUES (:id, :name, TRUE)
                ON CONFLICT (employee_name) DO UPDATE SET is_active = TRUE;
                """,
                {"id": str(uuid.uuid4()), "name": new_name},
            )
            clear_lookup_caches()
            st.success("Added / activated.")
            st.rerun()

    st.divider()
    st.subheader("Deactivate / activate employees")

    rows = exec_sql(
        """
        SELECT employee_id::text AS employee_id, employee_name, is_active
        FROM employees
        ORDER BY employee_name;
        """,
        fetch=True,
    )
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No employees yet.")
        return

    st.dataframe(df, use_container_width=True, hide_index=True)

    e1, e2, e3 = st.columns([1.6, 1, 1])
    with e1:
        pick = st.selectbox("Select employee", df["employee_name"].tolist(), key="emp_pick_toggle")
        emp_id = df.loc[df["employee_name"] == pick].iloc[0]["employee_id"]
    with e2:
        if st.button("Deactivate", use_container_width=True):
            exec_sql("UPDATE employees SET is_active = FALSE WHERE employee_id::text = :id;", {"id": emp_id})
            clear_lookup_caches()
            st.success("Deactivated.")
            st.rerun()
    with e3:
        if st.button("Activate", use_container_width=True):
            exec_sql("UPDATE employees SET is_active = TRUE WHERE employee_id::text = :id;", {"id": emp_id})
            clear_lookup_caches()
            st.success("Activated.")
            st.rerun()


# =============================================================================
# PAGE: MANAGE CUSTOMERS
# =============================================================================
def page_manage_customers():
    st.header("Manage Customers")

    st.subheader("Add customer")
    new_name = st.text_input("Customer name", key="add_cust_name").strip()
    if st.button("Add customer", type="primary"):
        if not new_name:
            st.warning("Enter a customer name.")
        else:
            exec_sql(
                """
                INSERT INTO customers (customer_id, customer_name, is_active)
                VALUES (:id, :name, TRUE)
                ON CONFLICT (customer_name) DO UPDATE SET is_active = TRUE;
                """,
                {"id": str(uuid.uuid4()), "name": new_name},
            )
            clear_lookup_caches()
            st.success("Added / activated.")
            st.rerun()

    st.divider()
    st.subheader("Deactivate / activate customers")

    rows = exec_sql(
        """
        SELECT customer_id::text AS customer_id, customer_name, is_active
        FROM customers
        ORDER BY customer_name;
        """,
        fetch=True,
    )
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No customers yet.")
        return

    st.dataframe(df, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns([1.6, 1, 1])
    with c1:
        pick = st.selectbox("Select customer", df["customer_name"].tolist(), key="cust_pick_toggle")
        cust_id = df.loc[df["customer_name"] == pick].iloc[0]["customer_id"]
    with c2:
        if st.button("Deactivate", use_container_width=True):
            exec_sql("UPDATE customers SET is_active = FALSE WHERE customer_id::text = :id;", {"id": cust_id})
            clear_lookup_caches()
            st.success("Deactivated.")
            st.rerun()
    with c3:
        if st.button("Activate", use_container_width=True):
            exec_sql("UPDATE customers SET is_active = TRUE WHERE customer_id::text = :id;", {"id": cust_id})
            clear_lookup_caches()
            st.success("Activated.")
            st.rerun()


# =============================================================================
# ROUTER
# =============================================================================
if page == "Submissions":
    page_submissions()
elif page == "Analytics":
    page_analytics()
elif page == "Manage Employees":
    page_manage_employees()
elif page == "Manage Customers":
    page_manage_customers()
else:
    st.error("Unknown page.")
