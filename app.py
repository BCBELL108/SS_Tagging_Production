import uuid
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
import plotly.express as px

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(page_title="Production Tracker", page_icon="🏷️", layout="wide")

LOGO_FILES = ["silverscreen_logo.png", "silverscreen_logo.PNG"]
WORK_TYPES = ["Tags", "Stickers", "Picking", "VAS"]

# Rates per 8-hour shift
TAG_RATES = {
    "Del Sol": 800,
    "Cariloha": 500,
    "Purpose Built": 600,
    "Purpose-Built PRO": 600,
    "Purpose-Built Retail": 600,
}
DEFAULT_TAG_RATE = 800
STICKER_RATE = 2400
STICKER_CUSTOMERS = {"Del Sol", "Cariloha"}
PICKING_RATE = 3000
VAS_RATE = 400

INTERNAL_CUSTOMER = "Internal (Picking/VAS)"

# Employee seed data
EMPLOYEES = [
    ("Yesenia", "Alcala villa"),
    ("Andie", "Dunsmore"),
    ("Scott", "Frank"),
    ("Robin", "Hranac"),
    ("Kirstin", "Hurley"),
    ("John", "Kneiblher"),
    ("Jay", "Lobos Virula"),
    ("Montana", "Marsh"),
    ("Izzy", "Price"),
    ("Joseph", "Qualye"),
    ("Randi", "Robertson"),
    ("Steve", "Zenz"),
]

# Add your full customer list here
CUSTOMERS = [
    INTERNAL_CUSTOMER,
    "Del Sol",
    "Cariloha",
    "Purpose Built",
    "Purpose-Built PRO",
    "Purpose-Built Retail",
    # ... rest of your 200+ customers
]

# =============================================================================
# DATABASE
# =============================================================================
@st.cache_resource
def get_engine():
    return create_engine(st.secrets["db_url"], pool_pre_ping=True, pool_size=5, max_overflow=10)

def run_sql(sql, params=None):
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text(sql), params or {})

def get_df(sql, params=None):
    eng = get_engine()
    with eng.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})

def init_db():
    run_sql("""
    CREATE TABLE IF NOT EXISTS employees (
        employee_id uuid PRIMARY KEY,
        first_name text NOT NULL,
        last_name text NOT NULL,
        is_active boolean DEFAULT TRUE,
        created_at timestamptz DEFAULT now()
    );
    
    CREATE UNIQUE INDEX IF NOT EXISTS emp_name_idx 
        ON employees(lower(first_name), lower(last_name));
    
    CREATE TABLE IF NOT EXISTS customers (
        customer_id uuid PRIMARY KEY,
        customer_name text UNIQUE NOT NULL,
        is_active boolean DEFAULT TRUE,
        created_at timestamptz DEFAULT now()
    );
    
    CREATE TABLE IF NOT EXISTS production_entries (
        entry_id uuid PRIMARY KEY,
        entry_date date NOT NULL,
        employee_id uuid REFERENCES employees(employee_id),
        customer_name text NOT NULL,
        work_type text NOT NULL,
        hours_worked numeric(5,2) CHECK (hours_worked > 0),
        actual_qty integer CHECK (actual_qty >= 0),
        expected_qty integer CHECK (expected_qty >= 0),
        created_at timestamptz DEFAULT now()
    );
    
    CREATE INDEX IF NOT EXISTS prod_date_idx ON production_entries(entry_date);
    CREATE INDEX IF NOT EXISTS prod_emp_idx ON production_entries(employee_id);
    """)

def seed_data():
    # Employees
    if get_df("SELECT 1 FROM employees LIMIT 1").empty:
        for fn, ln in EMPLOYEES:
            run_sql(
                "INSERT INTO employees (employee_id, first_name, last_name) VALUES (:id, :fn, :ln) ON CONFLICT DO NOTHING",
                {"id": str(uuid.uuid4()), "fn": fn, "ln": ln}
            )
    
    # Customers
    for name in CUSTOMERS:
        run_sql(
            "INSERT INTO customers (customer_id, customer_name) VALUES (:id, :name) ON CONFLICT DO NOTHING",
            {"id": str(uuid.uuid4()), "name": name}
        )

def boot():
    init_db()
    seed_data()

# =============================================================================
# HELPERS
# =============================================================================
def find_logo():
    for f in LOGO_FILES:
        if Path(f).exists():
            return f
    return None

def sidebar_nav():
    with st.sidebar:
        logo = find_logo()
        if logo:
            st.image(logo, use_container_width=True)
        st.markdown("---")
        page = st.radio("", ["📝 Submit", "📊 Analytics", "👥 Employees", "🏢 Customers"], label_visibility="collapsed")
        st.markdown("---")
        st.caption("**Rates (8hrs):** Tags: 500-800 • Stickers: 2400 • Picking: 3000 • VAS: 400")
    return page

def calc_expected(customer, work_type, hours):
    if work_type == "Tags":
        rate = TAG_RATES.get(customer, DEFAULT_TAG_RATE)
    elif work_type == "Stickers":
        rate = STICKER_RATE
    elif work_type == "Picking":
        rate = PICKING_RATE
    else:
        rate = VAS_RATE
    return int(round(rate * (hours / 8.0)))

# =============================================================================
# PAGES
# =============================================================================
def page_submit():
    st.title("📝 Production Submission")
    st.markdown("Enter your production for the day. Add multiple lines for split shifts.")

    # Get data
    emps = get_df("SELECT employee_id::text AS id, first_name, last_name FROM employees WHERE is_active = TRUE ORDER BY first_name, last_name")
    custs = get_df("""
        SELECT customer_name FROM customers WHERE is_active = TRUE 
        ORDER BY CASE 
            WHEN customer_name = 'Del Sol' THEN 1
            WHEN customer_name = 'Cariloha' THEN 2
            WHEN customer_name LIKE 'Purpose%' THEN 3
            ELSE 99 END, customer_name
    """)

    if emps.empty or custs.empty:
        st.error("❌ No employees or customers found!")
        return

    emp_names = (emps["first_name"] + " " + emps["last_name"]).tolist()
    emp_map = dict(zip(emp_names, emps["id"].tolist()))
    cust_list = custs["customer_name"].tolist()

    # Init session
    if "rows" not in st.session_state:
        st.session_state.rows = [{
            "date": date.today(),
            "emp": emp_names[0],
            "type": "Tags",
            "cust": cust_list[0],
            "hrs": 8.0,
            "qty": None,
        }]

    def add():
        last = st.session_state.rows[-1]
        st.session_state.rows.append({
            "date": last["date"],
            "emp": last["emp"],  # Keep same employee
            "type": last["type"],
            "cust": last["cust"],
            "hrs": 4.0,
            "qty": None,
        })

    def remove(i):
        if len(st.session_state.rows) > 1:
            st.session_state.rows.pop(i)

    # Render
    for i, r in enumerate(st.session_state.rows):
        st.markdown(f"### Line {i+1}")
        
        # First line shows employee selector, others show it as disabled text
        if i == 0:
            c1, c2, c3 = st.columns(3)
            r["date"] = c1.date_input("Date", r["date"], key=f"d{i}", max_value=date.today())
            r["emp"] = c2.selectbox("Employee", emp_names, emp_names.index(r["emp"]), key=f"e{i}")
            r["type"] = c3.selectbox("Type", WORK_TYPES, WORK_TYPES.index(r["type"]), key=f"t{i}")
        else:
            # Additional lines: lock employee to first line's selection
            r["emp"] = st.session_state.rows[0]["emp"]
            c1, c2 = st.columns(2)
            r["date"] = c1.date_input("Date", r["date"], key=f"d{i}", max_value=date.today())
            r["type"] = c2.selectbox("Type", WORK_TYPES, WORK_TYPES.index(r["type"]), key=f"t{i}")

        c1, c2 = st.columns(2)
        
        if r["type"] in ("Picking", "VAS"):
            r["cust"] = INTERNAL_CUSTOMER
            c1.selectbox("Customer", [INTERNAL_CUSTOMER], 0, key=f"c{i}", disabled=True)
        else:
            if r["cust"] not in cust_list:
                r["cust"] = cust_list[0]
            r["cust"] = c1.selectbox("Customer", cust_list, cust_list.index(r["cust"]), key=f"c{i}")
        
        r["hrs"] = c2.number_input("Hours", 0.25, 12.0, float(r["hrs"]), 0.25, key=f"h{i}")

        exp = calc_expected(r["cust"], r["type"], r["hrs"])
        r["qty"] = st.number_input("Actual Pieces", 0, value=exp if r["qty"] is None else int(r["qty"]), step=10, key=f"q{i}", help=f"Expected: {exp:,}")

        if r["type"] == "Stickers" and r["cust"] not in STICKER_CUSTOMERS:
            st.warning("⚠️ Stickers only for Del Sol & Cariloha")

        eff = (r["qty"] / exp * 100) if exp > 0 else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected", f"{exp:,}")
        c2.metric("Actual", f"{int(r['qty']):,}")
        c3.metric("Efficiency", f"{eff:.0f}%")

        if len(st.session_state.rows) > 1:
            if st.button("🗑️ Remove", key=f"r{i}"):
                remove(i)
                st.rerun()

        st.markdown("---")

    # Actions
    c1, c2 = st.columns(2)
    
    if c1.button("➕ Add Line", use_container_width=True):
        add()
        st.rerun()
    
    if c2.button("✅ Save All", type="primary", use_container_width=True):
        errors = []
        for idx, r in enumerate(st.session_state.rows, 1):
            if r["hrs"] <= 0:
                errors.append(f"Line {idx}: Hours > 0 required")
            if r["type"] == "Stickers" and r["cust"] not in STICKER_CUSTOMERS:
                errors.append(f"Line {idx}: Invalid customer for stickers")

        if errors:
            for e in errors:
                st.error(e)
            return

        # Save
        for r in st.session_state.rows:
            exp = calc_expected(r["cust"], r["type"], r["hrs"])
            run_sql(
                """INSERT INTO production_entries (entry_id, entry_date, employee_id, customer_name, work_type, hours_worked, actual_qty, expected_qty)
                   VALUES (:id, :dt, :emp, :cust, :type, :hrs, :qty, :exp)""",
                {
                    "id": str(uuid.uuid4()),
                    "dt": r["date"],
                    "emp": emp_map[r["emp"]],
                    "cust": r["cust"],
                    "type": r["type"],
                    "hrs": r["hrs"],
                    "qty": int(r["qty"]),
                    "exp": exp,
                }
            )

        # Reset
        st.session_state.rows = [{
            "date": date.today(),
            "emp": emp_names[0],
            "type": "Tags",
            "cust": cust_list[0],
            "hrs": 8.0,
            "qty": None,
        }]
        
        st.success("✅ Saved!")
        st.balloons()
        st.rerun()


def page_analytics():
    st.title("📊 Analytics")

    c1, c2 = st.columns(2)
    start = c1.date_input("Start", date.today() - timedelta(7))
    end = c2.date_input("End", date.today())

    df = get_df("""
        SELECT p.entry_date, e.first_name || ' ' || e.last_name AS employee,
               p.customer_name, p.work_type, p.hours_worked, p.expected_qty, p.actual_qty,
               ROUND((p.actual_qty::numeric / NULLIF(p.expected_qty, 0)) * 100, 1) AS eff
        FROM production_entries p
        JOIN employees e ON p.employee_id = e.employee_id
        WHERE p.entry_date BETWEEN :s AND :e
        ORDER BY p.entry_date DESC
    """, {"s": start, "e": end})

    if df.empty:
        st.info("📭 No data")
        return

    # Convert date to just date (remove time)
    df['entry_date'] = pd.to_datetime(df['entry_date']).dt.date

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entries", len(df))
    c2.metric("Hours", f"{df['hours_worked'].sum():.1f}")
    c3.metric("Produced", f"{df['actual_qty'].sum():,}")
    
    total_exp = df["expected_qty"].sum()
    total_act = df["actual_qty"].sum()
    eff = (total_act / total_exp * 100) if total_exp > 0 else 0
    c4.metric("Efficiency", f"{eff:.0f}%")

    st.markdown("---")

    # Charts by Work Type
    st.markdown("### 📈 Production by Type")
    
    for work_type in WORK_TYPES:
        df_type = df[df['work_type'] == work_type]
        if not df_type.empty:
            st.markdown(f"#### {work_type}")
            
            # Aggregate by date
            daily = df_type.groupby('entry_date').agg({'actual_qty': 'sum'}).reset_index()
            
            fig = px.bar(
                daily,
                x='entry_date',
                y='actual_qty',
                title=f"{work_type} Production",
                labels={'entry_date': 'Date', 'actual_qty': 'Quantity'}
            )
            fig.update_xaxes(type='category')  # Treat dates as categories for clean display
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Top performers
    st.markdown("### 🏆 Top Performers")
    top = df.groupby("employee").agg({"actual_qty": "sum", "expected_qty": "sum"}).reset_index()
    top["efficiency"] = (top["actual_qty"] / top["expected_qty"] * 100).round(1)
    top = top.sort_values("efficiency", ascending=False).head(10)
    st.dataframe(top, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.dataframe(df, use_container_width=True, hide_index=True)


def page_employees():
    st.title("👥 Employees")

    tab1, tab2 = st.tabs(["View", "Add"])

    with tab1:
        df = get_df("SELECT first_name, last_name, is_active FROM employees ORDER BY first_name")
        st.dataframe(df, use_container_width=True, hide_index=True)

    with tab2:
        c1, c2 = st.columns(2)
        fn = c1.text_input("First")
        ln = c2.text_input("Last")
        
        if st.button("➕ Add"):
            if not fn or not ln:
                st.error("❌ Both required")
            else:
                try:
                    run_sql("INSERT INTO employees (employee_id, first_name, last_name) VALUES (:id, :fn, :ln)",
                           {"id": str(uuid.uuid4()), "fn": fn.strip(), "ln": ln.strip()})
                    st.success(f"✅ Added {fn} {ln}")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ {e}")


def page_customers():
    st.title("🏢 Customers")

    tab1, tab2 = st.tabs(["View", "Add"])

    with tab1:
        df = get_df("SELECT customer_name, is_active FROM customers ORDER BY customer_name")
        st.dataframe(df, use_container_width=True, hide_index=True)

    with tab2:
        name = st.text_input("Customer Name")
        
        if st.button("➕ Add"):
            if not name:
                st.error("❌ Required")
            else:
                try:
                    run_sql("INSERT INTO customers (customer_id, customer_name) VALUES (:id, :name)",
                           {"id": str(uuid.uuid4()), "name": name.strip()})
                    st.success(f"✅ Added {name}")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ {e}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    boot()
    
    st.markdown("<h1 style='text-align:center;'>Production Tracker</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray; font-size:14px;'>Silverscreen Decoration & Fulfillment®</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    page = sidebar_nav()
    
    if page == "📝 Submit":
        page_submit()
    elif page == "📊 Analytics":
        page_analytics()
    elif page == "👥 Employees":
        page_employees()
    elif page == "🏢 Customers":
        page_customers()


if __name__ == "__main__":
    main()
