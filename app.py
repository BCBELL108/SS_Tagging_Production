import uuid
from pathlib import Path
from datetime import date, timedelta, datetime

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
import plotly.express as px

# =============================================================================
# APP CONFIG
# =============================================================================
st.set_page_config(page_title="Production Tracker", page_icon="🏷️", layout="wide")
APP_TITLE = "Production Tracker"

LOGO_CANDIDATES = [
    "silverscreen_logo.png",
    "silverscreen_logo.PNG",
    "assets/silverscreen_logo.png",
    "assets/silverscreen_logo.PNG",
]

# Work types (rates are internal only; not shown in UI)
WORK_TYPES = ["Tags", "Stickers", "Picking", "VAS"]

# Internal production rates (per 8-hour shift) - NEVER displayed
TAG_RATES_PER_DAY = {
    "Del Sol": 800,
    "Cariloha": 500,
    "Purpose Built": 600,
    "Purpose-Built PRO": 600,
    "Purpose-Built Retail": 600,
}
DEFAULT_TAG_RATE_PER_DAY = 800

STICKER_ALLOWED_CUSTOMERS = {"Del Sol", "Cariloha"}
STICKER_RATE_PER_DAY = 2400

PICKING_RATE_PER_DAY = 3000
VAS_RATE_PER_DAY = 400

INTERNAL_CUSTOMER_NAME = "Internal (Picking/VAS)"

# Always hide Brandon Bell in UI (even if present in DB)
HIDE_EMPLOYEE_FULLNAME_LOWER = "brandon bell"

# Employees to seed (does NOT include Brandon Bell)
EMPLOYEE_SEED = [
    {"first_name": "Yesenia", "last_name": "Alcala villa"},
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

# FULL customer list + internal customer
CUSTOMER_SEED = [
    INTERNAL_CUSTOMER_NAME,
    "2469 - The UPS Store",
    "33.Black, LLC",
    "4M Promotions",
    "503 Network LLC",
    "714 Creative",
    "A4 Promotions",
    "Abacus Products, Inc.",
    "ACI Printing Services, Inc.",
    "Adaptive Branding",
    "Ad Stuff, Inc.",
    "Albrecht (Branding by Beth)",
    "Alchemist Promo",
    "Alpenglow Sports Inc",
    "AMB3R LLC",
    "American Solutions for Business",
    "Anning Johnson Company",
    "Aramark (Vestis)",
    "Armstrong Print & Promotional",
    "Badass Lass",
    "Bimark, Inc.",
    "Blackridge Branding",
    "Blue Dragonfly Marketing, Inc",
    "Blue Label Distribution (HiLife)",
    "Bluelight Promotions",
    "BPL Supplies Inc",
    "Brand Original IPU",
    "Bravo Promotional Marketing",
    "Brent Binnall Enterprises",
    "Bright Print Works",
    "BSN Sports",
    "Bulldog Creative Agency",
    "B&W Wholesale",
    "Calla Products, LLC",
    "Care Youth Corporation",
    "Cariloha",
    "CDA Printing",
    "Classic Awards & Promotions",
    "Clayton AP Academy",
    "CLNC Sports dba Secondslide",
    "Clove and Twine",
    "Club Colors",
    "Clutch Creative",
    "Cole Apparel",
    "Color Graphics Screenprinting",
    "Colossal Printing Company LLC",
    "Cool Breeze Heating & Air Conditioning",
    "Corporate Couture",
    "Creative Marketing and Design AIA",
    "CrossFreedom",
    "Defero Swag",
    "Del Sol",
    "Deso Supply",
    "DFS West",
    "Divide Graphics",
    "Divot Dawgs",
    "Emblazeon",
    "eRetailing Associates, LLC",
    "Etched in Stone",
    "Eureka Shirt Circuit",
    "Evident Industries",
    "Factory Design Group",
    "Fastenal",
    "Feature Graphix",
    "Four Alarm Promotions IPU",
    "Four Twigs LLC",
    "Freedom USA (HiLife)",
    "Fuel",
    "GBrakes",
    "GeekHead Printing and Apparel",
    "Good News Collection",
    "Great Basin Decoration",
    "Gulf Coast Trades Center",
    "HALO/AdSource",
    "Happiscribble",
    "High Desert Print Company",
    "Home Means Nevada Co",
    "Hooked on Swag",
    "HSG Safety Supplies Inc.",
    "HSM Enterprises",
    "ICO Companies dba Red The Uniform Tailor",
    "Ideal Printing, Promos & Wearables",
    "Image Group",
    "Image Source",
    "Initial Impression",
    "Inkwell (Brandito)",
    "Innovative Impressions IPU",
    "Inproma LLC",
    "International Minute Press",
    "IZA Design Inc",
    "Jen McFerrin Creative",
    "Jetset Promotions LLC",
    "J&J Printing",
    "Johnson Promotions",
    "J&R Gear",
    "Kids Blanks",
    "Knoblauch Advertising",
    "Kug - Proforma",
    "Lakeview Threads",
    "Logo Boss",
    "Lookout Promotions",
    "LR Apparel",
    "LSK Branding",
    "Luxury Branded Goods",
    "Made to Order",
    "Madhouz LLC",
    "Makers NV",
    "Marco Ideas Unlimited",
    "Marco Polo Promotions LLC",
    "Matrix Promotional Marketing IPU",
    "Merch.com",
    "Monitor Premiums, LLC",
    "Montroy Signs & Graphics",
    "Moondeck",
    "Moore Promotions - Proforma",
    "Mountain Freak Boutique",
    "National Sports Apparel",
    "NDS AIA",
    "Needleworks Embroidery",
    "No Quarter Co",
    "North American Embroidery",
    "Northwood Creations",
    "Nothing Too Fancy",
    "On-Line Printing & Graphics",
    "Onyx Inc",
    "Opal Promotions",
    "Orangevale Copy Center",
    "Ozio Lifestyles LLC",
    "Paperworld Inc",
    "Par 5 Promotions",
    "Parle Enterprises, Inc",
    "Pica Marketing Group",
    "PIP Printing",
    "Premium Custom Solutions",
    "Print Head Inc",
    "Print Promo Factory",
    "Proforma Wine Country",
    "Proforma Your Best Corp.",
    "PromoCentric LLC",
    "Promo Dog Inc",
    "Promotional Edge",
    "Proud American Hunter",
    "Purpose-Built PRO",
    "Purpose-Built Retail",
    "Qhik Moto",
    "Quantum Graphics, Inc.",
    "Radar Promotions",
    "Rapt Clothing Inc",
    "Red Thread Labs",
    "Reno Motorsports Inc",
    "Reno Print Labs",
    "Reno Print Store",
    "Reno Typographers",
    "Rise Custom Apparel LLC",
    "Rite of Passage ATCS",
    "Rite of Passage Inc",
    "Rockland Aramark",
    "Round Up Creations LLC",
    "Rush Advertising LLC",
    "SanMar",
    "Score International",
    "SDG Promotions IPU",
    "Sierra Air",
    "Sierra Boat Company",
    "Sierra Mountain Graphics",
    "Signs by Van",
    "Silkletter",
    "Silkshop Screen Printing",
    "Silver Peak Promotions",
    "Silverscreen Decoration & Fulfillment",
    "Silverscreen Direct",
    "Skyward Corp dba Meridian Promotions",
    "SOBO Concepts LLC",
    "SpotFrog",
    "Spot On Signs",
    "Star Sports",
    "Sticker Pack",
    "Stock Roll Corp of America",
    "Swagger",
    "Swagoo Promotions",
    "Swizzle",
    "SynergyX1 LLC",
    "Tahoe Basics",
    "Tahoe LogoWear",
    "Teamworks",
    "Tee Shirt Bar",
    "The Brand Portal",
    "The Graphics Factory",
    "The Hat Source",
    "The Right Promotions",
    "The Sourcing Group, LLC",
    "The Sourcing Group Promo",
    "Thunder House Productions LLC",
    "TPG Trade Show & Events",
    "Treasure Mountain",
    "Triangle Design & Graphics LLC",
    "TR Miller",
    "TRSTY Media",
    "Truly Gifted",
    "Tugboat, Inc",
    "University of Nevada Equipment Room",
    "Unraveled Threads",
    "Upper Park Clothing",
    "UP Shirt Inc",
    "Vail Dunlap",
    "Washoe County",
    "Washoe Schools",
    "Way to Be Designs, LLC",
    "WearyLand",
    "Windy City Promos",
    "Wolfgangs",
    "W&T Graphix",
    "Xcel",
    "Yak Graphics",
    "YanceyWorks LLC",
    "Zazzle",
]

# =============================================================================
# DB (Neon / Postgres) — uses Streamlit connections secret:
# [connections.db]
# url = "postgresql://..."
# =============================================================================
@st.cache_resource
def get_engine():
    return create_engine(st.secrets["connections"]["db"]["url"], pool_pre_ping=True)

def fetch_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    with get_engine().begin() as conn:
        res = conn.execute(text(sql), params or {})
        rows = res.mappings().all()
    return pd.DataFrame(rows)

def exec_sql(sql: str, params: dict | None = None) -> None:
    with get_engine().begin() as conn:
        conn.execute(text(sql), params or {})

def init_db():
    exec_sql("""
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
      work_type text NOT NULL,
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
    """)

    exec_sql("""
    DO $$
    BEGIN
      IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'production_entries_work_type_check') THEN
        ALTER TABLE production_entries DROP CONSTRAINT production_entries_work_type_check;
      END IF;

      ALTER TABLE production_entries
      ADD CONSTRAINT production_entries_work_type_check
      CHECK (work_type IN ('Tags','Stickers','Picking','VAS'));
    EXCEPTION
      WHEN undefined_table THEN NULL;
    END $$;
    """)

def seed_employees():
    if not fetch_df("SELECT 1 FROM employees LIMIT 1;").empty:
        return

    batch = [{"id": str(uuid.uuid4()), "fn": r["first_name"], "ln": r["last_name"]} for r in EMPLOYEE_SEED]
    with get_engine().begin() as conn:
        conn.execute(
            text("""
                INSERT INTO employees (employee_id, first_name, last_name, is_active)
                VALUES (:id, :fn, :ln, TRUE)
                ON CONFLICT DO NOTHING
            """),
            batch,
        )

def seed_customers():
    existing = fetch_df("SELECT customer_name FROM customers;")
    existing_set = set(existing["customer_name"].tolist()) if not existing.empty else set()

    missing = [name for name in CUSTOMER_SEED if name not in existing_set]
    if not missing:
        return

    batch = [{"id": str(uuid.uuid4()), "name": n} for n in missing]
    with get_engine().begin() as conn:
        conn.execute(
            text("""
                INSERT INTO customers (customer_id, customer_name, is_active)
                VALUES (:id, :name, TRUE)
                ON CONFLICT (customer_name) DO NOTHING
            """),
            batch,
        )

def boot():
    init_db()
    seed_employees()
    seed_customers()

# =============================================================================
# UI HELPERS
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
        st.markdown("---")
        page = st.radio("Navigate", ["Submissions", "Analytics", "Employees", "Customers"], label_visibility="collapsed")
    return page

def expected_qty(customer: str, work_type: str, hours: float) -> int:
    # internal only
    if work_type == "Tags":
        rate = TAG_RATES_PER_DAY.get(customer, DEFAULT_TAG_RATE_PER_DAY)
    elif work_type == "Stickers":
        rate = STICKER_RATE_PER_DAY
    elif work_type == "Picking":
        rate = PICKING_RATE_PER_DAY
    else:  # VAS
        rate = VAS_RATE_PER_DAY
    return int(round(rate * (hours / 8.0)))

def previous_business_days(n: int, end_day: date | None = None) -> list[date]:
    end_day = end_day or date.today()
    out = []
    d = end_day
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)

def month_start(d: date) -> date:
    return d.replace(day=1)

def week_start_monday(d: date) -> date:
    return d - timedelta(days=d.weekday())

def actual_label_for(work_type: str) -> str:
    return {
        "Tags": "Pieces tagged",
        "Stickers": "Stickers adhered",
        "Picking": "Pieces picked",
        "VAS": "Pieces processed (VAS)",
    }.get(work_type, "Pieces completed")

def mmddyyyy_input(label: str, value: date, key: str) -> date:
    """
    Text input that enforces MM/DD/YYYY display/entry.
    Returns the prior value if parsing fails (and shows an error).
    """
    s = st.text_input(label, value.strftime("%m/%d/%Y"), key=key).strip()
    try:
        return datetime.strptime(s, "%m/%d/%Y").date()
    except ValueError:
        st.error(f"{label}: use MM/DD/YYYY (example: 02/01/2026)")
        return value

# =============================================================================
# CACHED LOOKUPS
# =============================================================================
@st.cache_data(ttl=30)
def get_active_employees_df():
    return fetch_df(
        """
        SELECT employee_id::text AS employee_id, first_name, last_name
        FROM employees
        WHERE is_active = TRUE
          AND lower(first_name || ' ' || last_name) <> :hide_name
        ORDER BY first_name, last_name
        """,
        {"hide_name": HIDE_EMPLOYEE_FULLNAME_LOWER},
    )

@st.cache_data(ttl=60)
def get_active_customers_df():
    return fetch_df(
        """
        SELECT customer_name
        FROM customers
        WHERE is_active = TRUE
        ORDER BY
          CASE
            WHEN customer_name = 'Del Sol' THEN 1
            WHEN customer_name = 'Cariloha' THEN 2
            WHEN customer_name LIKE 'Purpose%' THEN 3
            WHEN customer_name = :internal THEN 98
            ELSE 99
          END,
          customer_name
        """,
        {"internal": INTERNAL_CUSTOMER_NAME},
    )

# =============================================================================
# PAGES
# =============================================================================
def page_submissions():
    st.title("Submissions")
    st.write('Log a shift (or partial shift). Use "Add another submission" for split shifts.')

    emp_df = get_active_employees_df()
    cust_df = get_active_customers_df()

    if emp_df.empty:
        st.error("No active employees. Add employees in the Employees page.")
        return
    if cust_df.empty:
        st.error("No customers in DB.")
        return

    emp_labels = (emp_df["first_name"] + " " + emp_df["last_name"]).tolist()
    emp_label_to_id = dict(zip(emp_labels, emp_df["employee_id"].tolist()))
    customer_list = cust_df["customer_name"].tolist()

    if "draft_rows" not in st.session_state:
        st.session_state.draft_rows = [{
            "entry_date": date.today(),
            "employee_label": emp_labels[0],
            "work_type": "Tags",
            "customer": customer_list[0],
            "hours": 8.0,
            "actual": 0,  # do not suggest
        }]

    def add_row():
        # Always assume same employee + same date for split shifts
        last = st.session_state.draft_rows[-1]
        st.session_state.draft_rows.append({
            "entry_date": last["entry_date"],
            "employee_label": last["employee_label"],
            "work_type": last["work_type"],
            "customer": last["customer"],
            "hours": 4.0,
            "actual": 0,  # do not suggest
        })

    def remove_row(i: int):
        if len(st.session_state.draft_rows) <= 1:
            return
        st.session_state.draft_rows.pop(i)

    for i, row in enumerate(st.session_state.draft_rows):
        c1, c2, c3, c4, c5 = st.columns([1.35, 1.7, 1.3, 2.0, 1.0])

        with c1:
            row["entry_date"] = mmddyyyy_input("Date", row["entry_date"], key=f"d_{i}")

        row["employee_label"] = c2.selectbox(
            "Employee",
            emp_labels,
            index=emp_labels.index(row["employee_label"]),
            key=f"e_{i}",
        )

        row["work_type"] = c3.selectbox(
            "Work Type",
            WORK_TYPES,
            index=WORK_TYPES.index(row["work_type"]),
            key=f"wt_{i}",
        )

        if row["work_type"] in ("Picking", "VAS"):
            row["customer"] = INTERNAL_CUSTOMER_NAME
            c4.selectbox("Customer", [INTERNAL_CUSTOMER_NAME], index=0, key=f"c_{i}", disabled=True)
        else:
            if row["customer"] not in customer_list:
                row["customer"] = customer_list[0]
            row["customer"] = c4.selectbox(
                "Customer",
                customer_list,
                index=customer_list.index(row["customer"]),
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

        label = actual_label_for(row["work_type"])
        row["actual"] = st.number_input(
            label,
            min_value=0,
            value=int(row.get("actual", 0)),
            step=10,
            key=f"a_{i}",
        )

        if row["work_type"] == "Stickers" and row["customer"] not in STICKER_ALLOWED_CUSTOMERS:
            st.warning("Stickers are only allowed for **Del Sol** and **Cariloha**.")

        if len(st.session_state.draft_rows) > 1:
            if st.button("🗑️ Remove", key=f"rm_{i}"):
                remove_row(i)
                st.rerun()

        st.markdown("---")

    b1, b2 = st.columns([1, 2])
    if b1.button("➕ Add another submission", use_container_width=True):
        add_row()
        st.rerun()

    if b2.button("✅ Save ALL submissions", use_container_width=True):
        errors = []
        for idx, r in enumerate(st.session_state.draft_rows, start=1):
            if float(r["hours"]) <= 0:
                errors.append(f"Submission {idx}: Hours must be > 0.")
            if r["work_type"] == "Stickers" and r["customer"] not in STICKER_ALLOWED_CUSTOMERS:
                errors.append(f"Submission {idx}: Stickers only allowed for Del Sol and Cariloha.")
        if errors:
            st.error("Fix these before saving:\n\n- " + "\n- ".join(errors))
            return

        with get_engine().begin() as conn:
            batch = []
            for r in st.session_state.draft_rows:
                exp = expected_qty(r["customer"], r["work_type"], float(r["hours"]))
                batch.append({
                    "id": str(uuid.uuid4()),
                    "d": r["entry_date"],
                    "eid": emp_label_to_id[r["employee_label"]],
                    "c": r["customer"],
                    "wt": r["work_type"],
                    "h": float(r["hours"]),
                    "a": int(r["actual"]),
                    "e": int(exp),
                })

            conn.execute(
                text("""
                    INSERT INTO production_entries
                      (entry_id, entry_date, employee_id, customer_name, work_type,
                       hours_worked, actual_qty, expected_qty, notes)
                    VALUES
                      (:id, :d, :eid::uuid, :c, :wt, :h, :a, :e, NULL)
                """),
                batch
            )

        st.success("Saved ✅")
        st.session_state.draft_rows = [{
            "entry_date": date.today(),
            "employee_label": emp_labels[0],
            "work_type": "Tags",
            "customer": customer_list[0],
            "hours": 8.0,
            "actual": 0,
        }]
        st.rerun()

def page_analytics():
    st.title("Analytics")

    today = date.today()
    default_start = month_start(today)

    c1, c2, c3 = st.columns([1.2, 1.2, 2.6])
    with c1:
        start = mmddyyyy_input("Start", default_start, key="ana_start")
    with c2:
        end = mmddyyyy_input("End", today, key="ana_end")
    with c3:
        type_filter = st.multiselect("Work Types", WORK_TYPES, default=WORK_TYPES)

    if start > end:
        st.error("Start must be <= End.")
        return
    if not type_filter:
        st.info("Select at least one work type.")
        return

    df = fetch_df(
        """
        SELECT
          pe.entry_date,
          e.first_name || ' ' || e.last_name AS employee,
          pe.work_type,
          pe.customer_name,
          pe.actual_qty,
          pe.expected_qty
        FROM production_entries pe
        JOIN employees e ON e.employee_id = pe.employee_id
        WHERE pe.entry_date BETWEEN :s AND :e
          AND pe.work_type = ANY(:types)
          AND lower(e.first_name || ' ' || e.last_name) <> :hide_name
        """,
        {"s": start, "e": end, "types": type_filter, "hide_name": HIDE_EMPLOYEE_FULLNAME_LOWER},
    )

    if df.empty:
        st.info("No data for the selected range.")
        return

    total_actual = int(df["actual_qty"].sum())
    total_expected = int(df["expected_qty"].sum())
    eff = (total_actual / total_expected * 100.0) if total_expected else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Pieces", f"{total_actual:,}")
    k2.metric("Total Expected", f"{total_expected:,}")
    k3.metric("Efficiency", f"{eff:.0f}%")
    k4.metric("Submissions", f"{len(df):,}")

    st.markdown("---")

    st.subheader("Team total production — previous 5 business days")
    biz_days = previous_business_days(5, today)
    s5, e5 = biz_days[0], biz_days[-1]

    df_5 = fetch_df(
        """
        SELECT entry_date, SUM(actual_qty) AS total_actual
        FROM production_entries
        WHERE entry_date BETWEEN :s AND :e
        GROUP BY entry_date
        ORDER BY entry_date
        """,
        {"s": s5, "e": e5},
    )

    frame = pd.DataFrame({"entry_date": biz_days})
    if not df_5.empty:
        df_5["entry_date"] = pd.to_datetime(df_5["entry_date"]).dt.date
        frame = frame.merge(df_5, on="entry_date", how="left")
    frame["total_actual"] = frame["total_actual"].fillna(0).astype(int)

    fig1 = px.line(frame, x="entry_date", y="total_actual", markers=True)
    fig1.update_layout(xaxis_title="Day", yaxis_title="Pieces")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Monthly total production — weekly points (month-to-date)")
    m_start = month_start(today)
    df_m = fetch_df(
        """
        SELECT entry_date, SUM(actual_qty) AS total_actual
        FROM production_entries
        WHERE entry_date BETWEEN :mstart AND :mend
        GROUP BY entry_date
        ORDER BY entry_date
        """,
        {"mstart": m_start, "mend": today},
    )

    if df_m.empty:
        st.info("No month-to-date entries yet.")
    else:
        df_m["entry_date"] = pd.to_datetime(df_m["entry_date"]).dt.date
        df_m["week_start"] = df_m["entry_date"].apply(week_start_monday)
        weekly = df_m.groupby("week_start", as_index=False)["total_actual"].sum().sort_values("week_start")

        fig2 = px.line(weekly, x="week_start", y="total_actual", markers=True)
        fig2.update_layout(xaxis_title="Week (Mon start)", yaxis_title="Pieces")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.subheader("Employee output by work type (selected range)")
    by_emp = df.groupby(["work_type", "employee"], as_index=False)["actual_qty"].sum()
    by_emp = by_emp.sort_values(["work_type", "actual_qty"], ascending=[True, False])

    for wt in WORK_TYPES:
        sub = by_emp[by_emp["work_type"] == wt]
        if sub.empty:
            st.caption(f"No {wt} in this range.")
            continue
        fig = px.bar(sub, x="employee", y="actual_qty")
        fig.update_layout(title=f"{wt} — pieces by employee", xaxis_title="", yaxis_title="Pieces")
        st.plotly_chart(fig, use_container_width=True)

def page_employees():
    st.title("Employees")

    emp = fetch_df(
        """
        SELECT employee_id::text AS employee_id, first_name, last_name, is_active
        FROM employees
        WHERE lower(first_name || ' ' || last_name) <> :hide_name
        ORDER BY is_active DESC, first_name, last_name
        """,
        {"hide_name": HIDE_EMPLOYEE_FULLNAME_LOWER},
    )
    st.dataframe(emp[["first_name", "last_name", "is_active"]], use_container_width=True, hide_index=True)

    st.markdown("---")
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Add employee")
        fn = st.text_input("First name", key="emp_fn")
        ln = st.text_input("Last name", key="emp_ln")
        if st.button("➕ Add employee"):
            fn2, ln2 = fn.strip(), ln.strip()
            if not fn2 or not ln2:
                st.error("Enter both first and last name.")
            elif f"{fn2} {ln2}".strip().lower() == HIDE_EMPLOYEE_FULLNAME_LOWER:
                st.error("That employee is hidden from this app.")
            else:
                exec_sql(
                    """
                    INSERT INTO employees (employee_id, first_name, last_name, is_active)
                    VALUES (:id, :fn, :ln, TRUE)
                    ON CONFLICT DO NOTHING
                    """,
                    {"id": str(uuid.uuid4()), "fn": fn2, "ln": ln2},
                )
                get_active_employees_df.clear()
                st.success("Added ✅")
                st.rerun()

    with colB:
        st.subheader("Deactivate / Reactivate")
        if emp.empty:
            st.info("No employees available.")
            return

        labels = [
            f"{r.first_name} {r.last_name} ({'Active' if r.is_active else 'Inactive'})"
            for r in emp.itertuples(index=False)
        ]
        pick = st.selectbox("Select employee", labels, key="emp_pick")
        selected = emp.iloc[labels.index(pick)]
        new_state = not bool(selected["is_active"])
        btn = "Deactivate" if selected["is_active"] else "Reactivate"
        if st.button(f"🔁 {btn}", key="emp_toggle"):
            exec_sql(
                """
                UPDATE employees
                SET is_active = :s, updated_at = NOW()
                WHERE employee_id = :id::uuid
                """,
                {"s": new_state, "id": selected["employee_id"]},
            )
            get_active_employees_df.clear()
            st.success("Updated ✅")
            st.rerun()

def page_customers():
    st.title("Customers")

    cust = fetch_df(
        """
        SELECT customer_id::text AS customer_id, customer_name, is_active
        FROM customers
        ORDER BY is_active DESC,
          CASE
            WHEN customer_name = 'Del Sol' THEN 1
            WHEN customer_name = 'Cariloha' THEN 2
            WHEN customer_name LIKE 'Purpose%' THEN 3
            WHEN customer_name = :internal THEN 98
            ELSE 99
          END,
          customer_name
        """,
        {"internal": INTERNAL_CUSTOMER_NAME},
    )

    st.dataframe(cust[["customer_name", "is_active"]], use_container_width=True, hide_index=True)

    st.markdown("---")
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Add customer")
        name = st.text_input("Customer name", key="cust_name")
        if st.button("➕ Add customer"):
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
                get_active_customers_df.clear()
                st.success("Saved ✅")
                st.rerun()

    with colB:
        st.subheader("Deactivate / Reactivate")
        if cust.empty:
            st.info("No customers available.")
            return

        labels = [f"{r.customer_name} ({'Active' if r.is_active else 'Inactive'})" for r in cust.itertuples(index=False)]
        pick = st.selectbox("Select customer", labels, key="cust_pick")
        selected = cust.iloc[labels.index(pick)]
        new_state = not bool(selected["is_active"])
        btn = "Deactivate" if selected["is_active"] else "Reactivate"
        if st.button(f"🔁 {btn}", key="cust_toggle"):
            exec_sql(
                """
                UPDATE customers
                SET is_active = :s, updated_at = NOW()
                WHERE customer_id = :id::uuid
                """,
                {"s": new_state, "id": selected["customer_id"]},
            )
            get_active_customers_df.clear()
            st.success("Updated ✅")
            st.rerun()

# =============================================================================
# MAIN
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
