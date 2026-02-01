import uuid
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text, bindparam
import plotly.express as px

# =============================================================================
# APP CONFIG
# =============================================================================
st.set_page_config(page_title="Production Tracker", page_icon="🏷️", layout="wide")
APP_TITLE = "Production Tracker"

LOGO_CANDIDATES = ["silverscreen_logo.png", "silverscreen_logo.PNG"]

WORK_TYPES = ["Tags", "Stickers", "Picking", "VAS"]

# Per 8-hour shift rates
TAG_RATES_PER_DAY = {
    "Del Sol": 800,
    "Cariloha": 500,
    "Purpose Built": 600,
    "Purpose-Built PRO": 600,
    "Purpose-Built Retail": 600,
}
DEFAULT_TAG_RATE_PER_DAY = 800

STICKER_RATE_PER_DAY = 2400
PICKING_RATE_PER_DAY = 3000
VAS_RATE_PER_DAY = 400

# RULES (your constraints)
# Tags allowed: Del Sol, Cariloha, and Purpose Built (including variants)
TAGS_ALLOWED_EXACT = {"Del Sol", "Cariloha", "Purpose Built"}
TAGS_ALLOWED_PREFIXES = ("Purpose",)  # catches Purpose-Built PRO / Retail, etc.

STICKERS_ALLOWED_CUSTOMERS = {"Del Sol", "Cariloha"}
PICKING_ONLY_CUSTOMER = "Del Sol"  # forced
# VAS allowed: all customers

# Optional analytics hide
HIDE_EMPLOYEE_FULLNAME_LOWER = ""  # e.g. "brandon bell"

# Employees to seed
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

# FULL customer seed list (yours) + internal is NOT used anymore for rules; kept harmless if present
CUSTOMER_SEED = [
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
    "Purpose Built",
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
# DB
# =============================================================================
@st.cache_resource
def get_engine():
    return create_engine(st.secrets["connections"]["db"]["url"], pool_pre_ping=True)

def fetch_df(sql_obj, params: dict | None = None) -> pd.DataFrame:
    with get_engine().begin() as conn:
        stmt = sql_obj if hasattr(sql_obj, "compile") else text(sql_obj)
        res = conn.execute(stmt, params or {})
        return pd.DataFrame(res.mappings().all())

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
# CACHED LOOKUPS (speed)
# =============================================================================
@st.cache_data(ttl=600)
def get_active_employees_df() -> pd.DataFrame:
    return fetch_df("""
        SELECT employee_id::text AS employee_id, first_name, last_name
        FROM employees
        WHERE is_active = TRUE
        ORDER BY first_name, last_name
    """)

@st.cache_data(ttl=600)
def get_active_customers_df() -> pd.DataFrame:
    return fetch_df("""
        SELECT customer_name
        FROM customers
        WHERE is_active = TRUE
        ORDER BY customer_name
    """)

def clear_caches():
    st.cache_data.clear()

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
        page = st.radio(
            "Navigate",
            ["Submissions", "Manage Records", "Analytics", "Employees", "Customers"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("Rules: Pick=Del Sol only • Stickers=Del Sol/Cariloha • Tags=Del Sol/Cariloha/Purpose • VAS=All")
    return page

def is_tags_allowed(customer: str) -> bool:
    if customer in TAGS_ALLOWED_EXACT:
        return True
    return any(customer.startswith(pref) for pref in TAGS_ALLOWED_PREFIXES)

def allowed_customers_for_worktype(work_type: str, all_customers: list[str]) -> list[str]:
    if work_type == "Picking":
        return [PICKING_ONLY_CUSTOMER] if PICKING_ONLY_CUSTOMER in all_customers else [PICKING_ONLY_CUSTOMER]
    if work_type == "Stickers":
        return [c for c in all_customers if c in STICKERS_ALLOWED_CUSTOMERS]
    if work_type == "Tags":
        return [c for c in all_customers if is_tags_allowed(c)]
    # VAS = all
    return all_customers

def expected_qty(customer: str, work_type: str, hours: float) -> int:
    if work_type == "Tags":
        rate = TAG_RATES_PER_DAY.get(customer, DEFAULT_TAG_RATE_PER_DAY)
    elif work_type == "Stickers":
        rate = STICKER_RATE_PER_DAY
    elif work_type == "Picking":
        rate = PICKING_RATE_PER_DAY
    else:  # VAS
        rate = VAS_RATE_PER_DAY
    return int(round(rate * (hours / 8.0)))

def month_start(d: date) -> date:
    return d.replace(day=1)

def previous_business_days(n: int, end_day: date | None = None) -> list[date]:
    end_day = end_day or date.today()
    out = []
    d = end_day
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)

def actual_label_for(work_type: str) -> str:
    if work_type == "Picking":
        return "Actual picks completed"
    if work_type == "VAS":
        return "Actual VAS completed"
    if work_type == "Stickers":
        return "Actual stickers completed"
    return "Actual tags completed"

# =============================================================================
# PAGES
# =============================================================================
def page_submissions():
    st.title("Submissions")
    st.write("Pick your name once, then add lines if your day was split. (Form = fast)")

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
    all_customers = cust_df["customer_name"].tolist()

    # session defaults
    if "active_employee_label" not in st.session_state:
        st.session_state.active_employee_label = emp_labels[0]
    if "active_entry_date" not in st.session_state:
        st.session_state.active_entry_date = date.today()
    if "draft_lines" not in st.session_state:
        st.session_state.draft_lines = [{
            "work_type": "Tags",
            "customer": "Del Sol" if "Del Sol" in all_customers else (all_customers[0] if all_customers else "Del Sol"),
            "hours": 8.0,
            "actual": None,
        }]

    # header controls (outside form)
    top1, top2, top3 = st.columns([1.3, 2.2, 1.2])
    st.session_state.active_entry_date = top1.date_input("Date", value=st.session_state.active_entry_date)
    st.session_state.active_employee_label = top2.selectbox(
        "Employee",
        emp_labels,
        index=emp_labels.index(st.session_state.active_employee_label) if st.session_state.active_employee_label in emp_labels else 0
    )

    if top3.button("➕ Add line", use_container_width=True):
        last = st.session_state.draft_lines[-1]
        st.session_state.draft_lines.append({
            "work_type": last["work_type"],
            "customer": last["customer"],
            "hours": 4.0,
            "actual": None,
        })
        st.rerun()

    st.markdown("---")

    # FAST: form prevents reruns on every widget change
    with st.form("submission_form", clear_on_submit=False):
        for i, line in enumerate(st.session_state.draft_lines):
            st.markdown(f"### Line {i+1}")
            c1, c2, c3, c4, c5 = st.columns([1.2, 2.0, 1.0, 1.6, 1.2])

            wt = c1.selectbox("Work Type", WORK_TYPES, index=WORK_TYPES.index(line["work_type"]), key=f"wt_{i}")
            line["work_type"] = wt

            allowed_customers = allowed_customers_for_worktype(wt, all_customers)
            if not allowed_customers:
                allowed_customers = [PICKING_ONLY_CUSTOMER] if wt == "Picking" else all_customers[:1]

            # customer selection rules
            if wt == "Picking":
                line["customer"] = PICKING_ONLY_CUSTOMER
                c2.selectbox("Customer", [PICKING_ONLY_CUSTOMER], index=0, disabled=True, key=f"cust_{i}")
            else:
                if line["customer"] not in allowed_customers:
                    line["customer"] = allowed_customers[0]
                line["customer"] = c2.selectbox("Customer", allowed_customers, index=allowed_customers.index(line["customer"]), key=f"cust_{i}")

            line["hours"] = c3.number_input("Hours", min_value=0.25, max_value=12.0, value=float(line["hours"]), step=0.25, key=f"hrs_{i}")

            exp = expected_qty(line["customer"], wt, float(line["hours"]))
            default_actual = exp if line["actual"] is None else int(line["actual"])
            line["actual"] = c4.number_input(actual_label_for(wt), min_value=0, value=int(default_actual), step=10, key=f"act_{i}")

            # quick remove (inside form as checkbox)
            remove = c5.checkbox("Remove", key=f"rm_{i}")

            eff = (line["actual"] / exp * 100.0) if exp > 0 else 0.0
            k1, k2, k3 = st.columns(3)
            k1.metric("Expected (prorated)", f"{exp:,}")
            k2.metric("Actual", f"{int(line['actual']):,}")
            k3.metric("Efficiency", f"{eff:.0f}%")
            st.markdown("---")

        submitted = st.form_submit_button("✅ Save submission", use_container_width=True)

    # apply removals after form render
    to_remove = [i for i in range(len(st.session_state.draft_lines)) if st.session_state.get(f"rm_{i}")]
    if to_remove:
        st.session_state.draft_lines = [ln for idx, ln in enumerate(st.session_state.draft_lines) if idx not in set(to_remove)]
        if not st.session_state.draft_lines:
            st.session_state.draft_lines = [{
                "work_type": "Tags",
                "customer": "Del Sol" if "Del Sol" in all_customers else (all_customers[0] if all_customers else "Del Sol"),
                "hours": 8.0,
                "actual": None,
            }]
        st.rerun()

    if submitted:
        # validate rules
        errors = []
        for idx, r in enumerate(st.session_state.draft_lines, start=1):
            if float(r["hours"]) <= 0:
                errors.append(f"Line {idx}: Hours must be > 0.")

            if r["work_type"] == "Stickers" and r["customer"] not in STICKERS_ALLOWED_CUSTOMERS:
                errors.append(f"Line {idx}: Stickers only allowed for Del Sol and Cariloha.")

            if r["work_type"] == "Picking" and r["customer"] != PICKING_ONLY_CUSTOMER:
                errors.append(f"Line {idx}: Picking must be Del Sol only.")

            if r["work_type"] == "Tags" and not is_tags_allowed(r["customer"]):
                errors.append(f"Line {idx}: Tags only allowed for Del Sol, Cariloha, and Purpose Built.")

        if errors:
            st.error("Fix these before saving:\n\n- " + "\n- ".join(errors))
            return

        employee_id = emp_label_to_id[st.session_state.active_employee_label]
        entry_date = st.session_state.active_entry_date

        for r in st.session_state.draft_lines:
            exp = expected_qty(r["customer"], r["work_type"], float(r["hours"]))
            exec_sql(
                """
                INSERT INTO production_entries
                  (entry_id, entry_date, employee_id, customer_name, work_type,
                   hours_worked, actual_qty, expected_qty)
                VALUES
                  (:id, :d, :eid::uuid, :c, :wt, :h, :a, :e)
                """,
                {
                    "id": str(uuid.uuid4()),
                    "d": entry_date,
                    "eid": employee_id,
                    "c": r["customer"],
                    "wt": r["work_type"],
                    "h": float(r["hours"]),
                    "a": int(r["actual"]),
                    "e": int(exp),
                },
            )

        st.success("Saved ✅")
        # reset lines only, keep employee/date for speed
        st.session_state.draft_lines = [{
            "work_type": "Tags",
            "customer": "Del Sol" if "Del Sol" in all_customers else (all_customers[0] if all_customers else "Del Sol"),
            "hours": 8.0,
            "actual": None,
        }]
        st.rerun()

def page_manage_records():
    st.title("Manage Records")
    st.caption("Fast edit/delete for typos (no charts).")

    emp_df = get_active_employees_df()
    if emp_df.empty:
        st.error("No employees found.")
        return

    emp_labels = (emp_df["first_name"] + " " + emp_df["last_name"]).tolist()
    emp_label_to_id = dict(zip(emp_labels, emp_df["employee_id"].tolist()))

    c1, c2, c3, c4 = st.columns([1.2, 1.8, 1.2, 1.2])
    start = c1.date_input("Start", value=date.today() - timedelta(days=7))
    end = c2.date_input("End", value=date.today())
    employee = c3.selectbox("Employee (optional)", ["All"] + emp_labels)
    limit = c4.selectbox("Rows", [25, 50, 100, 200], index=1)

    load = st.button("Load records", type="primary")
    if not load:
        st.info("Choose filters then click **Load records**.")
        return

    base = """
        SELECT
          pe.entry_id::text AS entry_id,
          pe.entry_date,
          e.first_name || ' ' || e.last_name AS employee,
          pe.work_type,
          pe.customer_name,
          pe.hours_worked,
          pe.actual_qty,
          pe.expected_qty
        FROM production_entries pe
        JOIN employees e ON e.employee_id = pe.employee_id
        WHERE pe.entry_date BETWEEN :s AND :e
    """

    params = {"s": start, "e": end}
    if employee != "All":
        base += " AND pe.employee_id = :eid::uuid"
        params["eid"] = emp_label_to_id[employee]

    base += """
        ORDER BY pe.entry_date DESC, e.first_name, e.last_name, pe.work_type, pe.customer_name
        LIMIT :lim
    """
    params["lim"] = int(limit)

    df = fetch_df(base, params)
    if df.empty:
        st.warning("No records found for that filter.")
        return

    df_display = df.copy()
    df_display["entry_date"] = pd.to_datetime(df_display["entry_date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(
        df_display[["entry_date", "employee", "work_type", "customer_name", "hours_worked", "actual_qty", "expected_qty", "entry_id"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.subheader("Edit or delete a single record")

    labels = [
        f'{r.entry_date} | {r.employee} | {r.work_type} | {r.customer_name} | hrs {float(r.hours_worked):g} | qty {int(r.actual_qty)}'
        for r in df.itertuples(index=False)
    ]
    pick = st.selectbox("Select record", labels)
    row = df.iloc[labels.index(pick)]

    # Edit fields
    e1, e2, e3, e4 = st.columns([1.2, 1.6, 2.2, 1.2])
    new_date = e1.date_input("Date", value=pd.to_datetime(row["entry_date"]).date(), key="mr_date")
    new_work_type = e2.selectbox("Work Type", WORK_TYPES, index=WORK_TYPES.index(row["work_type"]), key="mr_wt")

    # customer options depend on work type
    all_customers = get_active_customers_df()["customer_name"].tolist()
    allowed_customers = allowed_customers_for_worktype(new_work_type, all_customers)

    if new_work_type == "Picking":
        new_customer = PICKING_ONLY_CUSTOMER
        e3.selectbox("Customer", [PICKING_ONLY_CUSTOMER], index=0, disabled=True, key="mr_cust_dis")
    else:
        if row["customer_name"] not in allowed_customers and allowed_customers:
            default_cust = allowed_customers[0]
        else:
            default_cust = row["customer_name"]
        if default_cust not in allowed_customers and allowed_customers:
            default_cust = allowed_customers[0]
        new_customer = e3.selectbox(
            "Customer",
            allowed_customers if allowed_customers else [row["customer_name"]],
            index=(allowed_customers.index(default_cust) if allowed_customers and default_cust in allowed_customers else 0),
            key="mr_cust",
        )

    new_hours = e4.number_input("Hours", min_value=0.25, max_value=12.0, value=float(row["hours_worked"]), step=0.25, key="mr_hours")
    new_actual = st.number_input(actual_label_for(new_work_type), min_value=0, value=int(row["actual_qty"]), step=10, key="mr_actual")

    # validate
    edit_errors = []
    if new_work_type == "Stickers" and new_customer not in STICKERS_ALLOWED_CUSTOMERS:
        edit_errors.append("Stickers only allowed for Del Sol and Cariloha.")
    if new_work_type == "Picking" and new_customer != PICKING_ONLY_CUSTOMER:
        edit_errors.append("Picking must be Del Sol only.")
    if new_work_type == "Tags" and not is_tags_allowed(new_customer):
        edit_errors.append("Tags only allowed for Del Sol, Cariloha, and Purpose Built.")
    if float(new_hours) <= 0:
        edit_errors.append("Hours must be > 0.")

    if edit_errors:
        st.warning("• " + "\n• ".join(edit_errors))

    cbtn1, cbtn2 = st.columns([1, 1])

    if cbtn1.button("💾 Save changes", type="primary", disabled=bool(edit_errors)):
        new_expected = expected_qty(new_customer, new_work_type, float(new_hours))
        exec_sql(
            """
            UPDATE production_entries
            SET entry_date = :d,
                customer_name = :c,
                work_type = :wt,
                hours_worked = :h,
                actual_qty = :a,
                expected_qty = :e
            WHERE entry_id = :id::uuid
            """,
            {
                "d": new_date,
                "c": new_customer,
                "wt": new_work_type,
                "h": float(new_hours),
                "a": int(new_actual),
                "e": int(new_expected),
                "id": row["entry_id"],
            },
        )
        st.success("Updated ✅ (Click Load records again to refresh list.)")

    confirm = cbtn2.checkbox("Confirm delete (permanent)")
    if cbtn2.button("🗑️ Delete record", disabled=not confirm):
        exec_sql("DELETE FROM production_entries WHERE entry_id = :id::uuid", {"id": row["entry_id"]})
        st.success("Deleted ✅ (Click Load records again to refresh list.)")

def page_analytics():
    st.title("Analytics")

    # Keep analytics lightweight by default
    show_charts = st.checkbox("Show charts (slower)", value=False)

    today = date.today()
    start_default = month_start(today)

    c1, c2, c3 = st.columns([1.2, 1.2, 2.6])
    start = c1.date_input("Start", value=start_default)
    end = c2.date_input("End", value=today)
    type_filter = c3.multiselect("Work Types", WORK_TYPES, default=WORK_TYPES)

    if start > end:
        st.error("Start must be <= End.")
        return
    if not type_filter:
        st.info("Select at least one work type.")
        return

    sql = text("""
        SELECT
          pe.entry_id::text AS entry_id,
          pe.entry_date,
          e.first_name || ' ' || e.last_name AS employee,
          pe.work_type,
          pe.customer_name,
          pe.hours_worked,
          pe.actual_qty,
          pe.expected_qty
        FROM production_entries pe
        JOIN employees e ON e.employee_id = pe.employee_id
        WHERE pe.entry_date BETWEEN :s AND :e
          AND pe.work_type IN :types
          AND (:hide_name = '' OR lower(e.first_name || ' ' || e.last_name) <> :hide_name)
    """).bindparams(bindparam("types", expanding=True))

    df = fetch_df(sql, {"s": start, "e": end, "types": type_filter, "hide_name": HIDE_EMPLOYEE_FULLNAME_LOWER})
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
    k4.metric("Lines", f"{len(df):,}")

    if not show_charts:
        st.caption("Tip: use **Manage Records** for fast edit/delete.")
        return

    st.markdown("---")
    st.subheader("Pieces by employee (selected range)")
    by_emp = df.groupby(["employee"], as_index=False)["actual_qty"].sum().sort_values("actual_qty", ascending=False)
    fig = px.bar(by_emp, x="employee", y="actual_qty")
    st.plotly_chart(fig, use_container_width=True)

def page_employees():
    st.title("Employees")

    if st.button("🔄 Refresh cache"):
        clear_caches()
        st.success("Cache cleared.")
        st.rerun()

    emp = fetch_df("""
        SELECT employee_id::text AS employee_id, first_name, last_name, is_active
        FROM employees
        ORDER BY is_active DESC, first_name, last_name
    """)
    st.dataframe(emp[["first_name", "last_name", "is_active"]], use_container_width=True, hide_index=True)

    st.markdown("---")
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Add employee")
        fn = st.text_input("First name", key="emp_fn")
        ln = st.text_input("Last name", key="emp_ln")
        if st.button("➕ Add employee"):
            if not fn.strip() or not ln.strip():
                st.error("Enter both first and last name.")
            else:
                exec_sql(
                    """
                    INSERT INTO employees (employee_id, first_name, last_name, is_active)
                    VALUES (:id, :fn, :ln, TRUE)
                    ON CONFLICT DO NOTHING
                    """,
                    {"id": str(uuid.uuid4()), "fn": fn.strip(), "ln": ln.strip()},
                )
                clear_caches()
                st.success("Added ✅")
                st.rerun()

    with colB:
        st.subheader("Deactivate / Reactivate")
        labels = [f"{r.first_name} {r.last_name} ({'Active' if r.is_active else 'Inactive'})" for r in emp.itertuples(index=False)]
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
            clear_caches()
            st.success("Updated ✅")
            st.rerun()

def page_customers():
    st.title("Customers")

    if st.button("🔄 Refresh cache"):
        clear_caches()
        st.success("Cache cleared.")
        st.rerun()

    cust = fetch_df("""
        SELECT customer_id::text AS customer_id, customer_name, is_active
        FROM customers
        ORDER BY is_active DESC, customer_name
    """)
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
                clear_caches()
                st.success("Saved ✅")
                st.rerun()

    with colB:
        st.subheader("Deactivate / Reactivate")
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
            clear_caches()
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
elif page == "Manage Records":
    page_manage_records()
elif page == "Analytics":
    page_analytics()
elif page == "Employees":
    page_employees()
else:
    page_customers()
