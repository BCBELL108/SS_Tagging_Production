\
import streamlit as st
import pandas as pd
from datetime import date, datetime
from sqlalchemy import text
import plotly.express as px

from db import get_engine, init_db, table_has_rows, seed_employees, seed_customers

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(
    page_title="Production Tracker",
    page_icon="🏷️",
    layout="wide",
)

APP_TITLE = "Production Tracker"
LOGO_PATH = "assets/silverscreen_logo.png"

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

DAILY_TEAM_TAG_TARGET = 800  # overall tags target (team-level reference)

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

CUSTOMER_SEED = [
    # Priority first
    "Del Sol",
    "Cariloha",
    "Purpose-Built PRO",
    "Purpose-Built Retail",
    # Everything else
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
    "Alpenglow Sports Inc",
    "AMB3R LLC",
    "American Solutions for Business",
    "Anning Johnson Company",
    "Aramark (Vestis)",
    "Armstrong Print & Promotional",
    "Badass Lass",
    "Bimark, Inc.",
    "Blackridge Branding",
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
    "Imagework Marketing",
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
    "YanceyWorks LLC",
    "Zazzle",
]

# =============================================================================
# DB helpers
# =============================================================================
@st.cache_resource
def _engine():
    eng = get_engine()
    init_db(eng)
    # Seed on first run (tables exist, might be empty)
    if not table_has_rows(eng, "employees"):
        seed_employees(eng, EMPLOYEE_SEED)
    if not table_has_rows(eng, "customers"):
        seed_customers(eng, CUSTOMER_SEED)
    return eng

def fetch_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    with _engine().begin() as conn:
        res = conn.execute(text(sql), params or {})
        rows = res.mappings().all()
    return pd.DataFrame(rows)

def exec_sql(sql: str, params: dict | None = None) -> None:
    with _engine().begin() as conn:
        conn.execute(text(sql), params or {})

# =============================================================================
# UI helpers
# =============================================================================
def sidebar():
    with st.sidebar:
        try:
            st.image(LOGO_PATH, use_container_width=True)
        except Exception:
            st.write("**(Add logo at assets/silverscreen_logo.png)**")
        st.markdown("----")
        page = st.radio("Navigate", ["Submissions", "Analytics", "Employees", "Customers"], label_visibility="collapsed")
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
    st.write("Log a shift (or partial shift) for **Tags** or **Stickers**.")

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

    colA, colB = st.columns([1, 1])
    with colA:
        entry_date = st.date_input("Date", value=date.today())
        emp_label_to_id = {format_employee(r): r["employee_id"] for _, r in emp_df.iterrows()}
        employee_label = st.selectbox("Employee", list(emp_label_to_id.keys()))
        work_type = st.radio("Work Type", ["Tags", "Stickers"], horizontal=True)
    with colB:
        customer = st.selectbox("Customer", cust_df["customer_name"].tolist())
        hours = st.number_input("Hours worked", min_value=0.25, max_value=12.0, value=8.0, step=0.25)
        if work_type == "Stickers" and customer not in STICKER_ALLOWED_CUSTOMERS:
            st.warning("Stickers are only allowed for **Del Sol** and **Cariloha** (per your standard).")
        exp = expected_qty(customer, work_type, float(hours))
        actual = st.number_input("Actual pieces completed", min_value=0, value=exp, step=10)
        notes = st.text_input("Notes (optional)", placeholder="e.g., helped on setup, downtime, etc.")

    # KPI preview
    eff = (actual / exp * 100.0) if exp > 0 else 0.0
    k1, k2, k3 = st.columns(3)
    k1.metric("Expected (prorated)", f"{exp:,}")
    k2.metric("Actual", f"{actual:,}")
    k3.metric("Efficiency", f"{eff:.0f}%")

    submitted = st.button("✅ Submit", use_container_width=True)

    if submitted:
        if work_type == "Stickers" and customer not in STICKER_ALLOWED_CUSTOMERS:
            st.error("Submission blocked: Stickers are only for Del Sol and Cariloha.")
            return

        import uuid
        exec_sql(
            """
            INSERT INTO production_entries
              (entry_id, entry_date, employee_id, customer_name, work_type, hours_worked, actual_qty, expected_qty, notes)
            VALUES
              (:id, :d, :eid::uuid, :c, :wt, :h, :a, :e, :n)
            """,
            {
                "id": str(uuid.uuid4()),
                "d": entry_date,
                "eid": emp_label_to_id[employee_label],
                "c": customer,
                "wt": work_type,
                "h": float(hours),
                "a": int(actual),
                "e": int(exp),
                "n": notes.strip() if notes else None,
            },
        )
        st.success("Saved ✅")

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

    df["efficiency_pct"] = (df["actual_qty"] / df["expected_qty"]).replace([pd.NA, pd.NaT, float("inf")], 0) * 100

    # KPIs
    total_actual = int(df["actual_qty"].sum())
    total_expected = int(df["expected_qty"].sum())
    eff = (total_actual / total_expected * 100.0) if total_expected else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Actual", f"{total_actual:,}")
    k2.metric("Total Expected", f"{total_expected:,}")
    k3.metric("Efficiency", f"{eff:.0f}%")
    k4.metric("Entries", f"{len(df):,}")

    st.markdown("----")

    # Daily totals
    daily = df.groupby(["entry_date", "work_type"], as_index=False)[["actual_qty", "expected_qty"]].sum()
    daily["efficiency_pct"] = (daily["actual_qty"] / daily["expected_qty"]).replace([pd.NA, pd.NaT, float("inf")], 0) * 100

    st.subheader("Daily totals")
    fig = px.line(daily, x="entry_date", y="actual_qty", color="work_type", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(daily, x="entry_date", y="expected_qty", color="work_type", markers=True)
    st.plotly_chart(fig2, use_container_width=True)

    # Employee ranking
    st.subheader("Employee performance (Actual vs Expected)")
    emp = df.groupby("employee", as_index=False)[["actual_qty", "expected_qty"]].sum()
    emp["efficiency_pct"] = (emp["actual_qty"] / emp["expected_qty"]).replace([pd.NA, pd.NaT, float("inf")], 0) * 100
    emp = emp.sort_values("efficiency_pct", ascending=False)

    fig3 = px.bar(emp, x="employee", y="efficiency_pct")
    fig3.update_layout(yaxis_title="Efficiency (%)", xaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(emp, use_container_width=True, hide_index=True)

    # Customer mix
    st.subheader("Customer mix (Actual pieces)")
    cust = df.groupby("customer_name", as_index=False)["actual_qty"].sum().sort_values("actual_qty", ascending=False).head(25)
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
    st.dataframe(emp[["first_name","last_name","is_active"]], use_container_width=True, hide_index=True)

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
                import uuid
                exec_sql(
                    """
                    INSERT INTO employees (employee_id, first_name, last_name, is_active)
                    VALUES (:id, :fn, :ln, TRUE)
                    """,
                    {"id": str(uuid.uuid4()), "fn": fn2, "ln": ln2},
                )
                st.success("Added ✅")
                st.rerun()

    with colB:
        st.subheader("Deactivate / Reactivate")
        active_labels = [f"{r.first_name} {r.last_name} ({'Active' if r.is_active else 'Inactive'})"
                         for r in emp.itertuples(index=False)]
        pick = st.selectbox("Select employee", active_labels)
        selected = emp.iloc[active_labels.index(pick)]
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
        st.dataframe(cust[["customer_name","is_active"]], use_container_width=True, hide_index=True)

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
                import uuid
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

            # guardrail: if trying to deactivate Del Sol/Cariloha/Purpose, warn
            if selected["customer_name"] in ("Del Sol", "Cariloha") or str(selected["customer_name"]).startswith("Purpose"):
                st.warning("Heads up: this is a primary customer in your workflow.")

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
# Main router
# =============================================================================
page = sidebar()

if page == "Submissions":
    page_submissions()
elif page == "Analytics":
    page_analytics()
elif page == "Employees":
    page_employees()
else:
    page_customers()
