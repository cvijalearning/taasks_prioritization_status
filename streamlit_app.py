# streamlit_app.py (v3.3 – inline editing enabled, Neon DB support, robust CSV, improved dashboard)
import os
import uuid
from datetime import datetime
from typing import List
import pandas as pd
import streamlit as st
from sqlalchemy import (
    create_engine, Table, Column, String, Text, DateTime, MetaData,
    select, func, insert, update, text, inspect, Integer, Boolean, Date
)
import altair as alt

# Daily Prioritization Constants
PRIORITIZED_GROUPS = ["Home", "Work", "Personal", "Wife", "Son", "Parents", "Auto", "PComp"]
PRIORITIZED_STATUSES = ["To Do", "In Progress", "Done", "Blocked", "Abandoned"]

ATTACHMENT_DIR = "data/attachments"
if not os.path.exists(ATTACHMENT_DIR):
    os.makedirs(ATTACHMENT_DIR, exist_ok=True)

# -----------------------------
# Auth (optional; disabled by default)
# -----------------------------
# Set AUTH_MODE to "off" (default) or "simple" (single password via env var or st.secrets)
AUTH_MODE = st.secrets.get("AUTH_MODE", os.getenv("AUTH_MODE", "off")).lower().strip()
SIMPLE_PASSWORD = st.secrets.get("APP_PASSWORD", os.getenv("APP_PASSWORD", ""))

def check_login():
    """Returns True if the user is allowed in.
    - AUTH_MODE == 'off' => always True (no login)
    - AUTH_MODE == 'simple' => require a password (APP_PASSWORD env var or st.secrets["password"])
    """
    if AUTH_MODE == "off":
        return True

    if not SIMPLE_PASSWORD:
        st.warning("Auth is enabled but no password is set. Set APP_PASSWORD in st.secrets or env.")
        return False

    password = SIMPLE_PASSWORD

    if st.session_state.get("auth_ok"):
        with st.sidebar:
            if st.button("Logout"):
                st.session_state["auth_ok"] = False
                st.rerun()
        return True

    with st.sidebar:
        st.subheader("🔐 Login")
        pw = st.text_input("Password", type="password")
        if st.button("Login"):
            if pw == password:
                st.session_state["auth_ok"] = True
                st.success("Logged in!")
                st.rerun()
            else:
                st.error("Invalid password")
    return False


# -----------------------------
# Database
# -----------------------------
def get_db_url():
    # 1. Check Streamlit Secrets (for Cloud deployment)
    url = st.secrets.get("DB_URL") or st.secrets.get("DATABASE_URL")
    if url:
        # Fix for SQLAlchemy 1.4+ which requires 'postgresql://' instead of 'postgres://'
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url

    # 2. Check Environment Variable
    env_url = os.getenv("DB_URL")
    if env_url:
        if env_url.startswith("postgres://"):
            env_url = env_url.replace("postgres://", "postgresql://", 1)
        return env_url

    # 3. Fallback to local SQLite
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    return f"sqlite:///{os.path.join(data_dir, 'prioritized_tasks.db')}"


def ensure_schema_migration(engine):
    """Adds new columns if upgrading an existing DB (priority, notes)."""
    try:
        insp = inspect(engine)
        if not insp.has_table('prioritized_tasks'):
            return
        cols = {c['name'] for c in insp.get_columns('prioritized_tasks')}
        with engine.begin() as conn:
            is_sqlite = engine.url.drivername == 'sqlite'
            bool_default = "0" if is_sqlite else "FALSE"
            
            if 'priority' not in cols:
                conn.execute(text("ALTER TABLE prioritized_tasks ADD COLUMN priority VARCHAR(32)"))
            if 'notes' not in cols:
                conn.execute(text("ALTER TABLE prioritized_tasks ADD COLUMN notes TEXT"))
            if 'due_date' not in cols:
                conn.execute(text("ALTER TABLE prioritized_tasks ADD COLUMN due_date DATE"))
            if 'actual_minutes' not in cols:
                conn.execute(text(f"ALTER TABLE prioritized_tasks ADD COLUMN actual_minutes INTEGER DEFAULT 0"))
            if 'is_running' not in cols:
                conn.execute(text(f"ALTER TABLE prioritized_tasks ADD COLUMN is_running BOOLEAN DEFAULT {bool_default}"))
            if 'timer_start_at' not in cols:
                conn.execute(text("ALTER TABLE prioritized_tasks ADD COLUMN timer_start_at DATETIME"))
    except Exception:
        # non-fatal
        pass


def get_engine_and_table():
    db_url = get_db_url()
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_size=3,        # keep it low for free tiers
        max_overflow=0,     # do not exceed the pool
        future=True
    )
    metadata = MetaData()

    # PostgreSQL requires 'false' instead of '0' for boolean defaults
    is_sqlite = engine.url.drivername == 'sqlite'
    bool_default = text('0') if is_sqlite else text('false')

    prioritized_tasks = Table(
        "prioritized_tasks",
        metadata,
        Column("id", String(36), primary_key=True),
        Column("title", Text, nullable=False),
        Column("group_name", String(32), nullable=False),
        Column("time_estimate", Text, nullable=True),
        Column("value_impact", Text, nullable=True),
        Column("priority_grade", String(32), nullable=True),
        Column("notes", Text, nullable=True),
        Column("status", String(32), nullable=False),
        Column("depends_on", Text, nullable=True),
        Column("due_date", Date, nullable=True),
        Column("actual_minutes", Integer, nullable=True, server_default=text('0')),
        Column("is_running", Boolean, nullable=True, server_default=bool_default),
        Column("timer_start_at", DateTime, nullable=True),
        Column("created_at", DateTime, nullable=False, server_default=func.now()),
        Column("updated_at", DateTime, nullable=False, server_default=func.now(), onupdate=func.now()),
    )

    archived_tasks = Table(
        "archived_tasks",
        metadata,
        Column("id", String(36), primary_key=True),
        Column("title", Text, nullable=False),
        Column("group_name", String(32), nullable=False),
        Column("time_estimate", Text, nullable=True),
        Column("value_impact", Text, nullable=True),
        Column("priority_grade", String(32), nullable=True),
        Column("notes", Text, nullable=True),
        Column("status", String(32), nullable=False),
        Column("depends_on", Text, nullable=True),
        Column("due_date", Date, nullable=True),
        Column("actual_minutes", Integer, nullable=True, server_default=text('0')),
        Column("is_running", Boolean, nullable=True, server_default=bool_default),
        Column("timer_start_at", DateTime, nullable=True),
        Column("created_at", DateTime, nullable=False, server_default=func.now()),
        Column("updated_at", DateTime, nullable=False, server_default=func.now(), onupdate=func.now()),
    )

    metadata.create_all(engine)
    ensure_schema_migration(engine)

    return engine, prioritized_tasks, archived_tasks

# -----------------------------
# CSV Parsing
# -----------------------------
def parse_csv_prioritized(uploaded_file):
    df = pd.read_csv(uploaded_file)
    
    # Standardize column names
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    
    # Map CSV columns to DB columns
    column_mapping = {
        "title": "title",
        "group": "group_name",
        "time": "time_estimate",
        "value": "value_impact",
        "grade": "priority_grade",
        "status": "status",
        "notes": "notes",
        "depends_on": "depends_on",
    }
    
    # Filter to only relevant columns and rename
    df = df[[col for col in column_mapping.keys() if col in df.columns]]
    df = df.rename(columns=column_mapping)
    
    # Add UUID and default values for missing columns
    df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    
    # Ensure all required columns exist, even if empty
    for col in ["title", "group_name", "status"]:
        if col not in df.columns:
            df[col] = ""
    
    # Fill NaNs with empty strings for text fields
    for col in ["time_estimate", "value_impact", "priority_grade", "notes", "depends_on"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
        else:
            df[col] = "" # Add if missing
            
    # Basic validation for required fields
    if df["title"].isnull().any() or (df["title"] == "").any():
        raise ValueError("Title column cannot be empty.")
    if df["group_name"].isnull().any() or (df["group_name"] == "").any():
        raise ValueError("Group column cannot be empty.")
    if df["status"].isnull().any() or (df["status"] == "").any():
        raise ValueError("Status column cannot be empty.")

    # Normalize group and status
    df['group_name'] = df['group_name'].apply(lambda x: x if x in PRIORITIZED_GROUPS else PRIORITIZED_GROUPS[0])
    df['status'] = df['status'].apply(lambda x: x if x in PRIORITIZED_STATUSES else PRIORITIZED_STATUSES[0])
    
    # Normalize priority grade
    def normalize_grade(grade):
        if pd.isna(grade): return None
        grade_str = str(grade).strip().upper()
        if grade_str.startswith("A"): return "A - Critical"
        if grade_str.startswith("B"): return "B - Important"
        if grade_str.startswith("C"): return "C - Medium"
        if grade_str.startswith("D"): return "D - Low"
        return None # Or a default
    df['priority_grade'] = df['priority_grade'].apply(normalize_grade)

    return df

# -----------------------------
# Pages
# -----------------------------
APP_TITLE = "Prioritized Task Manager"

def page_timeline_prioritized(engine, prioritized_tasks_tbl):
    st.header("📅 Project Timeline")
    with engine.begin() as conn:
        df = pd.read_sql(select(prioritized_tasks_tbl), conn)
    
    if df.empty or df['due_date'].isna().all():
        st.info("No tasks with due dates found. Add deadlines to see the timeline.")
        return

    # Filter to only tasks with due dates
    df_timeline = df[df['due_date'].notna()].copy()
    df_timeline['due_date'] = pd.to_datetime(df_timeline['due_date'])
    
    # Simple Gantt-like chart
    chart = alt.Chart(df_timeline).mark_bar().encode(
        x=alt.X('due_date:T', title='Deadline'),
        y=alt.Y('title:N', title='Task', sort='x'),
        color='status:N',
        tooltip=['title', 'due_date', 'status', 'priority_grade']
    ).properties(height=min(800, 100 + 30 * len(df_timeline)))
    
    st.altair_chart(chart, use_container_width=True)

def page_dashboard_prioritized(engine, prioritized_tasks_tbl, archived_tasks_tbl):
    st.header("📊 Statistics")
    
    with engine.begin() as conn:
        df = pd.read_sql(select(prioritized_tasks_tbl), conn)
        
    if df.empty:
        st.info("No data available.")
        return

    # Metrics
    total = len(df)
    done_count = len(df[df['status'] == 'Done'])
    todo_count = len(df[df['status'] == 'To Do'])
    inprog_count = len(df[df['status'] == 'In Progress'])
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tasks", total)
    c2.metric("Done", done_count)
    c3.metric("To Do", todo_count)
    c4.metric("In Progress", inprog_count)
    
    st.divider()
    
    # Charts
    c_left, c_right = st.columns(2)
    
    with c_left:
        st.subheader("Tasks by Group")
        chart_group = alt.Chart(df).mark_bar().encode(
            x=alt.X('group_name', title='Group'),
            y=alt.Y('count()', title='Count'),
            color='group_name'
        ).properties(height=300)
        st.altair_chart(chart_group, use_container_width=True)
        
    with c_right:
        st.subheader("Tasks by Priority")
        chart_grade = alt.Chart(df).mark_arc(innerRadius=50).encode(
            theta=alt.Theta("count()", stack=True),
            color=alt.Color("priority_grade"),
            tooltip=["priority_grade", "count()"]
        ).properties(height=300)
        st.altair_chart(chart_grade, use_container_width=True)

    st.subheader("Tasks by Status per Group")
    chart_status_group = alt.Chart(df).mark_bar().encode(
        x=alt.X('group_name', title='Group'),
        y=alt.Y('count()', title='Count'),
        color='status'
    ).properties(height=300)
    st.altair_chart(chart_status_group, use_container_width=True)

    # --- Dependency Graph ---
    st.subheader("🔗 Task Dependency Graph")
    deps = df[df['depends_on'].notna() & (df['depends_on'] != "")]
    if not deps.empty:
        dot = "digraph {\n"
        dot += "  rankdir=LR;\n"
        dot += "  node [shape=box, style=filled, color=lightblue];\n"
        # Nodes
        for _, row in df.iterrows():
            label = f"{row['title'][:30]}...\\n({row['status']})" if len(row['title']) > 30 else f"{row['title']}\\n({row['status']})"
            dot += f'  "{row["id"]}" [label="{label}"];\n'
        # Edges
        for _, row in deps.iterrows():
            parent_ids = [pid.strip() for pid in str(row['depends_on']).split(',')]
            for pid in parent_ids:
                # Only add edge if parent exists in df to avoid broken graph
                if pid in df['id'].values:
                    dot += f'  "{pid}" -> "{row["id"]}";\n'
        dot += "}"
        st.graphviz_chart(dot)
    else:
        st.info("No dependencies defined yet.")


def page_import_prioritized(engine, prioritized_tasks_tbl):
    st.header("📥 Import Tasks")
    st.markdown("""
    Upload a CSV file with columns: `Title`, `Group`, `Time`, `Value`, `Grade`, `Status`, `Notes`.
    - **Group**: Home, Work, Personal, Wife, Son, Parents, Auto, PComp
    - **Status**: To Do, In Progress, Done, Blocked, Abandoned
    - **Grade**: A, B, C, D (or full text)
    """)
    
    uploaded_file = st.file_uploader("Choose CSV", type=["csv"])
    if uploaded_file:
        try:
            df = parse_csv_prioritized(uploaded_file)
            st.success(f"Parsed {len(df)} tasks.")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("Import to Database", type="primary"):
                recs = df.to_dict(orient='records')
                # Ensure timestamps
                now = datetime.utcnow()
                for r in recs:
                    r['created_at'] = now
                    r['updated_at'] = now
                
                with engine.begin() as conn:
                    # simplistic upsert or insert
                    try:
                        conn.execute(insert(prioritized_tasks_tbl), recs)
                        st.success(f"Successfully imported {len(recs)} tasks!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Import failed: {str(e)}")
        except Exception as e:
            st.error(f"Error parsing file: {e}")


@st.fragment
def render_agile_management(engine, prioritized_tasks_tbl, all_task_options):
    # Load data for filtering
    with engine.begin() as conn:
        df = pd.read_sql(select(prioritized_tasks_tbl).order_by(prioritized_tasks_tbl.c.created_at.desc()), conn)

    if df.empty:
        st.info("No prioritized tasks yet.")
        return

    st.divider()
    
    # --- Filters ---
    c_f1, c_f2, c_f3 = st.columns(3)
    with c_f1:
        filter_group = st.multiselect("Filter Group", PRIORITIZED_GROUPS, default=None, key="mgmt_filter_group")
    with c_f2:
        filter_status = st.multiselect("Filter Status", PRIORITIZED_STATUSES, default=None, key="mgmt_filter_status")
    with c_f3:
        filter_grade = st.multiselect("Filter Priority Grade", ["A - Critical", "B - Important", "C - Medium", "D - Low"], default=None, key="mgmt_filter_grade")
        
    filtered_df = df.copy()
    if filter_group:
        filtered_df = filtered_df[filtered_df['group_name'].isin(filter_group)]
    if filter_status:
        filtered_df = filtered_df[filtered_df['status'].isin(filter_status)]
    if filter_grade:
        filtered_df = filtered_df[filtered_df['priority_grade'].isin(filter_grade)]

    # --- 1. Agile Board (Kanban) - Right below filters ---
    st.subheader("📋 Agile Board")
    cols = st.columns(len(PRIORITIZED_STATUSES))
    for i, status in enumerate(PRIORITIZED_STATUSES):
        with cols[i]:
            st.markdown(f"**{status}**")
            # Only show tasks matching filters in this column
            subset = filtered_df[filtered_df['status'] == status]
            
            for _, row in subset.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"Grade: {row['priority_grade'] or 'N/A'}")
                    if row['due_date']:
                        st.caption(f"📅 Due: {row['due_date']}")
                    
                    if row['notes']:
                        with st.expander("📝 View Notes"):
                            st.markdown(row['notes'])
                    
                    # --- Timer UI ---
                    spent = row['actual_minutes'] or 0
                    st.caption(f"⏱️ Spent: {spent}m")
                    if row['is_running']:
                        if st.button("⏹️ Stop Timer", key=f"stop_{row['id']}", type="primary"):
                            diff = datetime.utcnow() - row['timer_start_at']
                            mins = int(diff.total_seconds() / 60)
                            with engine.begin() as conn:
                                conn.execute(
                                    update(prioritized_tasks_tbl)
                                    .where(prioritized_tasks_tbl.c.id == row['id'])
                                    .values(
                                        is_running=False,
                                        actual_minutes=spent + mins,
                                        updated_at=datetime.utcnow()
                                    )
                                )
                            st.rerun()
                    else:
                        if st.button("▶️ Start Timer", key=f"start_{row['id']}"):
                            with engine.begin() as conn:
                                conn.execute(
                                    update(prioritized_tasks_tbl)
                                    .where(prioritized_tasks_tbl.c.id == row['id'])
                                    .values(
                                        is_running=True,
                                        timer_start_at=datetime.utcnow(),
                                        updated_at=datetime.utcnow()
                                    )
                                )
                            st.rerun()
                    
                    # --- Attachments List ---
                    task_dir = os.path.join(ATTACHMENT_DIR, str(row['id']))
                    if os.path.exists(task_dir):
                        files = [f for f in os.listdir(task_dir) if os.path.isfile(os.path.join(task_dir, f))]
                        if files:
                            with st.expander(f"📎 Files ({len(files)})"):
                                for f in files:
                                    fpath = os.path.join(task_dir, f)
                                    with open(fpath, "rb") as file_bytes:
                                        st.download_button(f"📥 {f}", data=file_bytes, file_name=f, key=f"dl_{row['id']}_{f}")
                    
                    new_status = st.selectbox(
                        "Move to:",
                        PRIORITIZED_STATUSES,
                        index=PRIORITIZED_STATUSES.index(status),
                        key=f"board_status_{row['id']}_frag",
                        label_visibility="collapsed"
                    )
                    
                    if new_status != status:
                        with engine.begin() as conn:
                            conn.execute(
                                update(prioritized_tasks_tbl)
                                .where(prioritized_tasks_tbl.c.id == row['id'])
                                .values(status=new_status, updated_at=datetime.utcnow())
                            )
                        st.rerun() # Partial rerun

                    # --- Card Edit Expander ---
                    # Use a versioned key to force collapse after update
                    v_key = f"v_{row['id']}"
                    if v_key not in st.session_state:
                        st.session_state[v_key] = 0
                    
                    with st.expander("📝 Edit Card", expanded=False):
                        with st.form(key=f"edit_card_form_{row['id']}_{st.session_state[v_key]}"):
                            edit_title = st.text_input("Title", value=row['title'])
                            edit_group = st.selectbox("Group", PRIORITIZED_GROUPS, index=PRIORITIZED_GROUPS.index(row['group_name']) if row['group_name'] in PRIORITIZED_GROUPS else 0, key=f"edit_group_{row['id']}_{st.session_state[v_key]}")
                            edit_time = st.text_input("Time Est.", value=row['time_estimate'] or "")
                            edit_value_impact = st.text_input("Value Impact", value=row['value_impact'] or "")
                            
                            grades = ["A - Critical", "B - Important", "C - Medium", "D - Low"]
                            grade_idx = grades.index(row['priority_grade']) if row['priority_grade'] in grades else 0
                            edit_priority_grade = st.selectbox("Priority Grade", grades, index=grade_idx, key=f"edit_grade_{row['id']}_{st.session_state[v_key]}")
                            
                            edit_notes = st.text_area("Notes", value=row['notes'] or "")
                            edit_due = st.date_input("Due Date", value=row['due_date'] if row['due_date'] else None)
                            
                            # Filter out current task from dependency options to avoid self-reference
                            dep_options = [opt for opt in all_task_options if opt['id'] != row['id']]
                            current_deps = [d.strip() for d in str(row['depends_on']).split(',')] if row['depends_on'] else []
                            
                            edit_deps = st.multiselect(
                                "Depends On",
                                options=dep_options,
                                default=[opt for opt in dep_options if opt['id'] in current_deps],
                                format_func=lambda x: f"{x['title']} (ID: {x['id'][:8]}...)",
                                key=f"edit_deps_{row['id']}_{st.session_state[v_key]}"
                            )
                            edit_depends_on = ",".join([opt['id'] for opt in edit_deps])
                            
                            new_attachment = st.file_uploader("Attach File", key=f"attach_{row['id']}")
                                                        
                            if st.form_submit_button("Update Card"):
                                if not edit_title:
                                    st.error("Title is required")
                                else:
                                    with engine.begin() as conn:
                                        conn.execute(
                                            update(prioritized_tasks_tbl)
                                            .where(prioritized_tasks_tbl.c.id == row['id'])
                                            .values(
                                                title=edit_title.strip(),
                                                group_name=edit_group,
                                                time_estimate=edit_time,
                                                value_impact=edit_value_impact,
                                                priority_grade=edit_priority_grade,
                                                notes=edit_notes,
                                                due_date=edit_due,
                                                depends_on=edit_depends_on,
                                                updated_at=datetime.utcnow()
                                            )
                                        )
                                        # Handle file upload
                                        if new_attachment:
                                            tdir = os.path.join(ATTACHMENT_DIR, str(row['id']))
                                            os.makedirs(tdir, exist_ok=True)
                                            with open(os.path.join(tdir, new_attachment.name), "wb") as f:
                                                f.write(new_attachment.getbuffer())
                                            st.success(f"Attached {new_attachment.name}")
                                            
                                    st.session_state[v_key] += 1 # Force key change to collapse
                                    st.rerun() # Partial rerun

                    # --- Card Delete Button ---
                    if st.button("🗑️ Delete Card", key=f"del_card_{row['id']}", type="secondary"):
                        with engine.begin() as conn:
                            conn.execute(
                                prioritized_tasks_tbl.delete()
                                .where(prioritized_tasks_tbl.c.id == row['id'])
                            )
                        st.success("Task deleted!")
                        st.rerun()

    st.divider()

    # --- 2. Task Editor (Detailed Table) ---
    st.subheader("📝 Task Details Editor")
    st.caption(f"Showing {len(filtered_df)} filtered tasks. Use 'Del' key or click row and use Trash icon to remove.")

    edited_df = st.data_editor(
        filtered_df,
        key="editor_prioritized_manage_frag",
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "id": st.column_config.TextColumn("ID", disabled=True),
            "title": st.column_config.TextColumn("Title"),
            "group_name": st.column_config.SelectboxColumn("Group", options=PRIORITIZED_GROUPS),
            "status": st.column_config.SelectboxColumn("Status", options=PRIORITIZED_STATUSES),
            "priority_grade": st.column_config.SelectboxColumn("Grade", options=["A - Critical", "B - Important", "C - Medium", "D - Low"]),
            "time_estimate": st.column_config.TextColumn("Time Est."),
            "value_impact": st.column_config.TextColumn("Value/Impact"),
            "notes": st.column_config.TextColumn("Notes"),
            "depends_on": st.column_config.TextColumn("Depends On"),
            "created_at": st.column_config.DatetimeColumn("Created", disabled=True, format="D MMM HH:mm"),
            "updated_at": st.column_config.DatetimeColumn("Updated", disabled=True, format="D MMM HH:mm"),
        },
        hide_index=True
    )

    if st.button("Save Table Changes", key="save_table_btn"):
        # Handle Deletions: Find IDs in original filtered_df that are missing in edited_df
        original_ids = set(filtered_df["id"].tolist())
        # Drop rows with no 'id' (newly added rows in editor - we handle additions via Form usually, but let's be safe)
        current_ids = set(edited_df["id"].dropna().tolist()) 
        deleted_ids = original_ids - current_ids
        
        if deleted_ids:
            with engine.begin() as conn:
                conn.execute(
                    prioritized_tasks_tbl.delete()
                    .where(prioritized_tasks_tbl.c.id.in_(list(deleted_ids)))
                )
            st.warning(f"Deleted {len(deleted_ids)} tasks.")

        # Handle Updates
        updates = []
        original_indexed = filtered_df.set_index("id")
        
        # Only iterate rows that still exist and have an ID
        records = edited_df.dropna(subset=["id"]).to_dict(orient="records")
        for rec in records:
            tid = rec.get("id")
            if not tid: continue
            
            if tid in original_indexed.index:
                orig_row = original_indexed.loc[tid]
                change_detected = False
                row_update = {}
                for col in ["title", "group_name", "status", "priority_grade", "time_estimate", "value_impact", "notes", "depends_on"]:
                    new_val = rec.get(col)
                    old_val = orig_row.get(col)
                    if new_val is None: new_val = ""
                    if old_val is None: old_val = ""
                    if str(new_val) != str(old_val):
                        row_update[col] = rec.get(col)
                        change_detected = True
                
                if change_detected:
                    row_update["updated_at"] = datetime.utcnow()
                    with engine.begin() as conn:
                        conn.execute(
                            update(prioritized_tasks_tbl)
                            .where(prioritized_tasks_tbl.c.id == tid)
                            .values(**row_update)
                        )
                    updates.append(tid)

        if updates or deleted_ids:
            if updates:
                st.success(f"Updated {len(updates)} tasks.")
            st.rerun()

def page_manage_prioritized(engine, prioritized_tasks_tbl, archived_tasks_tbl):
    st.header("⚙️ Agile Task Management")
    
    # Pre-fetch all tasks for dependency dropdowns
    with engine.begin() as conn:
        all_tasks = pd.read_sql(select(prioritized_tasks_tbl.c.id, prioritized_tasks_tbl.c.title), conn)
        all_task_options = all_tasks.to_dict(orient='records')

    # --- 1. Add New Task Form
    with st.expander("➕ Add New Prioritized Task", expanded=False):
        with st.form("add_prio_task_manage_main", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                title = st.text_input("Task Title")
                group = st.selectbox("Group", PRIORITIZED_GROUPS)
                time_est = st.text_input("Time Estimate (e.g. 30m, 2h)")
                value_impact = st.text_input("Value Impact")
            with col2:
                prio_grade = st.selectbox("Priority Grade", ["A - Critical", "B - Important", "C - Medium", "D - Low"])
                status = st.selectbox("Status", PRIORITIZED_STATUSES)
                due_date = st.date_input("Due Date", value=None)
                
                selected_deps = st.multiselect(
                    "Depends On",
                    options=all_task_options,
                    format_func=lambda x: f"{x['title']} (ID: {x['id'][:8]}...)"
                )
                depends_on = ",".join([opt['id'] for opt in selected_deps])
                
                notes = st.text_area("Notes")
            
            if st.form_submit_button("Create Task"):
                if not title:
                    st.error("Title is required")
                else:
                    new_task = {
                        "id": str(uuid.uuid4()),
                        "title": title.strip(),
                        "group_name": group,
                        "time_estimate": time_est,
                        "value_impact": value_impact,
                        "priority_grade": prio_grade,
                        "notes": notes,
                        "status": status,
                        "due_date": due_date,
                        "depends_on": depends_on,
                        "created_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                    with engine.begin() as conn:
                        conn.execute(insert(prioritized_tasks_tbl).values(**new_task))
                    st.success("Task added!")
                    st.rerun()

    # --- 2. Agile Management Suite (Fragmented for no-reload) ---
    render_agile_management(engine, prioritized_tasks_tbl, all_task_options)

def page_settings(engine, prioritized_tasks_tbl, archived_tasks_tbl):
    st.header("⚙️ Settings & Info")
    st.write("**Database URL:**")
    st.code(get_db_url())
    st.write("Auth mode:", AUTH_MODE)
    if AUTH_MODE == 'simple':
        st.caption("Set env var APP_PASSWORD to change the password.")

    st.subheader("Danger Zone")
    if st.button("Delete ALL prioritized tasks", type="primary"):
        with engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM prioritized_tasks")
        st.warning("All prioritized tasks deleted")

    st.subheader("🧹 Maintenance")
    if st.button("Archive Old Completed Tasks (14+ Days)"):
        with engine.begin() as conn:
            # Shift tasks to archive
            archive_query = text("""
                INSERT INTO archived_tasks 
                SELECT * FROM prioritized_tasks 
                WHERE (status = 'Done' OR status = 'Abandoned')
                AND updated_at < :cutoff
            """)
            cutoff = datetime.utcnow() - pd.Timedelta(days=14)
            result = conn.execute(archive_query, {"cutoff": cutoff})
            
            # Delete after archiving
            delete_query = text("""
                DELETE FROM prioritized_tasks 
                WHERE (status = 'Done' OR status = 'Abandoned')
                AND updated_at < :cutoff
            """)
            conn.execute(delete_query, {"cutoff": cutoff})
            
        st.success(f"Archived tasks successfully.")



# -----------------------------
# Main app
# -----------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    if not check_login():
        st.stop()

    engine, prioritized_tasks_tbl, archived_tasks_tbl = get_engine_and_table()

    # Sub-tabs for the main Daily Prioritization feature
    subtabs = st.tabs(["Dashboard", "Timeline", "Manage Tasks", "Import Tasks", "Settings"])
    
    with subtabs[0]:
        page_dashboard_prioritized(engine, prioritized_tasks_tbl, archived_tasks_tbl)

    with subtabs[1]:
        page_timeline_prioritized(engine, prioritized_tasks_tbl)
    
    with subtabs[2]:
        page_manage_prioritized(engine, prioritized_tasks_tbl, archived_tasks_tbl)

    with subtabs[3]:
        page_import_prioritized(engine, prioritized_tasks_tbl)
        
    with subtabs[4]:
        page_settings(engine, prioritized_tasks_tbl, archived_tasks_tbl)

if __name__ == "__main__":
    main()
