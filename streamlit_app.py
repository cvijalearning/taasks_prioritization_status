# streamlit_app.py (Activities Dashboard with Neon DB & Auth)
import os
import uuid
from datetime import datetime, date
from typing import List, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import (
    create_engine, Table, Column, String, Text, DateTime, MetaData, Date,
    CheckConstraint, select, func, insert, update, text, inspect
)
from sqlalchemy.exc import OperationalError
import altair as alt

APP_TITLE = "Activities Dashboard"

ALLOWED_GROUPS = ["PComp", "Home", "BH&CAP", "Fatherhood"]
ALLOWED_STATUSES = [
    "New",
    "Depends on (Optional)",
    "In Progress",
    "Done",
    "Abandoned",
    "Blocked",
]
PRIORITY_CHOICES = ["Low", "Medium", "High", "Critical"]

# -----------------------------
# Auth
# -----------------------------
# Set AUTH_MODE to "off" (default) or "simple" (single password via env var or st.secrets)
AUTH_MODE = st.secrets.get("AUTH_MODE", os.getenv("AUTH_MODE", "off")).lower().strip()
SIMPLE_PASSWORD = st.secrets.get("APP_PASSWORD", os.getenv("APP_PASSWORD", ""))

def check_login():
    """Returns True if the user is allowed in."""
    if AUTH_MODE == "off":
        return True

    if not SIMPLE_PASSWORD:
        st.warning("Auth is enabled but no password is set. Set APP_PASSWORD in st.secrets or env.")
        return False

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
            if pw == SIMPLE_PASSWORD:
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
    # 1. Check Streamlit Secrets
    url = st.secrets.get("DB_URL") or st.secrets.get("DATABASE_URL")
    if url:
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
    return f"sqlite:///{os.path.join(data_dir, 'tasks.db')}"

def ensure_schema_migration(engine):
    """Adds new columns if upgrading an existing DB."""
    try:
        insp = inspect(engine)
        if not insp.has_table('tasks'):
            return
        cols = {c['name'] for c in insp.get_columns('tasks')}
        with engine.begin() as conn:
            if 'priority' not in cols:
                conn.execute(text("ALTER TABLE tasks ADD COLUMN priority VARCHAR(32)"))
            if 'notes' not in cols:
                conn.execute(text("ALTER TABLE tasks ADD COLUMN notes TEXT"))
            if 'description' not in cols:
                conn.execute(text("ALTER TABLE tasks ADD COLUMN description TEXT"))
    except Exception:
        pass

def get_engine_and_table():
    db_url = get_db_url()
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_size=3,
        max_overflow=0,
        future=True
    )
    metadata = MetaData()

    tasks = Table(
        "tasks",
        metadata,
        Column("id", String(36), primary_key=True),
        Column("title", Text, nullable=False),
        Column("description", Text, nullable=True),
        Column("group_name", String(32), nullable=False),
        Column("status", String(32), nullable=False),
        Column("depends_on", Text, nullable=True),
        Column("due_date", Date, nullable=True),
        Column("priority", String(32), nullable=True),
        Column("notes", Text, nullable=True),
        Column("created_at", DateTime, nullable=False, server_default=func.now()),
        Column("updated_at", DateTime, nullable=False, server_default=func.now(), onupdate=func.now()),
        CheckConstraint(
            f"group_name IN ({', '.join([repr(g) for g in ALLOWED_GROUPS])})",
            name="ck_group_name",
        ),
        CheckConstraint(
            f"status IN ({', '.join([repr(s) for s in ALLOWED_STATUSES])})",
            name="ck_status",
        ),
    )
    metadata.create_all(engine)
    ensure_schema_migration(engine)
    return engine, tasks

def upsert_tasks(engine, tasks_tbl, records: List[dict]):
    with engine.begin() as conn:
        for rec in records:
            # Filter out None values to let DB defaults work
            rec = {k: v for k, v in rec.items() if v is not None}
            # Try to update
            res = conn.execute(
                update(tasks_tbl).where(tasks_tbl.c.id == rec["id"]).values(**rec)
            )
            # If no row updated, insert
            if res.rowcount == 0:
                conn.execute(insert(tasks_tbl).values(**rec))

def list_tasks_df(engine, tasks_tbl, group_filter: Optional[List[str]] = None):
    with engine.begin() as conn:
        stmt = select(tasks_tbl)
        if group_filter:
            stmt = stmt.where(tasks_tbl.c.group_name.in_(group_filter))
        df = pd.read_sql(stmt, conn)
    return df

# -----------------------------
# CSV parsing
# -----------------------------
def _infer_group_from_filename(name: str) -> Optional[str]:
    if not name:
        return None
    n = name.lower()
    if 'pcomp' in n:
        return 'PComp'
    if 'home' in n:
        return 'Home'
    if 'bh' in n:
        return 'BH&CAP'
    return None

def _canonicalize_group(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return None
    key = v.lower().replace(" ", "").replace("+", "&").replace("-", "")
    if key.startswith("bh") and "cap" in key:
        return "BH&CAP"
    if key in ("pcomp", "p-comp", "p_comp", "p.comp"):
        return "PComp"
    if key == "home":
        return "Home"
    return v

def parse_csv(file, source_name: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(file, skipinitialspace=True)
    raw_to_key = {c: str(c).strip().lower().replace(" ", "").replace("_", "") for c in df.columns}
    df = df.rename(columns=raw_to_key)
    name_map = {
        "task": "title", "title": "title",
        "groupname": "group_name", "group": "group_name",
        "status": "status",
        "dependson": "depends_on", "depends": "depends_on", "dependson(optional)": "depends_on",
        "duedate": "due_date",
        "note": "notes", "notes": "notes",
        "id": "id",
        "description": "description",
    }
    df = df.rename(columns={k: v for k, v in name_map.items() if k in df.columns})
    required = ["title", "group_name", "status"]
    for col in required:
        if col not in df.columns:
            if col == "group_name":
                inferred = _infer_group_from_filename(source_name or getattr(file, "name", ""))
                if inferred is None:
                    raise ValueError("CSV missing required column: group_name (and could not infer from filename)")
                df["group_name"] = inferred
            else:
                raise ValueError(f"CSV missing required column: {col}")
    for col in ["description", "depends_on", "id", "due_date", "notes"]:
        if col not in df.columns:
            df[col] = None
    if "due_date" in df.columns:
        def parse_date(x):
            if pd.isna(x) or str(x).strip() == "":
                return None
            try:
                return pd.to_datetime(x).date()
            except Exception:
                return None
        df["due_date"] = df["due_date"].apply(parse_date)
    if "id" not in df.columns:
        df["id"] = None
    df["id"] = df["id"].apply(lambda x: str(uuid.uuid4()) if pd.isna(x) or str(x).strip() == "" else str(x))
    for col in ["title", "description", "group_name", "status", "depends_on", "notes"]:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": None})
    df["group_name"] = df["group_name"].apply(_canonicalize_group)
    mask_empty = df["group_name"].isna() | (df["group_name"] == "")
    if mask_empty.any():
        inferred = _infer_group_from_filename(source_name or getattr(file, "name", ""))
        if inferred:
            df.loc[mask_empty, "group_name"] = inferred
    return df

def priority_badge(p: Optional[str]) -> str:
    if not p:
        return ''
    color = {
        'Low': '#e5e7eb',
        'Medium': '#bfdbfe',
        'High': '#fdba74',
        'Critical': '#fecaca',
    }.get(p, '#eee')
    return f"<span style='background:{color}; padding:2px 6px; border-radius:6px; font-size:0.85em'>{p}</span>"

# -----------------------------
# Pages
# -----------------------------
def page_dashboard(engine, tasks_tbl):
    st.header("📊 Dashboard")
    groups = st.multiselect("Filter by group", ALLOWED_GROUPS, default=ALLOWED_GROUPS)
    df = list_tasks_df(engine, tasks_tbl, groups)
    if df.empty:
        st.info("No tasks yet. Add some from the 'Add Task' or 'CSV Import' pages.")
        return

    total = len(df)
    done = (df['status'] == 'Done').sum()
    inprog = (df['status'] == 'In Progress').sum()
    blocked = (df['status'] == 'Blocked').sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", total)
    c2.metric("Done", int(done))
    c3.metric("In Progress", int(inprog))
    c4.metric("Blocked", int(blocked))

    df['group_name'] = df['group_name'].apply(_canonicalize_group)
    if 'priority' not in df.columns:
        df['priority'] = None
    df['priority'] = df['priority'].fillna('Unspecified')
    PRIORITY_ORDER = ["Critical", "High", "Medium", "Low", "Unspecified"]

    status_order = ALLOWED_STATUSES
    status_cat = pd.Categorical(df['status'], categories=status_order, ordered=True)
    pivot_status = (
        df.assign(status=status_cat)
          .pivot_table(index='group_name', columns='status', values='id', aggfunc='count', fill_value=0)
          .reindex(index=ALLOWED_GROUPS, fill_value=0)
          .reindex(columns=status_order, fill_value=0)
    )
    pivot_status['Total'] = pivot_status.sum(axis=1)
    total_row_status = pd.DataFrame([pivot_status.sum()], index=['Total'])
    pivot_status_disp = pd.concat([pivot_status, total_row_status])
    st.subheader("Counts by Group × Status")
    st.dataframe(pivot_status_disp, use_container_width=True)

    status_long = pivot_status.reset_index().melt('group_name', var_name='status', value_name='count')
    heat_status = (
        alt.Chart(status_long)
        .mark_rect()
        .encode(
            x=alt.X('status:N', sort=status_order, title='Status'),
            y=alt.Y('group_name:N', sort=ALLOWED_GROUPS, title='Group'),
            color=alt.Color('count:Q', scale=alt.Scale(scheme='blues'), title='Count')
        )
    )
    text_status = (
        alt.Chart(status_long)
        .mark_text(size=12)
        .encode(
            x='status:N', y='group_name:N', text='count:Q',
            color=alt.condition(alt.datum.count > 0, alt.value('black'), alt.value('#777'))
        )
    )
    st.altair_chart(heat_status + text_status, use_container_width=True)

    prio_cat = pd.Categorical(df['priority'], categories=PRIORITY_ORDER, ordered=True)
    pivot_prio = (
        df.assign(priority=prio_cat)
          .pivot_table(index='group_name', columns='priority', values='id', aggfunc='count', fill_value=0)
          .reindex(index=ALLOWED_GROUPS, fill_value=0)
          .reindex(columns=PRIORITY_ORDER, fill_value=0)
    )
    pivot_prio['Total'] = pivot_prio.sum(axis=1)
    total_row_prio = pd.DataFrame([pivot_prio.sum()], index=['Total'])
    pivot_prio_disp = pd.concat([pivot_prio, total_row_prio])
    st.subheader("Counts by Group × Priority")
    st.dataframe(pivot_prio_disp, use_container_width=True)

    prio_long = pivot_prio.reset_index().melt('group_name', var_name='priority', value_name='count')
    heat_prio = (
        alt.Chart(prio_long)
        .mark_rect()
        .encode(
            x=alt.X('priority:N', sort=PRIORITY_ORDER, title='Priority'),
            y=alt.Y('group_name:N', sort=ALLOWED_GROUPS, title='Group'),
            color=alt.Color('count:Q', scale=alt.Scale(scheme='oranges'), title='Count')
        )
    )
    text_prio = (
        alt.Chart(prio_long)
        .mark_text(size=12)
        .encode(
            x='priority:N', y='group_name:N', text='count:Q',
            color=alt.condition(alt.datum.count > 0, alt.value('black'), alt.value('#777'))
        )
    )
    st.altair_chart(heat_prio + text_prio, use_container_width=True)

    st.subheader("Stacked bars by Group – Status")
    agg_status = df.groupby(['group_name', 'status']).size().reset_index(name='count')
    chart_status = (
        alt.Chart(agg_status)
        .mark_bar()
        .encode(
            x=alt.X('group_name:N', sort=ALLOWED_GROUPS, title='Group'),
            y=alt.Y('count:Q', title='Tasks'),
            color=alt.Color('status:N', sort=status_order, title='Status')
        )
    ).properties(height=240)
    st.altair_chart(chart_status, use_container_width=True)

    st.subheader("Stacked bars by Group – Priority")
    agg_prio = df.groupby(['group_name', 'priority']).size().reset_index(name='count')
    chart_prio = (
        alt.Chart(agg_prio)
        .mark_bar()
        .encode(
            x=alt.X('group_name:N', sort=ALLOWED_GROUPS, title='Group'),
            y=alt.Y('count:Q', title='Tasks'),
            color=alt.Color('priority:N', sort=PRIORITY_ORDER, title='Priority')
        )
    ).properties(height=240)
    st.altair_chart(chart_prio, use_container_width=True)

    st.subheader("Kanban View")
    cols = st.columns(len(ALLOWED_STATUSES))
    for i, status in enumerate(ALLOWED_STATUSES):
        with cols[i]:
            st.markdown(f"**{status}**")
            subset = df[df['status'] == status]
            for _, r in subset.iterrows():
                with st.container(border=True):
                    p_html = priority_badge(r.get('priority'))
                    st.markdown((f"**{r['title']}**  " + p_html), unsafe_allow_html=True)
                    if pd.notna(r.get('description')) and str(r.get('description')).strip():
                        st.caption(str(r['description']))
                    if pd.notna(r.get('notes')) and str(r.get('notes')).strip():
                        st.caption("📝 " + str(r['notes']))
                    st.caption(f"Group: {r['group_name']} | ID: {r['id']}")
                    new_status = st.selectbox(
                        "Update status",
                        ALLOWED_STATUSES,
                        index=ALLOWED_STATUSES.index(status),
                        key=f"status_{r['id']}",
                        label_visibility="collapsed",
                    )
                    if new_status != status:
                        with engine.begin() as conn:
                            conn.execute(
                                update(tasks_tbl)
                                .where(tasks_tbl.c.id == r['id'])
                                .values(status=new_status, updated_at=datetime.utcnow())
                            )
                        st.success("Status updated")
                        st.rerun()

    st.subheader("Edit Tasks (inline)")
    edit_cols = ['id', 'title', 'group_name', 'status', 'priority', 'due_date', 'depends_on', 'notes', 'description']
    for c in edit_cols:
        if c not in df.columns:
            df[c] = None
    df_edit = df[edit_cols].copy()
    df_edit['priority'] = df_edit['priority'].fillna('')
    df_edit = df_edit.set_index('id', drop=False)
    edited_df = st.data_editor(
        df_edit,
        key="edit_tasks",
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "id": st.column_config.TextColumn("ID", disabled=True),
            "group_name": st.column_config.SelectboxColumn("Group", options=ALLOWED_GROUPS),
            "status": st.column_config.SelectboxColumn("Status", options=ALLOWED_STATUSES),
            "priority": st.column_config.SelectboxColumn("Priority", options=[""] + PRIORITY_CHOICES),
            "due_date": st.column_config.DateColumn("Due date", format="YYYY-MM-DD"),
            "depends_on": st.column_config.TextColumn("Depends on (IDs)", help="Comma-separated task IDs"),
            "notes": st.column_config.TextColumn("Notes"),
            "description": st.column_config.TextColumn("Description"),
        },
        hide_index=True,
    )
    if st.button("💾 Save changes"):
        updates = []
        original = df_edit
        if 'id' in edited_df.columns:
            edited_df = edited_df.set_index('id', drop=False)
        editable_fields = ['title', 'group_name', 'status', 'priority', 'due_date', 'depends_on', 'notes', 'description']
        for tid, row in edited_df.iterrows():
            if tid not in original.index:
                continue
            changed = {}
            for col in editable_fields:
                new_val = row[col]
                old_val = original.loc[tid, col]
                if col in ['priority', 'depends_on', 'notes', 'description'] and (new_val == "" or pd.isna(new_val)):
                    new_val = None
                if col == 'due_date' and pd.notna(new_val):
                    if isinstance(new_val, pd.Timestamp):
                        new_val = new_val.date()
                    elif isinstance(new_val, datetime):
                        new_val = new_val.date()
                    elif isinstance(new_val, str):
                        try:
                            new_val = pd.to_datetime(new_val).date()
                        except Exception:
                            new_val = None
                if pd.isna(old_val) and new_val is None:
                    continue
                if (pd.isna(old_val) and new_val is not None) or (not pd.isna(old_val) and new_val is None) or (str(old_val) != str(new_val)):
                    changed[col] = new_val
            if changed:
                changed['updated_at'] = datetime.utcnow()
                changed['id'] = tid
                updates.append(changed)
        if not updates:
            st.info("No changes detected.")
        else:
            with engine.begin() as conn:
                for rec in updates:
                    tid = rec.pop('id')
                    conn.execute(
                        update(tasks_tbl)
                        .where(tasks_tbl.c.id == tid)
                        .values(**rec)
                    )
            st.success(f"Saved changes to {len(updates)} task(s).")
            st.rerun()

    st.subheader("All Tasks")
    show_cols = ['id', 'title', 'group_name', 'status', 'priority', 'due_date', 'depends_on', 'notes', 'created_at', 'updated_at']
    for c in show_cols:
        if c not in df.columns:
            df[c] = None
    st.dataframe(df[show_cols].sort_values('updated_at', ascending=False), use_container_width=True)

def page_add_task(engine, tasks_tbl):
    st.header("➕ Add Task")
    with st.form("add_task_form"):
        title = st.text_input("Title", max_chars=255)
        description = st.text_area("Description", height=100)
        group = st.selectbox("Group", ALLOWED_GROUPS)
        status = st.selectbox("Status", ALLOWED_STATUSES, index=0)
        priority = st.selectbox("Priority (optional)", [""] + PRIORITY_CHOICES, index=1)
        depends_on = st.text_input("Depends on (comma-separated task IDs)")
        due_date = st.date_input("Due date", value=None)
        notes = st.text_area("Notes (optional)", height=80)
        submitted = st.form_submit_button("Add")
        if submitted:
            rec = {
                'id': str(uuid.uuid4()),
                'title': (title or '').strip(),
                'description': (description or '').strip() or None,
                'group_name': group,
                'status': status,
                'priority': (priority or '').strip() or None,
                'depends_on': (depends_on or '').strip() or None,
                'due_date': due_date if isinstance(due_date, date) else None,
                'notes': (notes or '').strip() or None,
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow(),
            }
            if not rec['title']:
                st.error("Missing 'Title'")
            elif rec['group_name'] not in ALLOWED_GROUPS:
                st.error("Invalid group")
            elif rec['status'] not in ALLOWED_STATUSES:
                st.error("Invalid status")
            else:
                upsert_tasks(engine, tasks_tbl, [rec])
                st.success("Task added")
                st.rerun()

def page_csv_import(engine, tasks_tbl):
    st.header("📥 CSV Import (multi-file)")
    st.write("Upload CSVs with headers like: **Task, Group_Name, Status, Depends_on, Priority, Notes**.")
    files = st.file_uploader("Choose CSV file(s)", type=["csv"], accept_multiple_files=True)
    if not files:
        return
    frames = []
    errors = []
    for f in files:
        try:
            df = parse_csv(f, source_name=getattr(f, 'name', None))
            frames.append(df)
            st.success(f"Parsed {len(df)} rows from {getattr(f, 'name', 'file')}")
        except Exception as e:
            errors.append((getattr(f, 'name', 'file'), str(e)))
    if errors:
        st.error("Some files failed to parse:")
        for name, e in errors:
            st.write(f"- {name}: {e}")
        if not frames:
            return
    df_all = pd.concat(frames, ignore_index=True)
    row_errors = []
    for i, row in df_all.iterrows():
        if not row.get('title'):
            row_errors.append((i, "Missing title"))
        if row.get('group_name') not in ALLOWED_GROUPS:
            row_errors.append((i, f"Invalid group_name: {row.get('group_name')}"))
        if row.get('status') not in ALLOWED_STATUSES:
            row_errors.append((i, f"Invalid status: {row.get('status')}"))
    if row_errors:
        st.error("Validation issues:")
        for i, e in row_errors[:100]:
            st.write(f"Row {i}: {e}")
        return
    st.success(f"Ready to import {len(df_all)} tasks.")
    if st.button("Import tasks"):
        recs = df_all.to_dict(orient='records')
        now = datetime.utcnow()
        for r in recs:
            r['created_at'] = now
            r['updated_at'] = now
        try:
            upsert_tasks(engine, tasks_tbl, recs)
            st.success(f"Imported {len(recs)} tasks")
            st.rerun()
        except OperationalError as e:
            st.error(f"Database error: {e}")

def page_settings(engine, tasks_tbl):
    st.header("⚙️ Settings & Info")
    st.write("**Database URL:**")
    st.code(get_db_url())
    st.write("Auth mode:", AUTH_MODE)
    st.subheader("Danger Zone")
    if st.button("Delete ALL tasks", type="primary"):
        with engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM tasks")
        st.warning("All tasks deleted")

# -----------------------------
# Main app
# -----------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    if not check_login():
        st.stop()

    engine, tasks_tbl = get_engine_and_table()

    tabs = st.tabs(["Dashboard", "Add Task", "CSV Import", "Settings"])
    with tabs[0]:
        page_dashboard(engine, tasks_tbl)
    with tabs[1]:
        page_add_task(engine, tasks_tbl)
    with tabs[2]:
        page_csv_import(engine, tasks_tbl)
    with tabs[3]:
        page_settings(engine, tasks_tbl)

if __name__ == "__main__":
    main()
