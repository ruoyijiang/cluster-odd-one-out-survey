import os
import time
import hashlib
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

try:
    import gspread
    from google.oauth2.service_account import Credentials
except ImportError:
    gspread = None
    Credentials = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRIALS_CSV = os.path.join(
    BASE_DIR,
    "odd_one_out_trials_with_both_image_types.csv"
)

RESPONSES_CSV = os.path.join(
    BASE_DIR,
    "physician_responses.csv"
)

PARTICIPANT_IDS = ["1", "2", "3", "4", "5"]
PARTICIPANT_PLACEHOLDER = "Select participant"

st.set_page_config(
    page_title="Odd-One-Out Image Evaluation",
    layout="wide"
)

# ----------------------------
# load trials
# ----------------------------
@st.cache_data
def load_trials():
    df = pd.read_csv(TRIALS_CSV)

    # convert old Misha absolute paths to local paths
    for label in ["A", "B", "C"]:
        for key in ["center_cells", "gen_imgs"]:
            col = f"image_{label}_{key}_path"
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda p: os.path.join(BASE_DIR, "images", os.path.basename(str(p)))
                    if pd.notna(p) else p
                )

    return df


trials_df = load_trials()

# ----------------------------
# session setup
# ----------------------------
if "participant_id" not in st.session_state:
    st.session_state.participant_id = ""

if "active_participant_id" not in st.session_state:
    st.session_state.active_participant_id = None

if "trial_idx" not in st.session_state:
    st.session_state.trial_idx = 0

if "answers" not in st.session_state:
    st.session_state.answers = {}

if "answer_rows" not in st.session_state:
    st.session_state.answer_rows = {}

if "next_sheet_row" not in st.session_state:
    st.session_state.next_sheet_row = 2

if "trial_order" not in st.session_state:
    st.session_state.trial_order = trials_df.index.tolist()

if "pending_jump_value" not in st.session_state:
    st.session_state.pending_jump_value = None

if "pending_sync_trial_ids" not in st.session_state:
    st.session_state.pending_sync_trial_ids = []

if "cloud_sync_warning" not in st.session_state:
    st.session_state.cloud_sync_warning = None


def trial_order_for_participant(pid):
    seed = int(hashlib.sha256(str(pid).encode("utf-8")).hexdigest()[:8], 16)
    return trials_df.sample(
        frac=1,
        random_state=seed
    ).index.tolist()


def build_response_row(pid, trial_number, trial_row, selected_answer=""):
    correct_answer = trial_row["correct_answer"]
    has_answer = selected_answer in {"A", "B", "C"}
    return {
        "timestamp": datetime.now().isoformat() if has_answer else "",
        "participant_id": str(pid),
        "trial_number": trial_number,
        "trial_id": int(trial_row["trial_id"]),
        "cluster_pair": trial_row["cluster_pair"],
        "anchor_cluster": trial_row["anchor_cluster"],
        "distractor_cluster": trial_row["distractor_cluster"],
        "selected_answer": selected_answer if has_answer else "",
        "correct_answer": correct_answer,
        "is_correct": str(selected_answer == correct_answer) if has_answer else "",
    }


def blank_response_rows_for_participant(pid, trial_order):
    return [
        build_response_row(pid, trial_number, trials_df.loc[row_idx])
        for trial_number, row_idx in enumerate(trial_order, start=1)
    ]


def row_to_values(row):
    return [str(row.get(column, "")) for column in get_response_columns()]


def get_gsheets_config():
    try:
        gsheets_config = st.secrets["connections"]["gsheets"]
    except Exception:
        return None

    if "spreadsheet" not in gsheets_config:
        return None

    return gsheets_config


def get_response_columns():
    return [
        "timestamp",
        "participant_id",
        "trial_number",
        "trial_id",
        "cluster_pair",
        "anchor_cluster",
        "distractor_cluster",
        "selected_answer",
        "correct_answer",
        "is_correct",
    ]


def get_storage_mode():
    if get_gsheets_config() and gspread is not None and Credentials is not None:
        return "gsheets"
    return "csv"


def gsheets_status_message():
    if not get_gsheets_config():
        return None
    if gspread is None or Credentials is None:
        return (
            "Google Sheets secrets are configured, but the required packages are not installed. "
            "Install dependencies from requirements.txt to enable cloud response storage."
        )
    return None


def set_cloud_sync_warning(message):
    st.session_state.cloud_sync_warning = message


def clear_cloud_sync_warning():
    st.session_state.cloud_sync_warning = None


def unique_trial_ids(trial_ids):
    seen = set()
    ordered = []
    for trial_id in trial_ids:
        if trial_id not in seen:
            ordered.append(trial_id)
            seen.add(trial_id)
    return ordered


def normalize_trial_id(value):
    if pd.isna(value) or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def mark_pending_sync(trial_ids):
    st.session_state.pending_sync_trial_ids = unique_trial_ids(
        st.session_state.pending_sync_trial_ids + list(trial_ids)
    )


def clear_pending_sync(trial_ids):
    cleared = set(trial_ids)
    st.session_state.pending_sync_trial_ids = [
        trial_id
        for trial_id in st.session_state.pending_sync_trial_ids
        if trial_id not in cleared
    ]


def run_gsheets_call(operation_name, callback, retries=3, base_delay=0.75):
    last_error = None
    for attempt in range(retries):
        try:
            result = callback()
            clear_cloud_sync_warning()
            return result
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(base_delay * (attempt + 1))

    set_cloud_sync_warning(
        f"Google Sheets sync is temporarily unavailable, so some responses may not be "
        f"saved to the shared sheet right away. You can keep going and use "
        f"'Download responses' as a backup."
    )
    return None


def build_gsheet_worksheet():
    gsheets_config = get_gsheets_config()
    if not gsheets_config or gspread is None or Credentials is None:
        return None

    credentials_info = {
        key: value
        for key, value in dict(gsheets_config).items()
        if key != "spreadsheet" and key != "worksheet"
    }
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = Credentials.from_service_account_info(
        credentials_info,
        scopes=scopes,
    )
    client = gspread.authorize(credentials)
    spreadsheet = client.open_by_url(gsheets_config["spreadsheet"])
    worksheet_name = gsheets_config.get("worksheet")
    if worksheet_name:
        return spreadsheet.worksheet(worksheet_name)
    return spreadsheet.sheet1


def get_gsheet_worksheet():
    return run_gsheets_call("connect to Google Sheets", build_gsheet_worksheet)


def read_responses_df():
    if get_storage_mode() == "gsheets":
        worksheet = get_gsheet_worksheet()
        if worksheet is None:
            return pd.DataFrame(columns=get_response_columns())

        records = run_gsheets_call("read responses from Google Sheets", worksheet.get_all_records)
        if records is None:
            return pd.DataFrame(columns=get_response_columns())
        if not records:
            return pd.DataFrame(columns=get_response_columns())

        response_df = pd.DataFrame(records)
        for column in get_response_columns():
            if column not in response_df.columns:
                response_df[column] = pd.NA
        return response_df[get_response_columns()]

    if os.path.exists(RESPONSES_CSV):
        response_df = pd.read_csv(RESPONSES_CSV)
        for column in get_response_columns():
            if column not in response_df.columns:
                response_df[column] = pd.NA
        return response_df[get_response_columns()]

    return pd.DataFrame(columns=get_response_columns())


def responses_for_participant(pid):
    response_df = read_responses_df()
    if response_df.empty:
        return response_df
    return response_df[response_df["participant_id"].astype(str) == str(pid)]


def load_saved_answers_and_rows(pid):
    if not pid:
        return {}, {}, 2

    if get_storage_mode() == "gsheets":
        worksheet = get_gsheet_worksheet()
        if worksheet is None:
            return {}, {}, 2

        records = run_gsheets_call("load saved responses from Google Sheets", worksheet.get_all_records)
        if records is None:
            return {}, {}, 2
        if not records:
            return {}, {}, 2

        saved_df = pd.DataFrame(records)
        if saved_df.empty:
            return {}, {}, 2

        saved_df["__row_number"] = range(2, len(saved_df) + 2)
        saved_df = saved_df[saved_df["participant_id"].astype(str) == str(pid)]
        if saved_df.empty:
            return {}, {}, len(records) + 2

        saved_df["__trial_id"] = saved_df["trial_id"].apply(normalize_trial_id)
        saved_df = saved_df.dropna(subset=["__trial_id"])
        saved_df = saved_df.drop_duplicates(subset=["__trial_id"], keep="last")
        saved_answer_df = saved_df[saved_df["selected_answer"].astype(str).isin(["A", "B", "C"])]
        answers = {
            int(answer["__trial_id"]): answer.drop(labels=["__row_number", "__trial_id"]).to_dict()
            for _, answer in saved_answer_df.iterrows()
        }
        answer_rows = {
            int(answer["__trial_id"]): int(answer["__row_number"])
            for _, answer in saved_df.iterrows()
        }
        return answers, answer_rows, len(records) + 2

    saved_df = read_responses_df()
    saved_df = saved_df[saved_df["participant_id"].astype(str) == str(pid)]
    if saved_df.empty:
        return {}, {}, 2

    saved_df["__trial_id"] = saved_df["trial_id"].apply(normalize_trial_id)
    saved_df = saved_df.dropna(subset=["__trial_id"])
    saved_df = saved_df.drop_duplicates(subset=["__trial_id"], keep="last")
    saved_answer_df = saved_df[saved_df["selected_answer"].astype(str).isin(["A", "B", "C"])]
    answers = {
        int(answer["__trial_id"]): answer.drop(labels="__trial_id").to_dict()
        for _, answer in saved_answer_df.iterrows()
    }
    return answers, {}, 2


def write_responses_df(response_df):
    ordered_df = response_df.copy()
    if not ordered_df.empty:
        ordered_df = ordered_df[get_response_columns()]
        ordered_df = ordered_df.sort_values(["participant_id", "trial_number", "trial_id"])

    if get_storage_mode() == "gsheets":
        worksheet = get_gsheet_worksheet()
        if worksheet is None:
            return

        rows = [get_response_columns()]
        if not ordered_df.empty:
            safe_df = ordered_df.fillna("")
            rows.extend(safe_df.astype(str).values.tolist())
        result = run_gsheets_call(
            "rewrite responses in Google Sheets",
            lambda: (worksheet.clear(), worksheet.update(rows))
        )
        if result is None:
            return
        return

    ordered_df.to_csv(RESPONSES_CSV, index=False)


def ensure_participant_rows(pid, trial_order, answer_rows):
    if get_storage_mode() != "gsheets":
        return answer_rows, None

    worksheet = get_gsheet_worksheet()
    if worksheet is None:
        return answer_rows, None

    missing_rows = [
        row
        for row in blank_response_rows_for_participant(pid, trial_order)
        if row["trial_id"] not in answer_rows
    ]
    if not missing_rows:
        return answer_rows, None

    if not answer_rows and st.session_state.next_sheet_row <= 2:
        header_result = run_gsheets_call(
            "ensure Google Sheets response header",
            lambda: worksheet.update("A1:J1", [get_response_columns()])
        )
        if header_result is None:
            return answer_rows, None

    start_row = st.session_state.next_sheet_row
    values = [row_to_values(row) for row in missing_rows]
    result = run_gsheets_call(
        "prefill participant response rows in Google Sheets",
        lambda values=values: worksheet.append_rows(values, value_input_option="USER_ENTERED")
    )
    if result is None:
        return answer_rows, None

    updated_rows = dict(answer_rows)
    for offset, row in enumerate(missing_rows):
        updated_rows[row["trial_id"]] = start_row + offset

    return updated_rows, start_row + len(missing_rows)


def load_saved_answers(pid):
    answers, _, _ = load_saved_answers_and_rows(pid)
    return answers


def next_trial_index(trial_order, answers):
    for idx, row_idx in enumerate(trial_order):
        candidate_trial_id = int(trials_df.loc[row_idx, "trial_id"])
        if candidate_trial_id not in answers:
            return idx
    return max(len(trial_order) - 1, 0)


def initialize_answer_widget_state(pid, answers):
    answer_to_display = {"A": "1", "B": "2", "C": "3"}
    for trial_id, answer in answers.items():
        selected_answer = answer.get("selected_answer")
        display_answer = answer_to_display.get(selected_answer)
        if display_answer:
            st.session_state[f"choice_{pid}_{trial_id}"] = display_answer


def reset_for_participant(pid):
    trial_order = trial_order_for_participant(pid)
    answers, answer_rows, next_sheet_row = load_saved_answers_and_rows(pid)

    st.session_state.participant_id = pid
    st.session_state.active_participant_id = pid
    st.session_state.answers = answers
    st.session_state.answer_rows = answer_rows
    st.session_state.next_sheet_row = next_sheet_row
    st.session_state.trial_order = trial_order
    updated_answer_rows, updated_next_sheet_row = ensure_participant_rows(pid, trial_order, answer_rows)
    st.session_state.answer_rows = updated_answer_rows
    if updated_next_sheet_row is not None:
        st.session_state.next_sheet_row = updated_next_sheet_row
    st.session_state.trial_idx = next_trial_index(trial_order, answers)
    st.session_state.pending_jump_value = None
    st.session_state.pending_sync_trial_ids = []
    initialize_answer_widget_state(pid, answers)


# ----------------------------
# sidebar
# ----------------------------
participant_options = [PARTICIPANT_PLACEHOLDER] + PARTICIPANT_IDS
default_participant = (
    st.session_state.participant_id
    if st.session_state.participant_id in PARTICIPANT_IDS
    else PARTICIPANT_PLACEHOLDER
)

participant_choice = st.sidebar.selectbox(
    "Participant ID",
    participant_options,
    index=participant_options.index(default_participant),
    format_func=lambda value: PARTICIPANT_PLACEHOLDER if value == PARTICIPANT_PLACEHOLDER else f"Participant {value}"
)

participant_id = "" if participant_choice == PARTICIPANT_PLACEHOLDER else participant_choice
st.session_state.participant_id = participant_id

if participant_id and participant_id != st.session_state.active_participant_id:
    reset_for_participant(participant_id)
    st.rerun()

if not participant_id:
    st.title("Odd-One-Out Image Evaluation")
    st.info("Select one of the five participants in the sidebar to begin.")
    st.stop()

storage_warning = gsheets_status_message()
if storage_warning:
    st.warning(storage_warning)
elif st.session_state.cloud_sync_warning:
    st.warning(st.session_state.cloud_sync_warning)

st.sidebar.write(f"Responses saved to:")
if get_storage_mode() == "gsheets":
    st.sidebar.code("Google Sheets")
else:
    st.sidebar.code(RESPONSES_CSV)

# ----------------------------
# current trial
# ----------------------------
trial_order = st.session_state.trial_order
trial_idx = st.session_state.trial_idx
n_trials = len(trial_order)
jump_key = f"jump_to_trial_{participant_id}"

if st.session_state.pending_jump_value is not None:
    st.session_state[jump_key] = st.session_state.pending_jump_value
    st.session_state.pending_jump_value = None

row = trials_df.loc[trial_order[trial_idx]]
trial_id = int(row["trial_id"])

st.title("Odd-One-Out Image Evaluation")

st.write(
    "For each option, only the generated image is shown. "
    "Select the option that looks most different from the other two."
)

st.progress((trial_idx + 1) / n_trials)
st.write(f"Trial {trial_idx + 1} of {n_trials}")

jump_to_trial = st.slider(
    "Jump to trial",
    min_value=1,
    max_value=n_trials,
    value=trial_idx + 1,
    key=jump_key,
)

# ----------------------------
# display images
# ----------------------------
image_labels = ["A", "B", "C"]
display_choices = ["1", "2", "3"]
answer_to_display = dict(zip(image_labels, display_choices))
display_to_answer = dict(zip(display_choices, image_labels))
outer_cols = st.columns(3)
IMAGE_WIDTH = 190

for outer_col, image_label, display_label in zip(outer_cols, image_labels, display_choices):
    with outer_col:
        st.subheader(display_label)

        gen_path = row[f"image_{image_label}_gen_imgs_path"]

        st.caption("Generated image")
        if pd.isna(gen_path) or not os.path.exists(gen_path):
            st.error("Missing")
        else:
            st.image(Image.open(gen_path), width=IMAGE_WIDTH)

# ----------------------------
# retrieve previous answer if going back
# ----------------------------
prev = st.session_state.answers.get(trial_id, {})

default_answer = prev.get("selected_answer")
default_choice = answer_to_display.get(default_answer)
choice_key = f"choice_{participant_id}_{trial_id}"

selected = st.radio(
    "Which option is the odd one out?",
    display_choices,
    index=display_choices.index(default_choice) if default_choice else None,
    horizontal=True,
    key=choice_key,
)

# ----------------------------
# save current answer to session
# ----------------------------
def save_current_answer():
    if selected is None:
        return

    selected_answer = display_to_answer[selected]
    st.session_state.answers[trial_id] = build_response_row(
        participant_id,
        trial_idx + 1,
        row,
        selected_answer=selected_answer,
    )


def write_all_answers():
    new_df = pd.DataFrame(list(st.session_state.answers.values()))

    if len(new_df) == 0:
        return

    old_df = read_responses_df()
    old_df = old_df[old_df["participant_id"].astype(str) != str(participant_id)]
    out_df = pd.concat([old_df, new_df], ignore_index=True)

    write_responses_df(out_df)


def persist_current_answer():
    if selected is None:
        return

    if get_storage_mode() != "gsheets":
        write_all_answers()
        return

    worksheet = get_gsheet_worksheet()
    if worksheet is None:
        mark_pending_sync([trial_id])
        return

    trial_ids_to_sync = unique_trial_ids(st.session_state.pending_sync_trial_ids + [trial_id])
    synced_trial_ids = []

    for pending_trial_id in trial_ids_to_sync:
        saved_answer = st.session_state.answers.get(pending_trial_id)
        if not saved_answer:
            mark_pending_sync([pending_trial_id])
            break

        row_values = row_to_values(saved_answer)
        row_number = st.session_state.answer_rows.get(pending_trial_id)

        if not row_number:
            updated_answer_rows, updated_next_sheet_row = ensure_participant_rows(
                participant_id,
                st.session_state.trial_order,
                st.session_state.answer_rows,
            )
            st.session_state.answer_rows = updated_answer_rows
            if updated_next_sheet_row is not None:
                st.session_state.next_sheet_row = updated_next_sheet_row
            row_number = st.session_state.answer_rows.get(pending_trial_id)

        if not row_number:
            mark_pending_sync([pending_trial_id])
            break

        result = run_gsheets_call(
            "update a response in Google Sheets",
            lambda row_number=row_number, row_values=row_values: (
                worksheet.update(f"A{row_number}:J{row_number}", [row_values])
            )
        )
        if result is None:
            break

        synced_trial_ids.append(pending_trial_id)

    if synced_trial_ids:
        clear_pending_sync(synced_trial_ids)
    if trial_id not in synced_trial_ids:
        mark_pending_sync([trial_id])


def current_participant_export():
    export_df = responses_for_participant(participant_id)
    if export_df.empty:
        export_df = pd.DataFrame(list(st.session_state.answers.values()))
    if export_df.empty:
        return b""

    export_df = export_df.sort_values(["trial_number", "trial_id"])
    return export_df.to_csv(index=False).encode("utf-8")


current_selected_answer = display_to_answer[selected] if selected else None
if current_selected_answer and prev.get("selected_answer") != current_selected_answer:
    save_current_answer()
    persist_current_answer()

if jump_to_trial != trial_idx + 1:
    save_current_answer()
    persist_current_answer()
    st.session_state.trial_idx = jump_to_trial - 1
    st.rerun()


def enable_keyboard_shortcuts():
    components.html(
        """
        <script>
        const doc = window.parent.document;
        const win = window.parent;

        if (win.__oddOneOutKeyHandler) {
          win.removeEventListener("keydown", win.__oddOneOutKeyHandler, true);
        }

        const isTypingTarget = (target) => {
          if (!target) return false;
          const tag = target.tagName;
          if (tag === "TEXTAREA" || target.isContentEditable) {
            return true;
          }
          if (tag === "INPUT") {
            const type = (target.type || "").toLowerCase();
            return ["text", "password", "email", "search", "tel", "url", "number"].includes(type);
          }
          return false;
        };

        const clickButtonByText = (label) => {
          const buttons = Array.from(doc.querySelectorAll("button"));
          const match = buttons.find(
            (button) => button.innerText && button.innerText.trim() === label
          );
          if (match) {
            match.click();
            return true;
          }
          return false;
        };

        const clickRadioByText = (label) => {
          const radios = Array.from(doc.querySelectorAll('label[data-baseweb="radio"]'));
          const match = radios.find(
            (radio) => radio.innerText && radio.innerText.trim() === label
          );
          if (match) {
            match.click();
            return true;
          }
          return false;
        };

        win.__oddOneOutKeyHandler = (event) => {
          if (isTypingTarget(event.target)) {
            return;
          }

          const key = event.key.toLowerCase();

          if (["1", "2", "3"].includes(key)) {
            if (clickRadioByText(key)) {
              event.preventDefault();
              event.stopPropagation();
            }
            return;
          }

          if (key === "j") {
            event.preventDefault();
            event.stopPropagation();
            if (clickButtonByText("Back")) {
              event.stopImmediatePropagation();
            }
            return;
          }

          if (key === "k") {
            event.preventDefault();
            event.stopPropagation();
            if (clickButtonByText("Continue") || clickButtonByText("Finish")) {
              event.stopImmediatePropagation();
            }
          }
        };

        win.addEventListener("keydown", win.__oddOneOutKeyHandler, true);
        </script>
        """,
        height=0,
    )


# ----------------------------
# navigation buttons
# ----------------------------
left, middle, right = st.columns([1, 1, 1])

with left:
    if st.button("Back", disabled=trial_idx == 0):
        save_current_answer()
        persist_current_answer()
        st.session_state.trial_idx -= 1
        st.session_state.pending_jump_value = st.session_state.trial_idx + 1
        st.rerun()

with middle:
    if st.button("Save progress"):
        save_current_answer()
        persist_current_answer()
        st.success("Progress saved.")

with right:
    next_label = "Finish" if trial_idx + 1 == n_trials else "Continue"

    if st.button(next_label):
        save_current_answer()
        persist_current_answer()

        if trial_idx + 1 < n_trials:
            st.session_state.trial_idx += 1
            st.session_state.pending_jump_value = st.session_state.trial_idx + 1
            st.rerun()
        else:
            st.success("Evaluation complete. Thank you.")

st.caption("Keyboard shortcuts: press 1/2/3 to choose, J for Back, K for Continue.")
enable_keyboard_shortcuts()

is_complete = len(st.session_state.answers) == n_trials
if st.session_state.answers:
    st.download_button(
        "Download responses",
        data=current_participant_export(),
        file_name=f"participant_{participant_id}_responses.csv",
        mime="text/csv",
    )

if is_complete:
    st.success("Evaluation complete. Thank you.")

# ----------------------------
# optional progress summary
# ----------------------------
st.sidebar.write("Progress")
st.sidebar.write(f"{len(st.session_state.answers)} / {n_trials} answered")
