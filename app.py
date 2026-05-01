import os
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

PARTICIPANT_IDS = ["1", "2", "3", "4"]
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


def trial_order_for_participant(pid):
    return trials_df.sample(
        frac=1,
        random_state=abs(hash(pid)) % (2**32)
    ).index.tolist()


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


def get_gsheet_worksheet():
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


def read_responses_df():
    if get_storage_mode() == "gsheets":
        worksheet = get_gsheet_worksheet()
        if worksheet is None:
            return pd.DataFrame(columns=get_response_columns())

        records = worksheet.get_all_records()
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


def load_saved_answers_and_rows(pid):
    if not pid:
        return {}, {}, 2

    if get_storage_mode() == "gsheets":
        worksheet = get_gsheet_worksheet()
        if worksheet is None:
            return {}, {}, 2

        records = worksheet.get_all_records()
        if not records:
            return {}, {}, 2

        saved_df = pd.DataFrame(records)
        if saved_df.empty:
            return {}, {}, 2

        saved_df["__row_number"] = range(2, len(saved_df) + 2)
        saved_df = saved_df[saved_df["participant_id"].astype(str) == str(pid)]
        if saved_df.empty:
            return {}, {}, len(records) + 2

        saved_df = saved_df.drop_duplicates(subset=["trial_id"], keep="last")
        answers = {
            int(answer["trial_id"]): answer.drop(labels="__row_number").to_dict()
            for _, answer in saved_df.iterrows()
        }
        answer_rows = {
            int(answer["trial_id"]): int(answer["__row_number"])
            for _, answer in saved_df.iterrows()
        }
        return answers, answer_rows, len(records) + 2

    saved_df = read_responses_df()
    saved_df = saved_df[saved_df["participant_id"].astype(str) == str(pid)]
    if saved_df.empty:
        return {}, {}, 2

    saved_df = saved_df.drop_duplicates(subset=["trial_id"], keep="last")
    answers = {
        int(answer["trial_id"]): answer.to_dict()
        for _, answer in saved_df.iterrows()
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
        worksheet.clear()
        worksheet.update(rows)
        return

    ordered_df.to_csv(RESPONSES_CSV, index=False)


def load_saved_answers(pid):
    answers, _, _ = load_saved_answers_and_rows(pid)
    return answers


def next_trial_index(trial_order, answers):
    for idx, row_idx in enumerate(trial_order):
        candidate_trial_id = int(trials_df.loc[row_idx, "trial_id"])
        if candidate_trial_id not in answers:
            return idx
    return max(len(trial_order) - 1, 0)


def reset_for_participant(pid):
    trial_order = trial_order_for_participant(pid)
    answers, answer_rows, next_sheet_row = load_saved_answers_and_rows(pid)

    st.session_state.participant_id = pid
    st.session_state.active_participant_id = pid
    st.session_state.answers = answers
    st.session_state.answer_rows = answer_rows
    st.session_state.next_sheet_row = next_sheet_row
    st.session_state.trial_order = trial_order
    st.session_state.trial_idx = next_trial_index(trial_order, answers)


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
    st.info("Select one of the four participants in the sidebar to begin.")
    st.stop()

storage_warning = gsheets_status_message()
if storage_warning:
    st.warning(storage_warning)

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

row = trials_df.loc[trial_order[trial_idx]]
trial_id = int(row["trial_id"])

st.title("Odd-One-Out Image Evaluation")

st.write(
    "For each option, only the generated image is shown. "
    "Select the option that looks most different from the other two."
)

st.progress((trial_idx + 1) / n_trials)
st.write(f"Trial {trial_idx + 1} of {n_trials}")

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

selected = st.radio(
    "Which option is the odd one out?",
    display_choices,
    index=display_choices.index(default_choice) if default_choice else None,
    horizontal=True,
    key=f"choice_{trial_id}"
)

# ----------------------------
# save current answer to session
# ----------------------------
def save_current_answer():
    if selected is None:
        return

    selected_answer = display_to_answer[selected]
    st.session_state.answers[trial_id] = {
        "timestamp": datetime.now().isoformat(),
        "participant_id": participant_id,
        "trial_number": trial_idx + 1,
        "trial_id": trial_id,
        "cluster_pair": row["cluster_pair"],
        "anchor_cluster": row["anchor_cluster"],
        "distractor_cluster": row["distractor_cluster"],
        "selected_answer": selected_answer,
        "correct_answer": row["correct_answer"],
        "is_correct": selected_answer == row["correct_answer"],
    }


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
        return

    saved_answer = st.session_state.answers.get(trial_id)
    if not saved_answer:
        return

    row_values = [str(saved_answer.get(column, "")) for column in get_response_columns()]
    row_number = st.session_state.answer_rows.get(trial_id)

    if row_number:
        worksheet.update(f"A{row_number}:J{row_number}", [row_values])
        return

    next_row = st.session_state.next_sheet_row
    worksheet.append_row(row_values)
    st.session_state.answer_rows[trial_id] = next_row
    st.session_state.next_sheet_row += 1


def current_participant_export():
    export_df = pd.DataFrame(list(st.session_state.answers.values()))
    if export_df.empty:
        return b""

    export_df = export_df.sort_values(["trial_number", "trial_id"])
    return export_df.to_csv(index=False).encode("utf-8")


current_selected_answer = display_to_answer[selected] if selected else None
if current_selected_answer and prev.get("selected_answer") != current_selected_answer:
    save_current_answer()
    persist_current_answer()


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
            if (clickButtonByText("Submit and continue") || clickButtonByText("Finish")) {
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
        st.rerun()

with middle:
    if st.button("Save progress"):
        save_current_answer()
        persist_current_answer()
        st.success("Progress saved.")

with right:
    next_label = "Finish" if trial_idx + 1 == n_trials else "Submit and continue"

    if st.button(next_label, disabled=selected is None):
        save_current_answer()
        persist_current_answer()

        if trial_idx + 1 < n_trials:
            st.session_state.trial_idx += 1
            st.rerun()
        else:
            st.success("Evaluation complete. Thank you.")

st.caption("Keyboard shortcuts: press 1/2/3 to choose, J for Back, K for Submit and continue.")
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
