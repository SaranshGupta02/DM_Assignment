import streamlit as st
import pandas as pd
import zipfile, os, tempfile, textwrap, traceback
from pathlib import Path
import concurrent.futures
import typing as t

# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(page_title="Code Postmortem Auditor", layout="wide")

st.title("🧠 Code Postmortem Auditor")
st.caption("Upload your project and CSV, enter API keys, and generate AI-driven code analysis reports.")

# ===========================
# API KEYS
# ===========================
with st.sidebar:
    st.header("🔑 API Keys")
    OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")
    GEMINI_API_KEY = st.text_input("Gemini API Key", type="password")
    CEREBRAS_API_KEY = st.text_input("Cerebras API Key", type="password")

# ===========================
# FILE INPUTS
# ===========================
st.subheader("📁 Upload Files")

csv_file = st.file_uploader("Upload CSV file", type=["csv"])
zip_file = st.file_uploader("Upload ZIP file (project source code)", type=["zip"])

# ===========================
# SYSTEM PROMPT
# ===========================
SYSTEM_PROMPT = st.text_area(
    "🧩 System Prompt",
    value="""You are an expert software auditor, code reviewer, and performance engineer.
Your role is to perform a complete postmortem analysis of the given code.
## Instruction
  - You must think like a professional auditor, focusing on correctness, performance, and security.
  - Your analysis should identify potential issues, weaknesses, and improvement areas related to efficiency, reliability, maintainability, and safety.
  - Do not use bullet points, tables, markdown formatting, or emojis.
  - Write the report entirely in well-structured, coherent paragraphs, maintaining a formal and technical tone.""",
    height=250,
)

# Allow user to customize the per-file user prompt template
USER_PROMPT_TEMPLATE = st.text_area(
    "🗣️ User Prompt Template",
    value=(
        "You are given a source code file. Analyze it thoroughly and write a single, detailed postmortem report.\n"
        "source code:\n{code}\n"
    ),
    height=200,
)

# ===========================
# HELPER FUNCTIONS
# ===========================
def detect_path_column(df: pd.DataFrame) -> str:
    candidates = []
    for col in df.columns:
        try:
            mask = df[col].astype(str).str.contains(r"\.java\b", case=False, na=False)
            if mask.any():
                candidates.append((col, mask.sum()))
        except Exception:
            pass
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    for col in df.columns:
        name = col.lower()
        if any(k in name for k in ["path", "file", "filepath", "location", "relative"]):
            return col
    return df.columns[0]

def read_text_safe(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        with open(p, "rb") as f:
            return f.read().decode("utf-8", errors="ignore")

def resolve_path(raw_value: t.Any, project_root: Path) -> Path:
    s = str(raw_value).strip().replace("\\", "/")
    s = s.lstrip("./").lstrip("/")
    return (project_root / s).resolve()

def _strip_wrapper_quotes(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    if s.startswith("'''") and s.endswith("'''"):
        return s[3:-3].strip()
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1].strip()
    return s

# ===========================
# MAIN PROCESSING
# ===========================
if csv_file and zip_file:
    if st.button("🚀 Run Postmortem Analysis"):
        with st.spinner("Extracting and analyzing... please wait"):
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, "project.zip")
            csv_path = os.path.join(temp_dir, "input.csv")

            with open(zip_path, "wb") as f:
                f.write(zip_file.read())
            with open(csv_path, "wb") as f:
                f.write(csv_file.read())

            DEST = Path(temp_dir) / "project"
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(DEST)

            df_in = pd.read_csv(csv_path)
            st.write("✅ CSV Loaded:", df_in.shape)
            st.dataframe(df_in.head(3))

            col = detect_path_column(df_in)
            st.info(f"Using CSV path column: `{col}`")

            from langchain_openai import ChatOpenAI
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_cerebras.chat_models import ChatCerebras
            from langchain.schema import SystemMessage, HumanMessage

            llm_openai = ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=OPENAI_API_KEY)
            llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2, api_key=GEMINI_API_KEY)
            if(CEREBRAS_API_KEY):
                llm_cerebras = ChatCerebras(model="llama3.1-70b", api_key=CEREBRAS_API_KEY, temperature=0.2)

            results = []

            progress = st.progress(0)
            total = len(df_in)
            log_box = st.expander("Row creation logs", expanded=True)

            for i, row in df_in.iterrows():
                raw_path = row[col]
                file_path = resolve_path(raw_path, DEST)
                found = file_path.exists() and file_path.suffix.lower() == ".java"
                if not found:
                    continue

                file_text = read_text_safe(file_path)
                if "{code}" in USER_PROMPT_TEMPLATE:
                    user_prompt = USER_PROMPT_TEMPLATE.replace("{code}", file_text)
                else:
                    user_prompt = f"{USER_PROMPT_TEMPLATE}\n\n{file_text}"

                def call_model(fn, llm, sys, usr):
                    try:
                        resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=usr)])
                        return _strip_wrapper_quotes(resp.content)
                    except Exception as e:
                        return f"[ERROR] {type(e).__name__}: {e}"

                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    fut_openai = executor.submit(call_model, "openai", llm_openai, SYSTEM_PROMPT, user_prompt)
                    fut_gemini = executor.submit(call_model, "gemini", llm_gemini, SYSTEM_PROMPT, user_prompt)
                    if(CEREBRAS_API_KEY):
                        fut_cerebras = executor.submit(call_model, "cerebras", llm_cerebras, SYSTEM_PROMPT, user_prompt)

                    try:
                        openai_res = fut_openai.result(timeout=60)
                        gemini_res = fut_gemini.result(timeout=60)
                        if(CEREBRAS_API_KEY):
                            cerebras_res = fut_cerebras.result(timeout=60)
                    except Exception as e:
                        openai_res = gemini_res = cerebras_res = f"[TIMEOUT] {e}"
                if(CEREBRAS_API_KEY):
                    results.append({
                        "csv_row": i,
                        "csv_path_value": raw_path,
                        "resolved_path": str(file_path),
                        "openai": openai_res,
                        "gemini": gemini_res,
                        "cerebras": cerebras_res,
                    })
                else:
                    results.append({
                        "csv_row": i,
                        "csv_path_value": raw_path,
                        "resolved_path": str(file_path),
                        "openai": openai_res,
                        "gemini": gemini_res,
                    })

                with log_box:
                    st.write({
                        "created_index": len(results) - 1,
                        "csv_row": i,
                        "csv_path_value": raw_path,
                        "resolved_path": str(file_path),
                    })

                progress.progress((i + 1) / total)

            df_out = pd.DataFrame(results)
            st.success("✅ Analysis complete!")
            st.dataframe(df_out.head())

            csv_out = os.path.join(temp_dir, "postmortem_results.csv")
            df_out.to_csv(csv_out, index=False)
            with open(csv_out, "rb") as f:
                st.download_button("📥 Download Results CSV", data=f, file_name="postmortem_results.csv")

else:
    st.warning("Please provide all API keys and upload both CSV and ZIP files to proceed.")
