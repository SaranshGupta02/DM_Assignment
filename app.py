import streamlit as st
import pandas as pd
import zipfile, os, tempfile, textwrap, traceback
from pathlib import Path
import typing as t
from supabase import create_client

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
    GEMINI_API_KEY = st.text_input("Gemini API Key", type="password")
    GEMINI_MODEL = st.selectbox("Gemini model", ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"], index=0)
    SUPABASE_URL = st.text_input("SUPABASE_URL", type="password")
    SUPABASE_KEY = st.text_input("SUPABASE_KEY", type="password")
    SUPABASE_TABLE = st.text_input("SUPABASE_TABLE_Name", value="postmortem_results")
    st.warning("Free keys may only allow ~150 rows per session. If you hit limits, switch keys or upgrade.")
    with st.expander("How to get API keys", expanded=False):
        st.markdown(
            (
                "- **Google Gemini**: Create a key in Google AI Studio → [Google AI Studio](https://aistudio.google.com)\n\n"
                "Keep your keys secret and rotate if rate limits are hit."
            )
        )
    st.header("💾 Output")
    AUTOSAVE_DIR_INPUT = st.text_input("Autosave folder", value=str(Path.cwd() / "results"))
    try:
        AUTOSAVE_DIR = Path(AUTOSAVE_DIR_INPUT)
        AUTOSAVE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        AUTOSAVE_DIR = Path.cwd()

# ===========================
# FILE INPUTS
# ===========================
st.subheader("📁 Upload Files")

csv_file = st.file_uploader("Upload CSV file", type=["csv"])
zip_file = st.file_uploader("Upload ZIP file (project source code)", type=["zip"])

# Row selection for processing subset of CSV rows
col_row1, col_row2 = st.columns(2)
with col_row1:
    START_ROW = st.number_input("Start row (inclusive)", min_value=0, value=0, step=1)
with col_row2:
    END_ROW = st.number_input("End row (exclusive, 0 = all)", min_value=0, value=0, step=1)

# Supabase setup instructions and helper
with st.expander("Supabase setup and table schema", expanded=False):
    st.markdown(
        """
        1. In Supabase project, go to Settings → API and copy Project URL and Service Role Key.
        2. Paste them in the sidebar inputs SUPABASE_URL and SUPABASE_KEY. Table defaults to `postmortem_results`.
        3. Create the table if it doesn't exist using SQL:

        ```sql
        create table if not exists postmortem_results (
          id bigserial primary key,
          created_at timestamptz default now(),
          csv_row bigint,
          csv_path_value text,
          resolved_path text,
          gemini text
        );
        ```
        """
    )

# ===========================
# FIXED PROMPT TEMPLATE
# ===========================
USER_PROMPT_TEMPLATE = """You are given a source code file. Your task is to generate a Postmortem Analysis Report following the structure below. The report must be written entirely in professional paragraph form, without bullet points or numbered lists. Maintain a technical, objective, and concise tone throughout.

FILE_PATH: {file_path}

--- BEGIN SOURCE CODE ---

{code}

--- END SOURCE CODE ---

If the file is missing, clearly state that the file was not found and no analysis can be performed.

If the file is present, perform a complete and detailed postmortem analysis of the code and provide the following sections written in cohesive paragraphs:

Overview: Explain what the code does, its programming language, and its main purpose.

Code Quality: Discuss the readability, structure, style, and adherence to standard best practices. Mention whether the naming conventions, indentation, and logical flow are consistent and clear.

Logic and Functional Analysis: Describe the logical soundness of the implementation. Identify any functional errors, redundant computations, or misleading assumptions. Explain how the logic behaves under typical and edge cases.

Performance Analysis: Examine computational efficiency, resource usage, and scalability. Highlight potential bottlenecks such as nested loops, repeated computations, or inefficient data structures, and suggest possible improvements.

Security Analysis: Assess the presence of security vulnerabilities including unsafe user input handling, injection risks, insecure file or database access, and exposed credentials. Discuss how well the code mitigates these risks.

Error Handling and Robustness: Evaluate the code's ability to handle exceptions and invalid inputs gracefully. Mention whether there are sufficient safeguards to prevent unexpected crashes or undefined behavior.

Maintainability and Scalability: Analyze the modularity, extensibility, and long-term maintainability of the code. Comment on whether it is easily adaptable to future changes or increasing data volumes.

Critical Issues Summary: Present a short paragraph summarizing the major issues discovered, their relative severity (low, medium, high, or critical), and general recommendations for correction.

Final Verdict: Conclude with a professional statement summarizing the overall reliability, safety, and production readiness of the code, ending with one of three ratings —   Safe,   Needs Fixes, or   Critical Problems Found — followed by a concise justification."""

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

def resolve_path(raw_value: t.Any, project_root: Path, csv_file_name: str) -> Path:
    """
    Returns a path like: project_root / csv_file_name_without_ext / raw_value
    """
    # Normalize raw_value
    s = str(raw_value).strip().replace("\\", "/")
    s = s.lstrip("./").lstrip("/")

    # Remove .csv from CSV file name
    csv_name_no_ext = Path(csv_file_name).stem

    # Build final path
    final_path = (project_root / csv_name_no_ext / s).resolve()
    return final_path

def _strip_wrapper_quotes(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    if s.startswith("'''") and s.endswith("'''"):
        return s[3:-3].strip()
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1].strip()
    return s

def _is_rate_limit_message(msg: t.Optional[str]) -> bool:
    """Heuristically detect rate limit / quota errors across providers."""
    if not msg:
        return False
    m = str(msg).lower()
    patterns = [
        "rate limit",           # generic
        "ratelimit",            # generic compact
        "429",                  # http status
        "quota exceeded",       # openai/generic
        "resource_exhausted",   # google/gemini
        "too many requests",    # generic
    ]
    return any(p in m for p in patterns)

# ===========================
# MAIN PROCESSING
# ===========================
if csv_file and zip_file:
    if st.button("🚀 Run Postmortem Analysis"):
        with st.spinner("Extracting and analyzing... please wait"):
            debug_box = st.expander("Debug logs", expanded=True)
            with debug_box:
                st.write({
                    "csv_uploaded": bool(csv_file),
                    "zip_uploaded": bool(zip_file),
                    "csv_uploaded_size": getattr(csv_file, "size", None),
                    "zip_uploaded_size": getattr(zip_file, "size", None),
                })
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, "project.zip")
            csv_path = os.path.join(temp_dir, "input.csv")
            # Precompute output CSV path for periodic autosaves
            csv_out = os.path.join(temp_dir, "postmortem_results.csv")
            # Persistent autosave file in user-selected folder
            persistent_csv_out = str(Path(AUTOSAVE_DIR) / "postmortem_results.csv")

            with open(zip_path, "wb") as f:
                f.write(zip_file.read())
            with open(csv_path, "wb") as f:
                f.write(csv_file.read())
            with debug_box:
                st.write({
                    "temp_dir": temp_dir,
                    "zip_path": zip_path,
                    "csv_path": csv_path,
                    "zip_path_size": os.path.getsize(zip_path),
                    "csv_path_size": os.path.getsize(csv_path),
                })

            DEST = Path(temp_dir) / "project"
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(DEST)
            # Print unzip destination to terminal
            print(f"[DEBUG] Unzipped project to: {DEST}")
            try:
                extracted_files = [str(p) for p in DEST.rglob("*") if p.is_file()]
            except Exception:
                extracted_files = []
            with debug_box:
                st.write({
                    "extracted_files_count": len(extracted_files),
                    "extracted_sample": extracted_files[:5],
                })

            try:
                df_in = pd.read_csv(csv_path)
                # df_in=df_in.iloc[:2,:]
                # Apply row capping based on user inputs
                _start = int(START_ROW) if START_ROW is not None else 0
                _end = int(END_ROW) if END_ROW and int(END_ROW) > 0 else None
                if _start != 0 or _end is not None:
                    df_in = df_in.iloc[_start:_end, :]
                    st.info(f"Processing row slice: iloc[{_start}:{_end if _end is not None else ''}, :]")
                # Reset index after slicing to ensure progress calculation works correctly
                df_in = df_in.reset_index(drop=True)
                st.write("✅ CSV Loaded:", df_in.shape)
                st.dataframe(df_in.head(3))
                with debug_box:
                    st.write({"csv_rows": len(df_in), "csv_cols": list(df_in.columns)})
            except Exception as e:
                with debug_box:
                    st.write({"csv_read_error": f"{type(e).__name__}: {e}"})
                raise

            col = detect_path_column(df_in)
            st.info(f"Using CSV path column: `{col}`")
            with debug_box:
                st.write({"detected_path_column": col})

            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage

            llm_gemini = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.2, api_key=GEMINI_API_KEY)
            with debug_box:
                st.write({
                    "api_keys": {
                        "gemini_provided": bool(GEMINI_API_KEY),
                    },
                    "models": [GEMINI_MODEL],
                })

            results = []
            supabase_client = None
            if SUPABASE_URL and SUPABASE_KEY:
                try:
                    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
                    with debug_box:
                        st.write({"supabase_client_initialized": True, "table": SUPABASE_TABLE})
                except Exception as e:
                    with debug_box:
                        st.write({"supabase_init_error": f"{type(e).__name__}: {e}"})

            progress = st.progress(0)
            total = len(df_in)
            log_box = st.expander("Row creation logs", expanded=True)

            for idx, (i, row) in enumerate(df_in.iterrows()):
                
                raw_path = row[col]
                file_path = resolve_path(raw_path, DEST,csv_file.name)
                # Print current CSV row to terminal
                print(file_path)
                found = file_path.exists() and file_path.suffix.lower() == ".java"
                if not found:
                    with debug_box:
                        st.write({
                            "skip_row": i,
                            "csv_path_value": raw_path,
                            "resolved_path": str(file_path),
                            "reason": "file not found or not .java",
                            "exists": file_path.exists(),
                            "suffix": file_path.suffix.lower(),
                        })
                    continue

                file_text = read_text_safe(file_path)
                file_path_str = str(file_path) if found else "[NOT FOUND]"
                user_prompt = USER_PROMPT_TEMPLATE.replace("{file_path}", file_path_str).replace("{code}", file_text)
                with debug_box:
                    st.write({
                        "row_index": i,
                        "resolved_path": str(file_path),
                        "file_text_length": len(file_text),
                        "user_prompt_length": len(user_prompt),
                    })

                def call_model(llm, usr):
                    try:
                        resp = llm.invoke([HumanMessage(content=usr)])
                        return _strip_wrapper_quotes(resp.content)
                    except Exception as e:
                        return f"[ERROR] {type(e).__name__}: {e}"

                with debug_box:
                    st.write({"calling_model_for_row": i})
                
                try:
                    gemini_res = call_model(llm_gemini, user_prompt)
                except Exception as e:
                    gemini_res = f"[ERROR] {type(e).__name__}: {e}"
                    with debug_box:
                        st.write({"model_error": f"{type(e).__name__}: {e}"})
                
                # Detect rate limits and break early with logs
                gemini_rl = _is_rate_limit_message(gemini_res)
                if gemini_rl:
                    with log_box:
                        st.error(f"Rate limit exceeded for Gemini. Halting further processing.")
                    with debug_box:
                        st.write({"rate_limit_exceeded": True, "provider": "Gemini", "row": int(i)})
                    break
                
                results.append({
                    "csv_row": i,
                    "csv_path_value": raw_path,
                    "resolved_path": str(file_path),
                    "gemini": gemini_res,
                })

                # Insert into Supabase per row if configured
                if supabase_client is not None and SUPABASE_TABLE:
                    try:
                        payload = {
                            "csv_row": int(i),
                            "csv_path_value": str(raw_path),
                            "resolved_path": str(file_path),
                            "gemini": str(gemini_res) if gemini_res is not None else None,
                        }
                        _ = supabase_client.table(SUPABASE_TABLE).insert(payload).execute()
                        print(f"[DEBUG] Supabase row inserted: {payload['csv_row']}")
                        with debug_box:
                            st.write({"supabase_insert_ok_row": int(i)})
                    except Exception as e:
                        print(f"[DEBUG] Supabase insert error for row {i}: {type(e).__name__}: {e}")
                        with debug_box:
                            st.write({"supabase_insert_error": {"row": int(i), "error": f"{type(e).__name__}: {e}"}})

                with log_box:
                    st.write({
                        "created_index": len(results) - 1,
                        "csv_row": i,
                        "csv_path_value": raw_path,
                        "resolved_path": str(file_path),
                    })
                with debug_box:
                    st.write({
                        "row_completed": i,
                        "results_count": len(results),
                        "gemini_len": None if gemini_res is None else len(str(gemini_res)),
                    })

                # Use enumerate index for progress to ensure it stays within [0.0, 1.0]
                progress_value = min((idx + 1) / total, 1.0)
                progress.progress(progress_value)

                # Periodic autosave every 10 completed results
                if len(results) > 0 and len(results) % 10 == 0:
                    try:
                        pd.DataFrame(results).to_csv(csv_out, index=False)
                        print(f"[DEBUG] Autosaved results CSV after {len(results)} rows -> {csv_out}")
                        # Also save to persistent location
                        pd.DataFrame(results).to_csv(persistent_csv_out, index=False)
                        print(f"[DEBUG] Autosaved persistent CSV after {len(results)} rows -> {persistent_csv_out}")
                        with debug_box:
                            st.write({
                                "autosave_rows": len(results),
                                "autosave_path": csv_out,
                                "autosave_size": os.path.getsize(csv_out) if os.path.exists(csv_out) else None,
                                "persistent_autosave_path": persistent_csv_out,
                                "persistent_autosave_size": os.path.getsize(persistent_csv_out) if os.path.exists(persistent_csv_out) else None,
                            })
                    except Exception as e:
                        print(f"[DEBUG] Autosave error after {len(results)} rows: {type(e).__name__}: {e}")
                        with debug_box:
                            st.write({
                                "autosave_error": f"{type(e).__name__}: {e}",
                                "autosave_rows": len(results),
                            })
                

            df_out = pd.DataFrame(results)
            st.success("✅ Analysis complete!")
            st.dataframe(df_out.head())
            with debug_box:
                st.write({"output_rows": len(df_out), "output_cols": list(df_out.columns)})

            df_out.to_csv(csv_out, index=False)
            # Final save to persistent location as well
            try:
                df_out.to_csv(persistent_csv_out, index=False)
                with debug_box:
                    st.write({
                        "final_persistent_save_path": persistent_csv_out,
                        "final_persistent_save_size": os.path.getsize(persistent_csv_out) if os.path.exists(persistent_csv_out) else None,
                    })
                print(f"[DEBUG] Final persistent CSV saved -> {persistent_csv_out}")
            except Exception as e:
                with debug_box:
                    st.write({"final_persistent_save_error": f"{type(e).__name__}: {e}"})
            with open(csv_out, "rb") as f:
                st.download_button("📥 Download Results CSV", data=f, file_name="postmortem_results.csv")
            with debug_box:
                try:
                    st.write({"csv_out_path": csv_out, "csv_out_size": os.path.getsize(csv_out)})
                except Exception as e:
                    st.write({"csv_out_stat_error": f"{type(e).__name__}: {e}"})

else:
    st.warning("Please provide Gemini API key and upload both CSV and ZIP files to proceed.")
