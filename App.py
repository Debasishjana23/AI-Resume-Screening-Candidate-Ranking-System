
# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import streamlit as st
import pandas as pd
import joblib
import re

from PyPDF2 import PdfReader


# ==========================================
# 2. PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide"
)


# ==========================================
# 3. CUSTOM WEBSITE DESIGN
# ==========================================

st.markdown("""
<style>
    .stApp {
        background-color: #f4f7fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .hero {
        background: linear-gradient(120deg, #102a43, #2563eb);
        padding: 35px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        font-size: 34px;
        margin-bottom: 10px;
    }

    .hero p {
        color: #e0ecff;
        font-size: 16px;
    }

    .section-title {
        color: #102a43;
        font-size: 23px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #e1e8f0;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
    }

    .metric-label {
        color: #64748b;
        font-size: 14px;
    }

    .metric-value {
        color: #102a43;
        font-size: 27px;
        font-weight: 700;
        margin-top: 6px;
    }

    .stButton > button {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 9px;
        padding: 12px 22px;
        font-weight: 600;
        width: 100%;
    }

    .stButton > button:hover {
        background: #1d4ed8;
        color: white;
    }

    div[data-testid="stFileUploader"] {
        background: white;
        border-radius: 12px;
        padding: 15px;
        border: 1px dashed #93a4b8;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        padding: 25px 0 5px 0;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. WEBSITE HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <h1>📄 AI Resume Screening</h1>
    <p>
        Smart candidate screening powered by Machine Learning.
        Upload resumes, analyze candidates, and view your rankings.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 5. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("resume_model.pkl")


# ==========================================
# 6. EXTRACT TEXT FROM PDF
# ==========================================

def extract_text(file):
    pdf = PdfReader(file)
    text = ""

    for page in pdf.pages:
        text += (page.extract_text() or "") + " "

    return text.strip()


# ==========================================
# 7. FIND EXPERIENCE
# ==========================================

def get_experience(text):
    years = re.findall(
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?",
        text,
        re.IGNORECASE
    )

    if years:
        return max(float(year) for year in years)

    return 0.0


# ==========================================
# 8. RANK RESUMES
# ==========================================

def rank_resumes(job_description, resumes, model):
    results = []

    required_exp = get_experience(job_description)
    class_index = list(model.classes_).index(1)

    for name, text in resumes:

        data = pd.DataFrame({
            "combined_text": [job_description + " " + text],
            "experience_years": [get_experience(text)],
            "job_experience_required": [required_exp]
        })

        probability = model.predict_proba(data)[0]
        score = round(probability[class_index] * 100, 2)

        results.append({
            "Resume": name,
            "Experience (Years)": get_experience(text),
            "Shortlist Probability (%)": score
        })

    results = pd.DataFrame(results)

    return results.sort_values(
        by="Shortlist Probability (%)",
        ascending=False
    ).reset_index(drop=True)


# ==========================================
# 9. JOB DESCRIPTION SECTION
# ==========================================

st.markdown(
    '<div class="section-title">🎯 Job Requirements</div>',
    unsafe_allow_html=True
)

job_description = st.text_area(
    "Paste the job description",
    height=170,
    placeholder=(
        "Example: We are looking for a Python developer "
        "with SQL, Machine Learning, and problem-solving skills..."
    )
)


# ==========================================
# 10. RESUME UPLOAD SECTION
# ==========================================

st.markdown(
    '<div class="section-title">📂 Upload Candidate Resumes</div>',
    unsafe_allow_html=True
)

uploaded_files = st.file_uploader(
    "Choose PDF resume files",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can upload multiple PDF resumes at once."
)

if uploaded_files:
    st.info(f"📎 {len(uploaded_files)} resume(s) selected.")


# ==========================================
# 11. ANALYZE BUTTON
# ==========================================

if st.button("🚀 Analyze & Rank Resumes"):

    if not job_description.strip():
        st.warning("Please enter a job description.")

    elif not uploaded_files:
        st.warning("Please upload at least one resume.")

    else:
        try:
            model = load_model()
            resumes = []
            failed_files = []

            with st.spinner("Reading resumes and analyzing candidates..."):

                for file in uploaded_files:
                    try:
                        text = extract_text(file)

                        if text:
                            resumes.append((file.name, text))
                        else:
                            failed_files.append(file.name)

                    except Exception:
                        failed_files.append(file.name)

                if resumes:
                    results = rank_resumes(
                        job_description,
                        resumes,
                        model
                    )

                    # Save results for display and download
                    st.session_state["results"] = results

                else:
                    st.session_state.pop("results", None)

            if failed_files:
                st.warning(
                    "Could not read: " + ", ".join(failed_files)
                )

            if not resumes:
                st.error(
                    "No readable PDF text found. "
                    "Try PDFs that contain selectable text."
                )

        except FileNotFoundError:
            st.error(
                "Model file not found. Run data_handaling.py "
                "first to create resume_model.pkl."
            )

        except Exception as error:
            st.error(f"Something went wrong: {error}")


# ==========================================
# 12. DISPLAY RANKING RESULTS
# ==========================================

if "results" in st.session_state:

    results = st.session_state["results"]

    st.markdown(
        '<div class="section-title">📊 Candidate Overview</div>',
        unsafe_allow_html=True
    )

    total_candidates = len(results)
    top_score = results["Shortlist Probability (%)"].max()
    average_score = results["Shortlist Probability (%)"].mean()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Candidates</div>
            <div class="metric-value">{total_candidates}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Highest Shortlist Probability</div>
            <div class="metric-value">{top_score:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Probability</div>
            <div class="metric-value">{average_score:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">🏆 Ranked Candidates</div>',
        unsafe_allow_html=True
    )

    # Add ranking numbers
    display_results = results.copy()
    display_results.insert(
        0, "Rank", range(1, len(display_results) + 1)
    )

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Shortlist Probability (%)": st.column_config.ProgressColumn(
                "Shortlist Probability",
                format="%.2f%%",
                min_value=0,
                max_value=100
            )
        }
    )

    # Show the top-ranked candidate
    best_candidate = results.iloc[0]

    st.success(
        f"Top-ranked resume: {best_candidate['Resume']} "
        f"— predicted shortlist probability: "
        f"{best_candidate['Shortlist Probability (%)']:.2f}%"
    )

    # Download results
    csv_data = display_results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Ranking Report (CSV)",
        data=csv_data,
        file_name="ranked_candidates.csv",
        mime="text/csv"
    )


# ==========================================
# 13. FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    AI Resume Screening System | Built with Streamlit & Machine Learning
    <br>
    Predictions are decision-support estimates, not final hiring decisions.
</div>
""", unsafe_allow_html=True)
