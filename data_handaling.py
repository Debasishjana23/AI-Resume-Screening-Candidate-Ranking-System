
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv(
    "AI Resume Screening & Candidate Ranking System dataset.csv"
)

print("Dataset loaded successfully!")
print("Total records:", len(df))


# ==========================================
# 2. CLEAN DATASET
# ==========================================

text_columns = [
    "resume_text",
    "resume_skills",
    "projects",
    "certifications",
    "job_description"
]

for col in text_columns:
    df[col] = df[col].fillna("").astype(str)

# Convert numerical columns
for col in ["experience_years", "job_experience_required"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows without resume text or job descriptions
df = df[
    (df["resume_text"].str.strip() != "") &
    (df["job_description"].str.strip() != "")
].copy()

# Remove rows with missing target labels
df = df.dropna(subset=["shortlisted"])

# Ensure shortlist labels are numeric
df["shortlisted"] = pd.to_numeric(
    df["shortlisted"], errors="coerce"
)

df = df.dropna(subset=["shortlisted"])
df = df[df["shortlisted"].isin([0, 1])].copy()
df["shortlisted"] = df["shortlisted"].astype(int)


# ==========================================
# 3. PREPARE TRAINING FEATURES
# ==========================================

# Combine the job description and candidate information
df["combined_text"] = (
    df["job_description"] + " " +
    df["resume_text"] + " " +
    df["resume_skills"] + " " +
    df["projects"] + " " +
    df["certifications"]
)

# Select input features
X = df[
    [
        "combined_text",
        "experience_years",
        "job_experience_required"
    ]
].copy()

# Replace missing experience values
X["experience_years"] = X["experience_years"].fillna(0)
X["job_experience_required"] = (
    X["job_experience_required"].fillna(0)
)

# Select the target column
y = df["shortlisted"]


# ==========================================
# 4. SPLIT TRAINING AND TESTING DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 5. PREPROCESS THE DATA
# ==========================================

# Convert text into TF-IDF features
text_transformer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)

# Fill missing numerical values and scale experience features
number_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Combine text and numerical preprocessing
preprocessor = ColumnTransformer([
    ("text", text_transformer, "combined_text"),
    (
        "numbers",
        number_transformer,
        ["experience_years", "job_experience_required"]
    )
])


# ==========================================
# 6. TRAIN MACHINE LEARNING MODEL
# ==========================================

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 7. EVALUATE MODEL
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==========================================
# 8. SAVE TRAINED MODEL
# ==========================================

joblib.dump(model, "resume_model.pkl")

print("\nTrained model saved successfully!")
print("Model file: resume_model.pkl")
