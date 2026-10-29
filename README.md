AI Resume Screening & Candidate Ranking System
About the Project
This project is a simple AI-based resume screening system built using Python and Streamlit.
It takes a job description and multiple PDF resumes as input. It then compares the resumes with the job description and ranks them based on their similarity score.
Technologies Used
•	Python
•	Streamlit
•	PyPDF2
•	Pandas
•	Scikit-learn
•	TF-IDF
•	Cosine Similarity
How It Works
1.	Enter a job description.
2.	Upload multiple PDF resumes.
3.	The system extracts text from each resume using PyPDF2.
4.	TF-IDF converts the job description and resumes into numerical vectors.
5.	Cosine Similarity compares the job description with each resume.
6.	Resumes are sorted according to their similarity score.
7.	The result is displayed in a table.
Main Functions
extract_text_from_pdf()
This function extracts text from the uploaded PDF resume.
rank_resumes()
This function compares the job description with all resumes using TF-IDF and Cosine Similarity and returns similarity scores.
Installation
Install the required libraries:
pip install streamlit PyPDF2 pandas scikit-learn
Run the Project
Run the following command:
streamlit run app.py
Then open the URL shown in the terminal, usually:
http://localhost:8501
Output
The system displays the uploaded resumes with their similarity scores.
Example:
Resume              Score
--------------------------------
resume1.pdf         0.82
resume2.pdf         0.65
resume3.pdf         0.48
A higher score means the resume is more similar to the given job description.
Project Structure
AI-Resume-Screening/
│
├── app.py
├── README.md
└── requirements.txt
Purpose
This project is created for learning and demonstrates how NLP and Machine Learning techniques can be used for basic resume screening and candidate ranking.

