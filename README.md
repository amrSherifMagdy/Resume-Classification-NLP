# 📄 Automated Resume Classification using NLP & Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Project Overview
Manual resume screening is time-consuming and prone to human bias. This project builds an automated **Natural Language Processing (NLP)** pipeline that analyzes raw resume text and classifies candidates into **25 distinct job categories** (e.g., Data Science, Web Designing, Java Developer, HR, etc.).

## 📊 Dataset
- **Source:** Resume Dataset (`UpdatedResumeDataSet.csv`)
- **Total Records:** 962 resumes across 25 job categories.
- **Class Balancing:** Applied **Random Oversampling** to balance minority classes (84 samples per category, totaling 2,100 balanced records).

## ⚙️ Methodology & Workflow
1. **Exploratory Data Analysis (EDA):** Visualized category distributions using bar and pie charts (`Matplotlib` & `Seaborn`).
2. **Data Balancing:** Handled class imbalance to prevent model bias toward majority job roles.
3. **Text Preprocessing (`cleanResume`):**
   - Removed URLs, hashtags (`#`), mentions (`@`), punctuation, and special characters.
   - Stripped non-ASCII characters and normalized whitespace.
4. **Feature Engineering:**
   - Encoded target labels using `LabelEncoder`.
   - Extracted numerical text features using **TF-IDF Vectorization** (`stop_words='english'`).
5. **Model Training & Evaluation:**
   - Split data into **80% Training** and **20% Testing** sets.
   - Trained multi-class classifiers using `OneVsRestClassifier` (KNN / SVC / Random Forest).

## 🚀 How to Run Locally
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Resume-Classification-NLP.git](https://github.com/YOUR_USERNAME/Resume-Classification-NLP.git)
   cd Resume-Classification-NLP

  1. Install dependencies:
   Bash
   pip install -r requirements.txt
   
  2. Launch Jupyter Notebook:
   Bash
   jupyter notebook notebooks/resume-classification.ipynb
   
🔮 Future Improvements
[ ] Perform train_test_split prior to oversampling to completely eliminate data leakage.

[ ] Build an interactive web interface using Streamlit to upload .pdf and .docx resumes directly.

[ ] Experiment with Transformer-based embeddings (e.g., BERT).
