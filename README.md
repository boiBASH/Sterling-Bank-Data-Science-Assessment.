# Sterling Bank Data Science Assessment

This repository contains the analysis, modeling, and deployment artifacts for the Sterling Bank Data Science Case Study. The goal of this project is to predict loan default status and deploy a user-friendly interface for interaction with the trained machine learning model.

---

## 📁 Repository Structure

```
Sterling-Bank-Data-Science-Assessment/
├── Data_Preprocessing.ipynb               # Data cleaning and preprocessing notebook
├── Exploratory_Data_Analysis_Feature_Encoding.ipynb  # Exploratory data analysis and feature engineering
├── Model_Building_Evaluation.ipynb        # Machine learning model training and evaluation notebook
├── Model_Building_Evaluation_with_Dagshub.ipynb # Model experimentation tracked with Dagshub
├── app.py                                 # Streamlit application for interactive deployment
├── cleaned_loan_data.xlsx                 # Preprocessed loan dataset
├── light_rf_model.pkl                     # Trained Random Forest model artifact
├── requirements.txt                       # Python dependencies for the project
├── sterling_bank_logo.png                 # Project branding asset
└── README.md                              # Project documentation
```

---

## 🚀 Project Workflow

The project workflow is structured into clear stages:

### 1. **Data Preprocessing** (`Data_Preprocessing.ipynb`)

* Handling missing values
* Outlier detection and treatment
* Data normalization and cleaning

### 2. **Exploratory Data Analysis (EDA)** (`Exploratory_Data_Analysis_Feature_Encoding.ipynb`)

* Exploratory visualization of dataset
* Feature encoding (categorical variables)
* Feature correlation analysis

### 3. **Model Building and Evaluation** (`Model_Building_Evaluation.ipynb` & `Model_Building_Evaluation_with_Dagshub.ipynb`)

* Training Logistic Regression and Random Forest Classifier
* Evaluating model performance with classification metrics
* Dagshub experiment tracking integration for reproducibility

### 4. **Deployment** (`app.py`)

* Interactive Streamlit web application
* User input for real-time prediction
* Visualization of prediction outcomes
* Link to the app: https://loanexplorer.streamlit.app/

---

## 🛠️ Technologies Used

* **Python**: Primary programming language
* **Pandas & NumPy**: Data manipulation and analysis
* **Scikit-learn**: Machine learning modeling
* **Matplotlib & Seaborn**: Data visualization
* **Streamlit**: Web application deployment
* **Dagshub**: Model and experiment tracking

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your_username/Sterling-Bank-Data-Science-Assessment.git
cd Sterling-Bank-Data-Science-Assessment
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 🤝 Contributing

Contributions, suggestions, and feedback are welcome. Feel free to open issues or submit pull requests.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For further questions or clarifications, please contact:

* **Email**: [Bashirudeenopeyemi772@example.com](mailto:Bashirudeenopeyemi772@example.com)
* **GitHub**: [boiBASH](https://github.com/boibash)

---

