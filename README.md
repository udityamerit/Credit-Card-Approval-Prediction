# Credit Card Approval Prediction 🚀

This project predicts whether an individual’s credit card application should be approved, based on demographic and credit behavior data. It uses various machine learning models and compares their performance to select and deploy the best one.

---

## 📁 Project Structure

```

Credit-Card-Approval-Prediction/
│
├── Dataset/
│   ├── application\_record.csv       # Applicant demographic & employment data
│   └── credit\_record.csv            # Credit history data
│
├── Images/
│   ├── Decision\_Tree\_plot.png
│   ├── Logistic\_Regression\_plot.png
│   ├── Random\_Forest\_plot.png
│   ├── XGBoost\_plot.png
│   └── model\_comparison\_metrics.png
│
├── models/
│   ├── Random\_Forest\_best\_model.pkl # Saved best performing model
│   ├── best\_threshold.txt           # Threshold used for classification
│   └── train\_columns.pkl            # Feature columns used during training
│
├── notebooks/
│   ├── 1\_Visualizing\_and\_analyzing\_data.ipynb  # EDA & insights
│   ├── 2\_Data\_preprocessing.ipynb              # Feature engineering & cleaning
│   ├── 3\_Model\_building.ipynb                  # Training and evaluation of models
│   └── 4\_Prediction.ipynb                      # Using the trained model to predict on new data
│
├── requirements.txt      # Python dependencies
└── .gitignore

````

---

## 📊 Dataset Description

Two datasets are used from [Kaggle Credit Card Approval dataset](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction):

- `application_record.csv` - Contains demographic and employment data.
- `credit_record.csv` - Contains the credit behavior history of each individual.

---

## 🧠 Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Each model is evaluated based on:
- **F1 Score**
- Accuracy
- Precision
- Recall
- ROC AUC

The model with the best F1 score is selected and saved for production use.

---

## 🧪 Model Evaluation

Visualizations such as ROC curves, confusion matrices, and performance bar plots are provided in the `Images/` folder.

✅ **Best model:** Random Forest  
📈 **Best metric:** Highest F1-score (see `model_comparison_metrics.png`)

---

## 📦 Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/udityamerit/Credit-Card-Approval-Prediction.git
cd Credit-Card-Approval-Prediction
````

### 2. Create and activate a virtual environment

```bash
python -m venv newenv
source newenv/bin/activate  # or newenv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Project

You can run each notebook in order:

1. `1_Visualizing_and_analyzing_data.ipynb`
2. `2_Data_preprocessing.ipynb`
3. `3_Model_building.ipynb`
4. `4_Prediction.ipynb`

---

## 💾 Model Inference

To load the saved model and threshold:

```python
import joblib

model = joblib.load("models/Random_Forest_best_model.pkl")
threshold = float(open("models/best_threshold.txt").read())
features = joblib.load("models/train_columns.pkl")
```

Use this to predict new input data.

---

## 🛠 Technologies

* Python 3
* scikit-learn
* xgboost
* pandas, numpy
* matplotlib, seaborn
* Jupyter Notebook

---

## 📷 Visualizations

Plots for each model and comparison metrics are stored in the `Images/` directory to help understand how each model performed.

---

## 🙌 Author
[Uditya Narayan Tiwari](https://github.com/udityamerit)
[Kirti Pratihar ](https://github.com/KirtiPratihar)

---

## 📄 License

This project is open-source under the MIT License.

