# Diabetes Prediction Using Machine Learning

This project focuses on predicting the likelihood of diabetes using the Pima Indians Diabetes Dataset. The project leverages various machine learning algorithms and tools to provide accurate predictions and insights from the data.

---

## Dataset
The dataset used in this project is the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) available on Kaggle. It consists of 768 entries with 8 features:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

The target variable is `Outcome`, which indicates whether an individual has diabetes (1) or not (0).

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or any Python IDE

### Steps to Install Dependencies
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd Diabetes-Prediction-ML
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the dataset is placed in the root directory as `diabetes.csv` or update the path in the notebook.

---

## Running the Project

1. **Prepare the Dataset**:
   Ensure the dataset file `diabetes.csv` is available in the project directory.

2. **Run the Notebook**:
   Open the notebook in Jupyter and execute the cells sequentially:
   ```bash
   jupyter notebook diabetes-prediction.ipynb
   ```

3. **Train and Evaluate Models**:
   The notebook includes implementations of the following algorithms:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Support Vector Classifier (SVC)
   - Random Forest Classifier
   - Decision Tree Classifier
   - Gradient Boosting and LightGBM

   Metrics such as accuracy, confusion matrix, F1-score, and ROC-AUC are computed to evaluate model performance.

---

## Key Features
- **Exploratory Data Analysis (EDA)**:
  - Visualization and statistical analysis of features.
  - Correlation matrix and feature importance.

- **Model Comparison**:
  - Multiple machine learning models with hyperparameter tuning.
  - Cross-validation for robust evaluation.

- **Metrics and Results**:
  - Detailed evaluation metrics for each model.
  - ROC curves and AUC scores for classification models.

---

## Example Outputs
Include metrics and visualizations from the notebook such as:
- Accuracy and F1-score comparison.
- Confusion matrix for the best model.
- ROC-AUC curves.

---

## References
- Dataset: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Machine Learning Libraries: scikit-learn, LightGBM

---

## License
Specify the license under which this code is shared, such as MIT or GPL.

---

## Acknowledgments
Special thanks to the UCI Machine Learning Repository for the dataset and the open-source community for the tools and frameworks used in this project.
