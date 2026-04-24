# NunieFinal1

## End-to-End Machine Learning with CRISP-DM

Author: Anthony R. Nuñez

This project implements a complete machine learning pipeline using the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology to predict Titanic passenger survival.

### Project Overview

This end-to-end machine learning project analyzes the Titanic dataset and builds a predictive model to determine factors that influence passenger survival. Understanding these factors provides valuable insights into risk factors and can inform safety protocols or resource allocation in similar disaster scenarios.

### Features

- **Data Preprocessing**: Handling missing values, feature scaling, and encoding categorical variables
- **Model Training**: Random Forest Classifier with scikit-learn pipeline
- **Feature Engineering**: Numerical and categorical feature transformation
- **Model Evaluation**: Accuracy scores and classification reports
- **Visualization**: Streamlit integration for interactive analysis

### Dataset

The project uses the Titanic dataset containing:
- **Numerical Features**: Age, Fare, Siblings/Spouses, Parents/Children
- **Categorical Features**: Passenger Class, Sex
- **Target Variable**: Survived (binary classification)

### Installation

```bash
pip install -r requirements.txt
```

### Usage

1. **Run the ML Pipeline**:
   ```bash
   python a_nunez_end_to_end_machine_learning_with_crisp_dm.py
   ```

2. **Launch the Streamlit App**:
   ```bash
   streamlit run a_nunez_end_to_end_machine_learning_with_crisp_dm.py
   ```

### Dependencies

- pandas: Data manipulation and analysis
- scikit-learn: Machine learning models and preprocessing
- kmodes: Clustering with categorical data
- streamlit: Interactive web application framework
- pyngrok: Expose local server to the internet

### CRISP-DM Methodology

1. **Business Understanding**: Predicting survival rates for resource allocation
2. **Data Understanding**: Exploratory analysis of Titanic dataset
3. **Data Preparation**: Cleaning, feature engineering, and preprocessing
4. **Modeling**: Training Random Forest Classifier
5. **Evaluation**: Assessing model performance with accuracy and classification metrics
6. **Deployment**: Streamlit app for interactive predictions

### Model Performance

The Random Forest Classifier achieves high accuracy in predicting Titanic passenger survival with comprehensive evaluation metrics including precision, recall, and F1-scores.

### Author

Anthony R. Nuñez

### License

MIT License
