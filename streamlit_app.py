import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")

st.title("🚢 Titanic Survival Prediction")
st.markdown("---")
st.markdown("### End-to-End Machine Learning with CRISP-DM")
st.write("*Author: Anthony R. Núñez*")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Home", "Data Analysis", "Model Training", "Make Predictions"])

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the Titanic dataset"""
    try:
        # Try to load from local file or use sample data
        df = pd.read_csv("titanic.csv")
    except:
        st.warning("Titanic dataset not found. Please upload the dataset.")
        uploaded_file = st.file_uploader("Upload titanic.csv", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            # Create sample data for demonstration
            df = pd.DataFrame({
                'Pclass': np.random.randint(1, 4, 100),
                'Sex': np.random.choice(['male', 'female'], 100),
                'Age': np.random.uniform(1, 80, 100),
                'Fare': np.random.uniform(0, 500, 100),
                'sibspouse': np.random.randint(0, 9, 100),
                'parentchild': np.random.randint(0, 7, 100),
                'Survived': np.random.choice([0, 1], 100),
                'Name': [f'Passenger_{i}' for i in range(100)]
            })
            st.info("Using sample data for demonstration. Upload your own dataset to see real results.")
    
    return df

@st.cache_resource
def train_model(X_train, X_test, y_train, y_test):
    """Train the machine learning model"""
    numerical_features = ['Age', 'Fare', 'sibspouse', 'parentchild']
    categorical_features = ['Pclass', 'Sex']
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
    ])
    
    model.fit(X_train, y_train)
    return model

# Page 1: Home
if page == "Home":
    st.header("Welcome to the Titanic Survival Prediction System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        ### About This Project
        
        This application uses machine learning to predict whether a Titanic passenger would have survived.
        It implements the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology.
        
        **Key Features:**
        - 📊 Data exploration and analysis
        - 🤖 Random Forest classification model
        - 📈 Interactive visualizations
        - 🔮 Real-time predictions
        """)
    
    with col2:
        st.info("""
        ### CRISP-DM Phases:
        1. **Business Understanding** - Predicting survival rates
        2. **Data Understanding** - Exploratory analysis
        3. **Data Preparation** - Cleaning & preprocessing
        4. **Modeling** - Training Random Forest
        5. **Evaluation** - Performance metrics
        6. **Deployment** - This Streamlit app
        """)
    
    st.markdown("---")
    st.write("Use the navigation menu on the left to explore different sections of the application.")

# Page 2: Data Analysis
elif page == "Data Analysis":
    st.header("📊 Data Analysis")
    
    df = load_and_prepare_data()
    
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Total Features", len(df.columns))
    with col3:
        st.metric("Survived", int(df['Survived'].sum()) if 'Survived' in df.columns else "N/A")
    with col4:
        st.metric("Did Not Survive", len(df) - int(df['Survived'].sum()) if 'Survived' in df.columns else "N/A")
    
    st.subheader("First Few Rows")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Dataset Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.bar_chart(missing)
    else:
        st.success("No missing values detected!")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Survived' in df.columns:
            survival_counts = df['Survived'].value_counts()
            fig = px.pie(names=['Did Not Survive', 'Survived'], 
                        values=[survival_counts[0], survival_counts[1]],
                        title="Survival Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Sex' in df.columns and 'Survived' in df.columns:
            survival_by_sex = pd.crosstab(df['Sex'], df['Survived'])
            fig = px.bar(survival_by_sex, title="Survival by Sex", 
                        labels={'value': 'Count', 'index': 'Sex'})
            st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Age' in df.columns:
            fig = px.histogram(df, x='Age', nbins=30, title="Age Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Pclass' in df.columns:
            class_counts = df['Pclass'].value_counts().sort_index()
            fig = px.bar(x=class_counts.index, y=class_counts.values,
                        title="Passenger Class Distribution",
                        labels={'x': 'Class', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)

# Page 3: Model Training
elif page == "Model Training":
    st.header("🤖 Model Training & Evaluation")
    
    df = load_and_prepare_data()
    
    # Data preparation
    df_clean = df.drop(columns=['Name'], errors='ignore')
    X = df_clean.drop('Survived', axis=1)
    y = df_clean['Survived']
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.info(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")
    
    # Train model
    with st.spinner('Training model...'):
        model = train_model(X_train, X_test, y_train, y_test)
    
    st.success("Model training completed!")
    
    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Accuracy Score", f"{accuracy:.4f}")
    
    with col2:
        st.metric("Test Set Size", len(X_test))
    
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)
    
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Did Not Survive', 'Survived'],
        y=['Did Not Survive', 'Survived'],
        text=cm,
        texttemplate='%{text}',
        colorscale='Blues'
    ))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("Model Information")
    st.write("**Model:** Random Forest Classifier")
    st.write("**Random State:** 42")
    st.write("**Number of Estimators:** 100")

# Page 4: Make Predictions
elif page == "Make Predictions":
    st.header("🔮 Make Predictions")
    
    df = load_and_prepare_data()
    df_clean = df.drop(columns=['Name'], errors='ignore')
    X = df_clean.drop('Survived', axis=1)
    y = df_clean['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with st.spinner('Loading model...'):
        model = train_model(X_train, X_test, y_train, y_test)
    
    st.write("Enter passenger details to predict survival:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = First, 2 = Second, 3 = Third")
        age = st.slider("Age", 0, 100, 30, help="Passenger age in years")
        sibspouse = st.slider("Siblings/Spouses", 0, 8, 0, help="Number of siblings/spouses aboard")
    
    with col2:
        sex = st.selectbox("Sex", ["male", "female"])
        fare = st.number_input("Fare", 0.0, 500.0, 50.0, help="Ticket fare in pounds")
        parentchild = st.slider("Parents/Children", 0, 6, 0, help="Number of parents/children aboard")
    
    if st.button("🔮 Predict Survival", key="predict"):
        # Create input dataframe
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex],
            'Age': [age],
            'Fare': [fare],
            'sibspouse': [sibspouse],
            'parentchild': [parentchild]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.success("✅ **SURVIVED**")
                st.write(f"Survival Probability: **{probability[1]:.2%}**")
            else:
                st.error("❌ **DID NOT SURVIVE**")
                st.write(f"Survival Probability: **{probability[1]:.2%}**")
        
        with col2:
            # Create probability visualization
            fig = go.Figure(data=[
                go.Bar(x=['Did Not Survive', 'Survived'], 
                       y=[probability[0], probability[1]],
                       marker_color=['red', 'green'])
            ])
            fig.update_layout(title="Prediction Probability", yaxis_title="Probability")
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("*Built with Streamlit | Machine Learning with Scikit-learn | Author: Anthony R. Núñez*")
