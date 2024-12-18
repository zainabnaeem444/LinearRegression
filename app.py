import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Title of the app
st.title("Linear Regression on CSV Data")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Show the first few rows of the dataframe
    st.write("Data Preview:")
    st.dataframe(df.head())
    
    # Select features (X) and target (Y)
    st.sidebar.subheader("Select features for linear regression")

    # Create feature and target selection dropdowns
    feature_columns = df.columns.tolist()
    target_column = st.sidebar.selectbox("Select target (Y)", feature_columns)
    
    feature_columns.remove(target_column)
    selected_features = st.sidebar.multiselect("Select features (X)", feature_columns)
    
    if len(selected_features) > 0:
        # Prepare the data
        X = df[selected_features]
        Y = df[target_column]
        
        # Train-Test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        # Train a Linear Regression model
        model = LinearRegression()
        model.fit(X_train, Y_train)
        
        # Predict using the model
        Y_pred = model.predict(X_test)
        
        # Show the model's coefficients
        st.write(f"Linear Regression Model Coefficients: {dict(zip(selected_features, model.coef_))}")
        st.write(f"Intercept: {model.intercept_}")
        
        # Evaluation Metrics
        st.write(f"Mean Squared Error (MSE): {mean_squared_error(Y_test, Y_pred)}")
        st.write(f"R-squared: {r2_score(Y_test, Y_pred)}")
        
        # Plotting
        st.subheader("Regression Line Visualization")
        
        # If only one feature is selected, we can plot a 2D regression line
        if len(selected_features) == 1:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=X_test[selected_features[0]], y=Y_test, color='blue', label='Actual')
            sns.lineplot(x=X_test[selected_features[0]], y=Y_pred, color='red', label='Prediction')
            plt.xlabel(selected_features[0])
            plt.ylabel(target_column)
            plt.title(f"Linear Regression: {selected_features[0]} vs {target_column}")
            st.pyplot(plt)
        else:
            st.write("Multiple features selected. Cannot visualize the regression line in 2D.")
