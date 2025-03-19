import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import time

# Set page config
st.set_page_config(
    page_title="Ontario Energy Demand System",
    page_icon="🔋",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background-color: #000000;
        padding: 20px;
        color: white;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        background-color: #006A4D;
        padding: 10px;
        color: white;
        text-align: left;
        margin-top: 0;
        margin-bottom: 20px;
    }
    .stButton button {
        background-color: #006A4D;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-size: 16px;
        border: none;
    }
    .stButton button:hover {
        background-color: #004d38;
    }
    .section-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<div class="main-header"><h1>Ontario Energy Demand System</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header"><h3>Ontario Energy Forecasting</h3></div>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Home", "Visualization", "Predict", "Evaluation"])

# Home page
if page == "Home":
    st.title("Welcome to Ontario Energy Forecasting System")
    st.write("Please select an option from the sidebar menu to continue.")
    
    # Add some descriptive content
    st.write("""
    ## About the System
    
    This system provides tools for analyzing and forecasting Ontario's energy demand and prices. It leverages advanced machine learning models to provide accurate predictions and insights.
    
    ### Key Features:
    - Data visualization and exploration
    - Multi-model prediction system
    - Model evaluation and comparison
    - Historical trend analysis
    """)
    
    # Display some example images
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://via.placeholder.com/400x300?text=Energy+Demand+Graph", caption="Example Energy Demand Graph")
    with col2:
        st.image("https://via.placeholder.com/400x300?text=Prediction+Results", caption="Example Prediction Results")

# Visualization page
elif page == "Visualization":
    st.markdown('<p class="section-title">Data Visualization</p>', unsafe_allow_html=True)
    
    # Create a form for visualization settings
    with st.form("visualization_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            data_source = st.selectbox(
                "Data Source",
                ["IESO Historical Data", "Weather Data", "Economic Indicators"]
            )
            
            visualization_type = st.selectbox(
                "Visualization Type",
                ["Line Chart", "Bar Chart", "Heat Map", "Scatter Plot"]
            )
        
        with col2:
            time_period = st.selectbox(
                "Time Period",
                ["Last Month", "Last 6 Months", "Last Year", "Last 5 Years"]
            )
        
        submitted = st.form_submit_button("Generate Visualization")
    
    # Generate visualization when form is submitted
    if submitted:
        st.success(f"Generating {visualization_type} for {data_source} over {time_period}")
        
        # Create placeholder data
        if time_period == "Last Month":
            days = 30
        elif time_period == "Last 6 Months":
            days = 180
        elif time_period == "Last Year":
            days = 365
        else:
            days = 1825
        
        # Generate sample data
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
        
        if data_source == "IESO Historical Data":
            # Generate energy demand data with daily and seasonal patterns
            base_demand = 18000  # Base demand in MW
            daily_pattern = np.sin(np.linspace(0, 2*np.pi, 24)).repeat(days//24 + 1)[:days]
            seasonal_pattern = 3000 * np.sin(np.linspace(0, 2*np.pi, 365)).repeat(days//365 + 1)[:days]
            random_noise = np.random.normal(0, 1000, days)
            values = base_demand + 3000 * daily_pattern + seasonal_pattern + random_noise
            
            data = pd.DataFrame({
                'Date': dates,
                'Energy Demand (MW)': values
            })
            
            title = "Ontario Energy Demand"
            y_label = "Energy Demand (MW)"
        
        elif data_source == "Weather Data":
            # Generate temperature data
            base_temp = 10  # Base temperature in Celsius
            seasonal_pattern = 15 * np.sin(np.linspace(0, 2*np.pi, 365)).repeat(days//365 + 1)[:days]
            random_noise = np.random.normal(0, 5, days)
            values = base_temp + seasonal_pattern + random_noise
            
            data = pd.DataFrame({
                'Date': dates,
                'Temperature (°C)': values
            })
            
            title = "Ontario Temperature"
            y_label = "Temperature (°C)"
        
        else:  # Economic Indicators
            # Generate economic indicator data (e.g., GDP growth)
            base_value = 100
            trend = np.linspace(0, 20, days)
            seasonal_pattern = 5 * np.sin(np.linspace(0, 8*np.pi, days))
            random_noise = np.random.normal(0, 3, days)
            values = base_value + trend + seasonal_pattern + random_noise
            
            data = pd.DataFrame({
                'Date': dates,
                'Economic Index': values
            })
            
            title = "Ontario Economic Index"
            y_label = "Index Value"
        
        # Create the visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if visualization_type == "Line Chart":
            plt.plot(data['Date'], data.iloc[:, 1], linewidth=2)
            plt.title(title)
            plt.xlabel("Date")
            plt.ylabel(y_label)
            plt.grid(True, alpha=0.3)
            
        elif visualization_type == "Bar Chart":
            # For bar chart, use monthly averages
            data.set_index('Date', inplace=True)
            monthly_data = data.resample('M').mean()
            monthly_data.plot(kind='bar', ax=ax)
            plt.title(f"Monthly Average {title}")
            plt.xlabel("Month")
            plt.ylabel(y_label)
            plt.xticks(rotation=45)
            
        elif visualization_type == "Heat Map":
            # Create a heatmap of daily values (reshape data)
            data.set_index('Date', inplace=True)
            
            # If there's enough data, create a month-hour heatmap
            if days >= 30:
                data_pivot = data.iloc[:, 0].groupby([data.index.month, data.index.hour]).mean().unstack()
                sns.heatmap(data_pivot, cmap="YlOrRd", annot=True, fmt=".0f", ax=ax)
                plt.title(f"{title} Heatmap by Month and Hour")
                plt.xlabel("Hour of Day")
                plt.ylabel("Month")
            else:
                st.error("Not enough data for a heatmap. Please select a longer time period.")
            
        elif visualization_type == "Scatter Plot":
            # Create a scatter plot with trend line
            plt.scatter(data.index, data.iloc[:, 0], alpha=0.5)
            
            # Add trend line
            z = np.polyfit(range(len(data)), data.iloc[:, 0], 1)
            p = np.poly1d(z)
            plt.plot(data.index, p(range(len(data))), "r--", linewidth=2)
            
            plt.title(f"{title} Scatter Plot with Trend")
            plt.xlabel("Date")
            plt.ylabel(y_label)
        
        st.pyplot(fig)
        
        # Display data sample
        st.subheader("Data Sample")
        st.dataframe(data.head())

# Predict page
elif page == "Predict":
    st.markdown('<p class="section-title">Prediction Configuration</p>', unsafe_allow_html=True)
    
    # Create a form for prediction settings
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            model = st.selectbox(
                "Select Model",
                ["ARIMA", "XGBoost", "LSTM", "ARIMA + LSTM + XGBoost"]
            )
            
            target = st.selectbox(
                "Prediction Target",
                ["Electricity Demand", "Price"]
            )
        
        with col2:
            duration = st.selectbox(
                "Forecast Duration",
                ["2 Years", "5 Years", "10 Years"]
            )
        
        submitted = st.form_submit_button("Run Prediction")
    
    # Generate predictions when form is submitted
    if submitted:
        with st.spinner(f"Running {model} prediction for {target} over {duration}..."):
            # Simulate processing time
            time.sleep(2)
            
            # Generate sample prediction data
            if duration == "2 Years":
                periods = 24
            elif duration == "5 Years":
                periods = 60
            else:
                periods = 120
            
            future_dates = pd.date_range(start=pd.Timestamp.now(), periods=periods, freq='M')
            
            if target == "Electricity Demand":
                # Generate forecast with uncertainty
                base_forecast = 18000 + np.random.normal(0, 100, periods)
                trend = np.linspace(0, 2000, periods)  # Increasing trend
                seasonal_pattern = 3000 * np.sin(np.linspace(0, 2*np.pi*periods/12, periods))
                forecast = base_forecast + trend + seasonal_pattern
                
                lower_bound = forecast * 0.9
                upper_bound = forecast * 1.1
                
                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'Forecast (MW)': forecast,
                    'Lower Bound': lower_bound,
                    'Upper Bound': upper_bound
                })
                
                title = "Electricity Demand Forecast"
                y_label = "Demand (MW)"
            
            else:  # Price
                # Generate price forecast
                base_price = 50 + np.random.normal(0, 5, periods)
                trend = np.linspace(0, 20, periods)  # Increasing trend
                seasonal_pattern = 15 * np.sin(np.linspace(0, 2*np.pi*periods/12, periods))
                forecast = base_price + trend + seasonal_pattern
                
                lower_bound = forecast * 0.85
                upper_bound = forecast * 1.15
                
                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'Forecast ($/MWh)': forecast,
                    'Lower Bound': lower_bound,
                    'Upper Bound': upper_bound
                })
                
                title = "Electricity Price Forecast"
                y_label = "Price ($/MWh)"
            
            # Create the forecast visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.plot(forecast_df['Date'], forecast_df.iloc[:, 1], 'b-', linewidth=2, label='Forecast')
            plt.fill_between(forecast_df['Date'], 
                            forecast_df['Lower Bound'], 
                            forecast_df['Upper Bound'], 
                            color='b', alpha=0.2, label='Uncertainty')
            plt.title(title)
            plt.xlabel("Date")
            plt.ylabel(y_label)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            st.success("Prediction completed!")
            st.pyplot(fig)
            
            # Display forecast data
            st.subheader("Forecast Data")
            st.dataframe(forecast_df)
            
            # Display model information
            st.subheader("Model Information")
            st.write(f"**Model:** {model}")
            st.write(f"**Target:** {target}")
            st.write(f"**Duration:** {duration}")
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Forecast", f"{forecast.mean():.2f}")
            with col2:
                st.metric("Maximum Forecast", f"{forecast.max():.2f}")
            with col3:
                st.metric("Minimum Forecast", f"{forecast.min():.2f}")

# Evaluation page
elif page == "Evaluation":
    st.markdown('<p class="section-title">Model Evaluation</p>', unsafe_allow_html=True)
    
    # Create a form for evaluation settings
    with st.form("evaluation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            model = st.selectbox(
                "Select Model",
                ["ARIMA", "XGBoost", "LSTM", "Ensemble"]
            )
            
            metric = st.selectbox(
                "Evaluation Metric",
                ["RMSE", "MAE", "MAPE", "All Metrics"]
            )
        
        with col2:
            test_period = st.selectbox(
                "Test Period",
                ["Last 3 Months", "Last 6 Months", "Last Year"]
            )
        
        submitted = st.form_submit_button("Run Evaluation")
    
    # Generate evaluation results when form is submitted
    if submitted:
        with st.spinner(f"Evaluating {model} using {metric} for {test_period}..."):
            # Simulate processing time
            time.sleep(2)
            
            # Generate sample evaluation data
            models = ["ARIMA", "XGBoost", "LSTM", "Ensemble"]
            metrics = ["RMSE", "MAE", "MAPE"]
            
            if model == "Ensemble":
                model_data = models
            else:
                model_data = [model]
                
            if metric == "All Metrics":
                metric_data = metrics
            else:
                metric_data = [metric]
            
            # Create evaluation results
            results = {}
            for m in model_data:
                results[m] = {}
                for met in metric_data:
                    if met == "RMSE":
                        # Lower is better for RMSE
                        if m == "Ensemble":
                            results[m][met] = 350 + np.random.normal(0, 20)
                        elif m == "LSTM":
                            results[m][met] = 400 + np.random.normal(0, 30)
                        elif m == "XGBoost":
                            results[m][met] = 450 + np.random.normal(0, 40)
                        else:  # ARIMA
                            results[m][met] = 500 + np.random.normal(0, 50)
                    elif met == "MAE":
                        # Lower is better for MAE
                        if m == "Ensemble":
                            results[m][met] = 250 + np.random.normal(0, 15)
                        elif m == "LSTM":
                            results[m][met] = 300 + np.random.normal(0, 20)
                        elif m == "XGBoost":
                            results[m][met] = 350 + np.random.normal(0, 25)
                        else:  # ARIMA
                            results[m][met] = 400 + np.random.normal(0, 30)
                    else:  # MAPE
                        # Lower is better for MAPE
                        if m == "Ensemble":
                            results[m][met] = 5 + np.random.normal(0, 0.5)
                        elif m == "LSTM":
                            results[m][met] = 6 + np.random.normal(0, 0.6)
                        elif m == "XGBoost":
                            results[m][met] = 7 + np.random.normal(0, 0.7)
                        else:  # ARIMA
                            results[m][met] = 8 + np.random.normal(0, 0.8)
            
            # Create a DataFrame for the results
            if len(model_data) > 1:
                # Multiple models
                eval_df = pd.DataFrame(
                    {m: {met: results[m][met] for met in metric_data} for m in model_data}
                )
            else:
                # Single model
                eval_df = pd.DataFrame(
                    {met: results[model_data[0]][met] for met in metric_data}, 
                    index=[0]
                )
            
            st.success("Evaluation completed!")
            
            # Display evaluation results
            st.subheader("Evaluation Results")
            
            if len(model_data) > 1 and len(metric_data) > 1:
                # Multiple models and metrics - create a heatmap
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(eval_df, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
                plt.title(f"Model Evaluation Results for {test_period}")
                st.pyplot(fig)
                
                # Show the data
                st.dataframe(eval_df)
                
            elif len(model_data) > 1:
                # Multiple models, single metric - create a bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                eval_df.loc[metric_data[0]].plot(kind='bar', ax=ax)
                plt.title(f"{metric_data[0]} Comparison for {test_period}")
                plt.xlabel("Model")
                plt.ylabel(metric_data[0])
                plt.xticks(rotation=0)
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Show the data
                st.dataframe(eval_df)
                
            elif len(metric_data) > 1:
                # Single model, multiple metrics - create a bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                eval_df.iloc[0].plot(kind='bar', ax=ax)
                plt.title(f"{model_data[0]} Evaluation for {test_period}")
                plt.xlabel("Metric")
                plt.ylabel("Value")
                plt.xticks(rotation=0)
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Show the data
                st.dataframe(eval_df)
                
            else:
                # Single model, single metric - show the value with a metric
                st.metric(
                    f"{metric_data[0]} for {model_data[0]}",
                    f"{eval_df.iloc[0][metric_data[0]]:.2f}"
                )
            
            # Add a time series plot of actual vs predicted values
            st.subheader("Actual vs Predicted")
            
            # Generate sample data
            if test_period == "Last 3 Months":
                days = 90
            elif test_period == "Last 6 Months":
                days = 180
            else:
                days = 365
            
            dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
            
            # Generate actual values
            base_value = 18000
            daily_pattern = np.sin(np.linspace(0, 2*np.pi, 24)).repeat(days//24 + 1)[:days]
            seasonal_pattern = 3000 * np.sin(np.linspace(0, 2*np.pi, 365)).repeat(days//365 + 1)[:days]
            random_noise = np.random.normal(0, 1000, days)
            actual_values = base_value + 3000 * daily_pattern + seasonal_pattern + random_noise
            
            # Generate predicted values with error
            if model == "Ensemble":
                error_factor = 0.05
            elif model == "LSTM":
                error_factor = 0.07
            elif model == "XGBoost":
                error_factor = 0.09
            else:  # ARIMA
                error_factor = 0.12
                
            predicted_values = actual_values * (1 + np.random.normal(0, error_factor, days))
            
            # Create a DataFrame
            comparison_df = pd.DataFrame({
                'Date': dates,
                'Actual': actual_values,
                'Predicted': predicted_values,
                'Error': predicted_values - actual_values
            })
            
            # Plot the comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.plot(comparison_df['Date'], comparison_df['Actual'], 'b-', linewidth=2, label='Actual')
            plt.plot(comparison_df['Date'], comparison_df['Predicted'], 'r--', linewidth=2, label='Predicted')
            plt.title(f"Actual vs Predicted Values for {test_period}")
            plt.xlabel("Date")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Plot the error
            fig, ax = plt.subplots(figsize=(10, 4))
            plt.plot(comparison_df['Date'], comparison_df['Error'], 'g-', linewidth=1)
            plt.title(f"Prediction Error for {test_period}")
            plt.xlabel("Date")
            plt.ylabel("Error")
            plt.axhline(y=0, color='r', linestyle='-')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Show a sample of the comparison data
            st.subheader("Comparison Data Sample")
            st.dataframe(comparison_df.head())

# Add a footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px; background-color: #f0f0f0;">
    <p>© 2025 Ontario Energy Forecasting System | All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)