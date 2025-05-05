"""

This is the main application file that creates the Streamlit web interface
for the Sales Forecasting application. It integrates data loading, forecasting,
and AI summary generation components.

The application allows users to:
- View historical sales data
- Generate sales forecasts using different models
- Get AI-generated insights and recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from forecast_engine import ForecastEngine
from ai_summary import AISummaryGenerator
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize components
forecast_engine = ForecastEngine()
ai_summary = AISummaryGenerator()

def plot_forecast(df, forecast, forecast_dates, model_type):
    """Create an interactive plot of historical data and forecast"""
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['revenue'],
        name='Historical Data',
        line=dict(color='blue')
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast,
        name=f'{model_type} Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Sales Forecast using {model_type}',
        xaxis_title='Date',
        yaxis_title='Revenue',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def main():
    st.title("Sales Forecast Application")
    st.markdown("""
    This application helps you analyze sales data and generate forecasts using multiple models.
    Upload your sales data or use the sample data to get started.
    """)

    # Sidebar
    st.sidebar.header("Configuration")

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload your sales data (CSV)", type=['csv'])

    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Forecasting Model",
        ["ARIMA", "Random Forest", "Moving Average"]
    )

    # Forecast period
    forecast_periods = st.sidebar.slider(
        "Forecast Period (days)",
        min_value=7,
        max_value=90,
        value=30
    )

    # Load data
    @st.cache_data
    def load_data(file):
        if file is not None:
            return pd.read_csv(file)
        return pd.read_csv("sample_sales.csv")

    # Main content
    try:
        # Load data
        df = load_data(uploaded_file)
        
        # Display raw data
        st.subheader("Raw Data")
        st.dataframe(df.head())
        
        # Generate forecasts
        forecast_results = {}
        model_type_lower = model_type.lower().replace(" ", "_")
        
        with st.spinner(f"Generating {model_type} forecast..."):
            forecast = forecast_engine.generate_forecast(
                df,
                model_type=model_type_lower,
                forecast_periods=forecast_periods
            )
            forecast_results[model_type] = forecast
            
            # Generate forecast dates
            forecast_dates = pd.date_range(
                start=pd.to_datetime(df['date'].max()) + pd.Timedelta(days=1),
                periods=forecast_periods
            )
            
            # Plot results
            st.subheader("Sales Forecast")
            fig = plot_forecast(df, forecast, forecast_dates, model_type)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display forecast values
            st.subheader("Forecast Values")
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Predicted Revenue': forecast
            })
            st.dataframe(forecast_df.style.format({'Predicted Revenue': '${:,.2f}'}))
        
        # Generate AI summary
        st.subheader("AI Analysis")
        with st.spinner("Generating AI analysis..."):
            summary = ai_summary.generate_summary(df, forecast_results)
            st.text(summary)
        
        # Generate recommendations
        st.subheader("Recommendations")
        with st.spinner("Generating recommendations..."):
            recommendations = ai_summary.generate_recommendations(df, forecast_results)
            for rec in recommendations:
                st.write(f"- {rec}")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your data format and try again.")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Sales Forecast App",
        page_icon="📈",
        layout="wide"
    )
    main() 