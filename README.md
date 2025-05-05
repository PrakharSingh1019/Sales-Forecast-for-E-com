# Sales Forecast for E-com

A powerful sales forecasting application that helps businesses predict future sales trends using advanced analytics and machine learning techniques.

## Features

- Interactive data visualization
- Machine learning-based sales forecasting
- AI-powered insights and summaries
- User-friendly web interface
- Historical data analysis
- Customizable forecast parameters

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone this repository:
```bash
git clone [your-repository-url]
cd sales_forecast_app
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
   - Copy the `.env.example` file to `.env`
   - Fill in your API keys and configuration settings

## Project Structure

- `app.py` - Main Streamlit application
- `forecast_engine.py` - Core forecasting logic
- `data_loader.py` - Data loading and preprocessing
- `ai_summary.py` - AI-powered insights generation
- `requirements.txt` - Project dependencies
- `sample_sales.csv` - Sample data for testing

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload your sales data or use the sample data provided

4. Configure your forecast parameters:
   - Select the forecast period
   - Choose the forecasting model
   - Set confidence intervals

5. View and analyze the results:
   - Interactive charts
   - Forecast metrics
   - AI-generated insights

## Dependencies

- pandas - Data manipulation and analysis
- numpy - Numerical computing
- matplotlib & seaborn - Data visualization
- scikit-learn - Machine learning algorithms
- statsmodels - Statistical models
- streamlit - Web application framework
- supabase - Database integration
- openai - AI-powered insights
- python-dotenv - Environment variable management

