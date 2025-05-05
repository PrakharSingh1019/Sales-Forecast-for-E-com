"""
AI Summary Generator Module
This module provides functionality to generate insights and summaries from sales data
using local analysis without external API dependencies.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

class AISummaryGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def generate_summary(self, df, forecast_results=None):
        """Generate AI-powered summary of sales data and forecasts"""
        df['date'] = pd.to_datetime(df['date'])
        
        # Basic statistics
        total_revenue = df['revenue'].sum()
        avg_daily_revenue = df.groupby('date')['revenue'].sum().mean()
        best_selling_product = df.groupby('product')['quantity'].sum().idxmax()
        
        # Trend analysis
        daily_revenue = df.groupby('date')['revenue'].sum()
        revenue_trend = "increasing" if daily_revenue.iloc[-1] > daily_revenue.iloc[0] else "decreasing"
        
        # Product analysis
        product_performance = df.groupby('product').agg({
            'quantity': 'sum',
            'revenue': 'sum'
        }).sort_values('revenue', ascending=False)
        
        # Generate summary text
        summary = f"""
Sales Analysis Summary:
----------------------
Total Revenue: ${total_revenue:,.2f}
Average Daily Revenue: ${avg_daily_revenue:,.2f}
Best Selling Product: {best_selling_product}
Overall Revenue Trend: {revenue_trend}

Product Performance:
------------------"""
        
        for product, data in product_performance.iterrows():
            summary += f"\n{product}:"
            summary += f"\n  - Total Quantity: {data['quantity']:,.0f}"
            summary += f"\n  - Total Revenue: ${data['revenue']:,.2f}"
        
        # Add forecast insights if available
        if forecast_results is not None:
            summary += "\n\nForecast Insights:"
            summary += "\n-----------------"
            if isinstance(forecast_results, dict):
                for model, forecast in forecast_results.items():
                    avg_forecast = np.mean(forecast)
                    summary += f"\n{model.title()} Model:"
                    summary += f"\n  - Average Predicted Revenue: ${avg_forecast:,.2f}"
        
        return summary
    
    def generate_recommendations(self, df, forecast_results=None):
        """Generate AI-powered recommendations based on sales data and forecasts"""
        recommendations = []
        
        # Analyze product performance
        product_performance = df.groupby('product').agg({
            'quantity': 'sum',
            'revenue': 'sum'
        })
        
        # Identify underperforming products
        avg_revenue = product_performance['revenue'].mean()
        for product, data in product_performance.iterrows():
            if data['revenue'] < avg_revenue:
                recommendations.append(f"Consider reviewing pricing or marketing strategy for {product}")
        
        # Analyze daily patterns
        daily_revenue = df.groupby('date')['revenue'].sum()
        if daily_revenue.std() > daily_revenue.mean() * 0.5:
            recommendations.append("High revenue volatility detected. Consider implementing inventory management strategies.")
        
        # Add forecast-based recommendations
        if forecast_results is not None:
            if isinstance(forecast_results, dict):
                forecasts = list(forecast_results.values())
                if len(forecasts) > 1:
                    forecast_std = np.std([np.mean(f) for f in forecasts])
                    if forecast_std > np.mean([np.mean(f) for f in forecasts]) * 0.2:
                        recommendations.append("High forecast variance between models. Consider using ensemble forecasting.")
        
        return recommendations
    
    def generate_insights(self, data, forecast):
        """
        Generate insights from sales data and forecasts.
        
        Args:
            data (pd.DataFrame): Processed sales data
            forecast (dict): Dictionary containing forecast results
            
        Returns:
            str: Generated insights
        """
        return self.generate_summary(data, forecast)
    
    def create_executive_summary(self, insights):
        """Create an executive summary from the generated insights."""
        # Placeholder for future implementation
        pass
    
    def get_recommendations(self, insights):
        """Generate recommendations based on insights."""
        # Placeholder for future implementation
        pass 