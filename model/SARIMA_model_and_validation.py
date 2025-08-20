import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load Adelaide-London data"""
    df = pd.read_csv('TechChallenge_Data.csv')
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month_num'].astype(str).str.zfill(2) + '-01')
    df.set_index('Date', inplace=True)
    df['Route'] = df['AustralianPort'] + '-' + df['ForeignPort']
    
    route_data = df[df['Route'] == 'Adelaide-London']['Passengers_Total'].dropna()
    return route_data

def fit_sarima_model(ts, order, seasonal_order):
    """Fit SARIMA model"""
    print("\nFITTING SARIMA MODEL")
    print(f"Model: SARIMA{order}{seasonal_order}")
    
    # Split data: use last 6 months for testing
    split_point = len(ts) - 6
    train = ts.iloc[:split_point]
    test = ts.iloc[split_point:]
    
    print(f"Training data: {len(train)} observations")
    print(f"Test data: {len(test)} observations")
    
    # Fit model
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    fitted_model = model.fit(disp=False)
    
    print(f"AIC: {fitted_model.aic:.2f}")
    print("Model fit ")
    
    return fitted_model, train, test

def validate_model(fitted_model, train, test):
    """Validate model performance"""
    print("\nMODEL VALIDATION")
    
    # 1. Residual diagnostics
    residuals = fitted_model.resid
    
    # Ljung-Box test for independence
    lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
    lb_pvalue = lb_test['lb_pvalue'].min()
    
    # Normality test
    jb_stat, jb_pvalue = stats.jarque_bera(residuals)
    
    print("Diagnostic Tests:")
    print(f"Ljung-Box p-value: {lb_pvalue:.4f} {'✓' if lb_pvalue > 0.05 else '✗'}")
    print(f"Normality p-value: {jb_pvalue:.4f} {'✓' if jb_pvalue > 0.05 else '✗'}")
    
    # 2. Forecast accuracy
    forecast = fitted_model.get_forecast(steps=len(test))
    forecast_mean = forecast.predicted_mean
    
    # Calculate accuracy metrics
    mae = np.mean(np.abs(forecast_mean - test))
    mape = np.mean(np.abs((test - forecast_mean) / test)) * 100
    
    print("\nForecast Accuracy:")
    print(f"  MAE: {mae:.1f} passengers")
    print(f"  MAPE: {mape:.1f}%")
    
    return forecast_mean, {'mae': mae, 'mape': mape}

def run_sarima_analysis():
    """Main analysis function"""
    print("SARIMA MODEL VALIDATION - Adelaide-London")
    
    # Load data
    ts = load_data()
    print(f"Loaded {len(ts)} observations")
    
    # Basic model specification (simplified)
    order = (1, 1, 1)  # Simple SARIMA parameters
    seasonal_order = (1, 1, 1, 12)  # Monthly seasonality
    
    # Fit model
    fitted_model, train, test = fit_sarima_model(ts, order, seasonal_order)
    
    # Validate model
    forecast, accuracy = validate_model(fitted_model, train, test)
    
    # Generate future forecast
    print(f"\n12-MONTH FORECAST")
    future_forecast = fitted_model.get_forecast(steps=12)
    future_mean = future_forecast.predicted_mean
    
    print(f"Average forecast: {future_mean.mean():.0f} passengers/month")
    print(f"Forecast range: {future_mean.min():.0f} - {future_mean.max():.0f}")
    
    print("\nAnalysis complete!")
    return fitted_model, accuracy

if __name__ == "__main__":
    model, results = run_sarima_analysis()