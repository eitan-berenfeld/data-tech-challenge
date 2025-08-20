import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare data for Adelaide-London route"""
    df = pd.read_csv('TechChallenge_Data.csv')
    
    # Create datetime index
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month_num'].astype(str).str.zfill(2) + '-01')
    df.set_index('Date', inplace=True)
    
    # Create route identifier
    df['Route'] = df['AustralianPort'] + '-' + df['ForeignPort']
    
    # Get Adelaide-London data
    route_data = df[df['Route'] == 'Adelaide-London']['Passengers_Total'].dropna()
    
    print(f"Data loaded: {len(route_data)} observations")
    print(f"Period: {route_data.index.min()} to {route_data.index.max()}")
    
    return route_data

def test_stationarity(ts):
    """Test if series is stationary"""
    print("\nSTATIONARITY TEST")
    
    # ADF Test
    adf_result = adfuller(ts)
    print(f"ADF p-value: {adf_result[1]:.4f}")
    
    # KPSS Test
    kpss_result = kpss(ts)
    print(f"KPSS p-value: {kpss_result[1]:.4f}")
    
    # Decision
    if adf_result[1] <= 0.05 and kpss_result[1] >= 0.05:
        print("Result: STATIONARY (d=0)")
        return True, 0
    else:
        print("Result: NON-STATIONARY (d=1)")
        return False, 1

def test_seasonality(ts):
    """Test for seasonal patterns"""
    print("\nSEASONALITY TEST")
    
    if len(ts) < 24:
        print("Insufficient data for seasonal test")
        return False, 0
    
    # Seasonal decomposition
    decomp = seasonal_decompose(ts, model='additive', period=12)
    
    # Calculate seasonal strength
    seasonal_var = np.var(decomp.seasonal.dropna())
    residual_var = np.var(decomp.resid.dropna())
    seasonal_strength = seasonal_var / (seasonal_var + residual_var)
    
    print(f"Seasonal strength: {seasonal_strength:.3f}")
    
    if seasonal_strength > 0.3:
        print("Result: STRONG seasonality (D=1)")
        return True, 1
    else:
        print("Result: WEAK seasonality (D=0)")
        return False, 0

def validate_assumptions():
    """Main validation function"""
    print("SARIMA ASSUMPTION VALIDATION - Adelaide-London")
    
    # Load data
    ts = load_data()
    
    # Test stationarity
    is_stationary, d = test_stationarity(ts)
    
    # Test seasonality
    has_seasonality, D = test_seasonality(ts)
    
    # Model recommendation
    print("\nMODEL SPECIFICATION")
    if has_seasonality:
        print(f"Recommended model: SARIMA(1,{d},1)(1,{D},1,12)")
        model_params = {
            'order': (1, d, 1),
            'seasonal_order': (1, D, 1, 12)
        }
    else:
        print(f"Recommended model: ARIMA(1,{d},1)")
        model_params = {
            'order': (1, d, 1),
            'seasonal_order': None
        }
    
    print("Validation complete!")
    return ts, model_params

if __name__ == "__main__":
    ts, params = validate_assumptions()