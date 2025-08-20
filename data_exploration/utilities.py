import pandas as pd
import numpy as np
from datetime import datetime

def load_and_clean_data(filepath):
    """Load CSV data and perform initial cleaning operations."""
    df = pd.read_csv(filepath)
    
    # Check data quality - verify totals = in + out
    columns = [('Passengers_In', 'Passengers_Out', 'Passengers_Total'), 
               ('Freight_In_(tonnes)', 'Freight_Out_(tonnes)', 'Freight_Total_(tonnes)'),
               ('Mail_In_(tonnes)', 'Mail_Out_(tonnes)', 'Mail_Total_(tonnes)')]
    
    for x, y, z in columns:
        calculated_total = round(df[x] + df[y], 3)
        mismatches = df[z] != calculated_total
    
    return df

def create_datetime_index(df):
    """Create proper datetime index and clean up date columns."""
    # Create datetime index from year and month
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month_num'].astype(str).str.zfill(2) + '-01')
    df.set_index('Date', inplace=True)
    df.drop(['Month', 'Month_num'], axis=1, inplace=True)
    
    # Create route identifier
    df['Route'] = df['AustralianPort'] + '-' + df['ForeignPort']
    
    # Extract year and month from Date for filtering
    df['Year'] = pd.to_datetime(df.index).year
    df['Month'] = pd.to_datetime(df.index).month
    
    return df

def filter_data_by_period(data, start_year, start_month, end_year, end_month):
    """Filter data by year/month range."""
    return data[
        ((data['Year'] == start_year) & (data['Month'] >= start_month)) |
        ((data['Year'] > start_year) & (data['Year'] < end_year)) |
        ((data['Year'] == end_year) & (data['Month'] <= end_month))
    ].copy()

def get_cargo_columns():
    """Return mapping of cargo types to their column names."""
    return {
        'passengers': {'in': 'Passengers_In', 'out': 'Passengers_Out', 'total': 'Passengers_Total'},
        'freight': {'in': 'Freight_In_(tonnes)', 'out': 'Freight_Out_(tonnes)', 'total': 'Freight_Total_(tonnes)'},
        'mail': {'in': 'Mail_In_(tonnes)', 'out': 'Mail_Out_(tonnes)', 'total': 'Mail_Total_(tonnes)'}
    }

def get_default_cargo_weights():
    """Return default cargo weights for passenger equivalent calculations."""
    return {'passengers': 1, 'freight': 10, 'mail': 20}

def calculate_passenger_equivalents(df, cargo_weights=None):
    """Calculate passenger equivalent volumes for freight and mail."""
    if cargo_weights is None:
        cargo_weights = get_default_cargo_weights()
    
    df = df.copy()
    df['freightPassengerEquiv'] = df['Freight_Total_(tonnes)'] * cargo_weights['freight']
    df['mailPassengerEquiv'] = df['Mail_Total_(tonnes)'] * cargo_weights['mail']
    df['totalPassengerEquiv'] = (
        df['Passengers_Total'] + 
        df['freightPassengerEquiv'] + 
        df['mailPassengerEquiv']
    )
    return df

def filter_and_aggregate_by_period(df, start_year, start_month, end_year, end_month, 
                                  group_col, agg_dict, aggregate_by='Year'):
    """Filter data and aggregate by time periods (Month or Year)."""
    filtered_df = filter_data_by_period(df, start_year, start_month, end_year, end_month)
    
    if aggregate_by == 'Month':
        # Aggregate by month and year
        periods = filtered_df[['Year', 'Month']].drop_duplicates().sort_values(['Year', 'Month'])
        results = []
        
        for _, row in periods.iterrows():
            year, month = row['Year'], row['Month']
            period_data = filtered_df[(filtered_df['Year'] == year) & (filtered_df['Month'] == month)]
            aggregated = period_data.groupby(group_col).agg(agg_dict).reset_index()
            results.append((year, month, aggregated))
        
        return results
    else:  # Year
        # Aggregate by year only
        results = []
        for year in sorted(filtered_df['Year'].unique()):
            period_data = filtered_df[filtered_df['Year'] == year]
            aggregated = period_data.groupby(group_col).agg(agg_dict).reset_index()
            results.append((year, None, aggregated))
        
        return results

def get_top_entities(data, sort_col, level='Most', top_n=10):
    """Get top/bottom entities based on sort criteria."""
    if data.empty:
        return data
    
    sort_ascending = (level == 'Least')
    return data.nsmallest(top_n, sort_col) if sort_ascending else data.nlargest(top_n, sort_col)

def format_results_summary(entities, metric_col, entity_col, year, month=None):
    """Format results into a standard summary format."""
    if entities.empty:
        entity_names = 'No data'
        avg_metric = 'No data'
    else:
        entity_names = ', '.join(entities[entity_col].tolist())
        if metric_col is not None and metric_col in entities.columns:
            avg_metric = entities[metric_col].mean()
            avg_metric = f"{avg_metric:.3f}" if isinstance(avg_metric, (int, float)) else str(avg_metric)
        else:
            avg_metric = 'No data'
    
    result = {'Year': year}
    if month is not None:
        result['Month'] = month
    
    return result, entity_names, avg_metric