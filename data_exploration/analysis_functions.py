import pandas as pd
import numpy as np
from utilities import (
    filter_data_by_period, get_cargo_columns, get_default_cargo_weights,
    calculate_passenger_equivalents, filter_and_aggregate_by_period,
    get_top_entities, format_results_summary
)

def analyze_traffic_routes(df, start_year, start_month, end_year, end_month, 
                          traffic_level, traffic_type, aggregate_by='Month'):
    """
    Analyze traffic data by route and return most/least trafficked routes.
    """
    col_mapping = {
        'Passengers': ('Passengers_In', 'Passengers_Out', 'Passengers_Total', ''),
        'Freight': ('Freight_In_(tonnes)', 'Freight_Out_(tonnes)', 'Freight_Total_(tonnes)', 't'),
        'Mail': ('Mail_In_(tonnes)', 'Mail_Out_(tonnes)', 'Mail_Total_(tonnes)', 't')
    }
    
    if traffic_type not in col_mapping:
        raise ValueError("traffic_type must be 'Passengers', 'Freight', or 'Mail'")
    if traffic_level not in ['Most', 'Least']:
        raise ValueError("traffic_level must be 'Most' or 'Least'")
    if aggregate_by not in ['Month', 'Year']:
        raise ValueError("aggregate_by must be 'Month' or 'Year'")
    
    in_col, out_col, total_col, unit_suffix = col_mapping[traffic_type]
    
    filtered_df = df[
        ((df['Year'] == start_year) & (df['Month'] >= start_month)) |
        ((df['Year'] > start_year) & (df['Year'] < end_year)) |
        ((df['Year'] == end_year) & (df['Month'] <= end_month))
    ].copy()
    
    results = []
    
    if aggregate_by == 'Month':
        for year, month in filtered_df[['Year', 'Month']].drop_duplicates().sort_values(['Year', 'Month']).values:
            period_data = filtered_df[(filtered_df['Year'] == year) & (filtered_df['Month'] == month)]
            result_row = {'Year': year, 'Month': month}
            
            route_data = period_data.groupby('Route')[[in_col, out_col, total_col]].sum()
            
            for direction, col in [('In', in_col), ('Out', out_col), ('Total', total_col)]:
                if not route_data.empty and col in route_data.columns:
                    extreme_value = route_data[col].max() if traffic_level == 'Most' else route_data[col].min()
                    tied_routes = route_data[route_data[col] == extreme_value].index.tolist()
                    
                    routes_str = ', '.join(tied_routes)
                    formatted_value = f"{int(extreme_value)}" if not unit_suffix else f"{extreme_value:.1f}{unit_suffix}"
                    
                    result_row.update({f'{direction}_Route': routes_str, f'{direction}_Value': formatted_value})
                else:
                    result_row.update({f'{direction}_Route': 'No data', f'{direction}_Value': 'No data'})
            
            results.append(result_row)
    
    else:  # aggregate_by == 'Year'
        for year in sorted(filtered_df['Year'].unique()):
            period_data = filtered_df[filtered_df['Year'] == year]
            result_row = {'Year': year}
            
            route_data = period_data.groupby('Route')[[in_col, out_col, total_col]].sum()
            
            for direction, col in [('In', in_col), ('Out', out_col), ('Total', total_col)]:
                if not route_data.empty and col in route_data.columns:
                    extreme_value = route_data[col].max() if traffic_level == 'Most' else route_data[col].min()
                    tied_routes = route_data[route_data[col] == extreme_value].index.tolist()
                    
                    routes_str = ', '.join(tied_routes)
                    formatted_value = f"{int(extreme_value)}" if not unit_suffix else f"{extreme_value:.1f}{unit_suffix}"
                    
                    result_row.update({f'{direction}_Route': routes_str, f'{direction}_Value': formatted_value})
                else:
                    result_row.update({f'{direction}_Route': 'No data', f'{direction}_Value': 'No data'})
            
            results.append(result_row)
    
    return pd.DataFrame(results)

def analyze_port_flow_efficiency(df, start_year, start_month, end_year, end_month, 
                                cargo_types=['passengers', 'freight', 'mail'],
                                aggregate_by='Year', efficiency_level='Most',
                                specific_ports=None, min_volume_threshold=0, top_n=10):
    """Analyze port flow efficiency (in vs out balance)."""
    
    filtered_df = filter_data_by_period(df, start_year, start_month, end_year, end_month)
    if specific_ports:
        filtered_df = filtered_df[filtered_df['AustralianPort'].isin(specific_ports)]
    
    cargo_mapping = get_cargo_columns()
    
    agg_dict = {}
    for cargo_type in cargo_types:
        if cargo_type in cargo_mapping:
            mapping = cargo_mapping[cargo_type]
            agg_dict.update({mapping['in']: 'sum', mapping['out']: 'sum'})
    
    period_results = filter_and_aggregate_by_period(
        filtered_df, start_year, start_month, end_year, end_month, 'AustralianPort', agg_dict, aggregate_by
    )
    
    results = []
    for year, month, port_data in period_results:
        result_row, _, _ = format_results_summary(port_data, None, 'AustralianPort', year, month)
        
        for cargo_type in cargo_types:
            if cargo_type in cargo_mapping:
                mapping = cargo_mapping[cargo_type]
                
                if mapping['in'] in port_data.columns and mapping['out'] in port_data.columns:
                    epsilon = 1 if cargo_type == 'passengers' else 0.001
                    port_data[f'{cargo_type}_balance_ratio'] = (
                        port_data[mapping['in']] / (port_data[mapping['out']] + epsilon)
                    )
                    port_data[f'{cargo_type}_efficiency'] = (
                        1 - abs(1 - port_data[f'{cargo_type}_balance_ratio'])
                    )
                    port_data[f'{cargo_type}_total_volume'] = (
                        port_data[mapping['in']] + port_data[mapping['out']]
                    )
                    
                    filtered_ports = port_data[port_data[f'{cargo_type}_total_volume'] >= min_volume_threshold]
                    top_ports = get_top_entities(filtered_ports, f'{cargo_type}_efficiency', efficiency_level, top_n)
                    
                    if not top_ports.empty:
                        port_names = ', '.join(top_ports['AustralianPort'].tolist())
                        avg_efficiency = top_ports[f'{cargo_type}_efficiency'].mean()
                        result_row.update({
                            f'{cargo_type.title()}_Ports': port_names,
                            f'{cargo_type.title()}_Avg_Efficiency': f"{avg_efficiency:.3f}"
                        })
                    else:
                        result_row.update({
                            f'{cargo_type.title()}_Ports': 'No data',
                            f'{cargo_type.title()}_Avg_Efficiency': 'No data'
                        })
        
        results.append(result_row)
    
    return pd.DataFrame(results)

def analyze_hub_utilization(df, start_year, start_month, end_year, end_month,
                           aggregate_by='Year', utilization_level='Most',
                           specific_ports=None, min_routes_threshold=1,
                           cargo_weights=None, top_n=10):
    """Analyze hub utilization levels."""
    
    if cargo_weights is None:
        cargo_weights = get_default_cargo_weights()
    
    filtered_df = filter_data_by_period(df, start_year, start_month, end_year, end_month)
    if specific_ports:
        filtered_df = filtered_df[filtered_df['AustralianPort'].isin(specific_ports)]
    
    agg_dict = {
        'Passengers_Total': 'sum',
        'Freight_Total_(tonnes)': 'sum',
        'Mail_Total_(tonnes)': 'sum',
        'Route': 'nunique'
    }
    
    period_results = filter_and_aggregate_by_period(
        filtered_df, start_year, start_month, end_year, end_month, 'AustralianPort', agg_dict, aggregate_by
    )
    
    results = []
    for year, month, hub_data in period_results:
        result_row, _, _ = format_results_summary(hub_data, None, 'AustralianPort', year, month)
        
        if not hub_data.empty:
            hub_data.columns = ['Port', 'Total_Passengers', 'Total_Freight', 'Total_Mail', 'Unique_Routes']
            hub_data = hub_data[hub_data['Unique_Routes'] >= min_routes_threshold]
            
            if not hub_data.empty:
                hub_data['totalPassengerEquiv'] = (
                    hub_data['Total_Passengers'] + 
                    hub_data['Total_Freight'] * cargo_weights['freight'] + 
                    hub_data['Total_Mail'] * cargo_weights['mail']
                )
                hub_data['utilizationScore'] = hub_data['totalPassengerEquiv'] / hub_data['Unique_Routes']
                
                top_hubs = get_top_entities(hub_data, 'utilizationScore', utilization_level, top_n)
                
                hub_names = ', '.join(top_hubs['Port'].tolist())
                avg_utilization = top_hubs['utilizationScore'].mean()
                avg_routes = top_hubs['Unique_Routes'].mean()
                
                result_row.update({
                    'Hub_Ports': hub_names,
                    'Avg_Utilization_Score': f"{avg_utilization:.1f}",
                    'Avg_Routes': f"{avg_routes:.1f}"
                })
            else:
                result_row.update({
                    'Hub_Ports': 'No data',
                    'Avg_Utilization_Score': 'No data',
                    'Avg_Routes': 'No data'
                })
        else:
            result_row.update({
                'Hub_Ports': 'No data',
                'Avg_Utilization_Score': 'No data',
                'Avg_Routes': 'No data'
            })
        
        results.append(result_row)
    
    return pd.DataFrame(results)

def analyze_geographical_patterns(df, start_year, start_month, end_year, end_month,
                                 pattern_type='country_performance', top_n=10):
    """Analyze geographical patterns in traffic data."""
    
    filtered_df = filter_data_by_period(df, start_year, start_month, end_year, end_month)
    
    if pattern_type == 'country_performance':
        country_totals = filtered_df.groupby('Country').agg({
            'Passengers_Total': 'sum',
            'Freight_Total_(tonnes)': 'sum',
            'Mail_Total_(tonnes)': 'sum',
            'Route': 'nunique'
        }).reset_index()
        
        country_totals = country_totals.sort_values('Passengers_Total', ascending=False).head(top_n)
        return country_totals
    
    elif pattern_type == 'port_connectivity':
        port_connectivity = filtered_df.groupby('AustralianPort').agg({
            'Country': 'nunique',
            'ForeignPort': 'nunique',
            'Passengers_Total': 'sum'
        }).reset_index()
        
        port_connectivity = port_connectivity.sort_values('Country', ascending=False).head(top_n)
        return port_connectivity
    
    elif pattern_type == 'regional_growth':
        yearly_country = filtered_df.groupby(['Year', 'Country']).agg({
            'Passengers_Total': 'sum'
        }).reset_index()
        
        # Calculate growth rates
        yearly_country['prev_year_passengers'] = yearly_country.groupby('Country')['Passengers_Total'].shift(1)
        yearly_country['growth_rate'] = (
            (yearly_country['Passengers_Total'] - yearly_country['prev_year_passengers']) / 
            yearly_country['prev_year_passengers'] * 100
        )
        
        latest_year = yearly_country['Year'].max()
        latest_growth = yearly_country[yearly_country['Year'] == latest_year].dropna()
        
        return latest_growth.sort_values('growth_rate', ascending=False).head(top_n)
    
    return pd.DataFrame()