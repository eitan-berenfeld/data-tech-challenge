import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utilities import filter_data_by_period

# Simple, clean styling
plt.style.use('default')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def create_investment_priority_analysis(df, start_year, start_month, end_year, end_month):
    """Single chart showing top investment priorities - routes and ports combined."""
    filtered_df = filter_data_by_period(df, start_year, start_month, end_year, end_month)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Get top 10 routes by passenger volume
    route_totals = filtered_df.groupby('Route')['Passengers_Total'].sum().nlargest(10)
    
    # Create simple horizontal bar chart
    bars = ax.barh(range(len(route_totals)), route_totals.values, 
                   color='steelblue', alpha=0.7)
    
    ax.set_yticks(range(len(route_totals)))
    ax.set_yticklabels([route.replace('-', ' → ') for route in route_totals.index])
    ax.set_title('Investment Priority: Top Routes for Capacity Expansion\n(Based on Total Passenger Volume 1985-1989)', 
                 fontsize=14, fontweight='bold', pad=25)
    ax.set_xlabel('Total Passengers')
    
    # Add value labels with better spacing
    max_value = max(route_totals.values)
    for i, (bar, value) in enumerate(zip(bars, route_totals.values)):
        ax.text(value + max_value * 0.02, i, f'{value:,.0f}', 
                va='center', fontsize=10, fontweight='bold')
    
    # Add investment recommendation text with better positioning
    ax.text(0.02, 0.95, 
            'INVESTMENT RECOMMENDATION:\n• Focus on top 3 routes for immediate capacity expansion\n• Routes 4-7 for medium-term growth\n• Monitor routes 8-10 for future opportunities', 
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    return fig

def create_market_expansion_opportunities(df, start_year, start_month, end_year, end_month):
    """Single chart showing geographic expansion opportunities with context note."""
    filtered_df = filter_data_by_period(df, start_year, start_month, end_year, end_month)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    
    # Calculate country performance and growth
    yearly_country = filtered_df.groupby(['Year', 'Country'])['Passengers_Total'].sum().reset_index()
    
    country_metrics = []
    for country in filtered_df['Country'].unique():
        country_data = yearly_country[yearly_country['Country'] == country].sort_values('Year')
        if len(country_data) >= 3:
            total_passengers = country_data['Passengers_Total'].sum()
            # Calculate growth rate
            first_year = country_data.iloc[0]['Passengers_Total']
            last_year = country_data.iloc[-1]['Passengers_Total']
            if first_year > 0:
                growth_rate = ((last_year - first_year) / first_year * 100)
            else:
                growth_rate = 0
            
            country_metrics.append({
                'Country': country,
                'Total_Passengers': total_passengers,
                'Growth_Rate': growth_rate
            })
    
    # Create DataFrame and get top markets by total volume
    metrics_df = pd.DataFrame(country_metrics)
    top_markets = metrics_df.nlargest(12, 'Total_Passengers')
    
    # Create scatter plot - simple and clean
    colors = ['red' if growth > 20 else 'orange' if growth > 0 else 'gray' 
              for growth in top_markets['Growth_Rate']]
    
    scatter = ax.scatter(top_markets['Total_Passengers']/1e6, top_markets['Growth_Rate'],
                        s=120, c=colors, alpha=0.7, edgecolors='black', linewidth=1)
    
    ax.set_title('Market Expansion Strategy: Volume vs Growth Analysis\n(1985-1989 Period Shows Market Consolidation Trends)', 
                 fontsize=14, fontweight='bold', pad=25)
    ax.set_xlabel('Total Passengers (Millions)')
    ax.set_ylabel('Growth Rate (%)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # Add country labels for key markets only to avoid clutter
    # Manually position labels to avoid overlap
    labeled_countries = []
    for _, row in top_markets.iterrows():
        # Only label the most important markets
        if row['Total_Passengers'] > 2e6 or row['Growth_Rate'] > 30:
            # Manual positioning to avoid overlap
            x_pos = row['Total_Passengers']/1e6
            y_pos = row['Growth_Rate']
            
            # Adjust label positions based on country to avoid overlap
            if row['Country'] == 'Japan':
                offset_x, offset_y = 10, 5
            elif row['Country'] == 'New Zealand':
                offset_x, offset_y = -15, 5
            elif row['Country'] == 'Singapore':
                offset_x, offset_y = 10, -10
            elif row['Country'] == 'USA':
                offset_x, offset_y = 8, 8
            else:
                offset_x, offset_y = 8, 8
            
            ax.annotate(row['Country'], 
                       (x_pos, y_pos),
                       xytext=(offset_x, offset_y), textcoords='offset points', 
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Add quadrant labels with better spacing
    ax.text(0.5, 50, 'High Growth\nSmall Market\n(DEVELOP)', 
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='orange', alpha=0.6))
    
    ax.text(7, 50, 'High Growth\nLarge Market\n(EXPAND)', 
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='red', alpha=0.6))
    
    ax.text(7, -30, 'Low Growth\nLarge Market\n(MAINTAIN)', 
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.6))
    
    # Add context note about the period
    ax.text(0.02, 0.02, 
            'NOTE: Negative growth reflects industry consolidation during 1985-1989.\nFocus on countries with positive growth and large market size.',
            transform=ax.transAxes, fontsize=9, va='bottom',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_operational_efficiency_analysis(df, start_year, start_month, end_year, end_month):
    """Hub analysis with improved scaling and visibility for mid-tier ports."""
    filtered_df = filter_data_by_period(df, start_year, start_month, end_year, end_month)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    
    # Port analysis - volume vs efficiency
    port_metrics = filtered_df.groupby('AustralianPort').agg({
        'Passengers_Total': 'sum',
        'Freight_Total_(tonnes)': 'sum',
        'Route': 'nunique',
        'Country': 'nunique'
    }).reset_index()
    
    # Calculate key metrics
    port_metrics['Total_Volume'] = (port_metrics['Passengers_Total'] + 
                                   port_metrics['Freight_Total_(tonnes)'] * 10)
    port_metrics['Efficiency'] = port_metrics['Total_Volume'] / port_metrics['Route']
    
    # Focus on significant ports but separate into tiers for better visibility
    significant_ports = port_metrics[
        port_metrics['Total_Volume'] > port_metrics['Total_Volume'].quantile(0.2)
    ].copy()
    
    # Create tiered analysis - separate major hubs from regional ports
    major_hubs = significant_ports[significant_ports['Total_Volume'] > significant_ports['Total_Volume'].quantile(0.8)]
    regional_ports = significant_ports[significant_ports['Total_Volume'] <= significant_ports['Total_Volume'].quantile(0.8)]
    
    # Plot major hubs (Sydney, Melbourne) with larger markers
    major_colors = ['red' if vol > major_hubs['Total_Volume'].quantile(0.5) else 'orange'
                   for vol in major_hubs['Total_Volume']]
    
    scatter1 = ax.scatter(major_hubs['Route'], major_hubs['Efficiency'],
                         s=major_hubs['Country']*40,  # Larger for visibility
                         c=major_colors, alpha=0.8, edgecolors='black', linewidth=2,
                         label='Major Hubs')
    
    # Plot regional ports with smaller, consistent markers
    scatter2 = ax.scatter(regional_ports['Route'], regional_ports['Efficiency'],
                         s=regional_ports['Country']*25,
                         c='lightblue', alpha=0.7, edgecolors='navy', linewidth=1,
                         label='Regional Ports')
    
    ax.set_title('Hub Development Strategy: Route Network vs Efficiency\n(Size = International Connectivity, Color = Investment Priority)', 
                 fontsize=14, fontweight='bold', pad=25)
    ax.set_xlabel('Number of Routes (Network Size)')
    ax.set_ylabel('Efficiency (Volume per Route)')
    
    # Add port labels with better positioning
    # Label major hubs
    for _, row in major_hubs.iterrows():
        ax.annotate(row['AustralianPort'], 
                   (row['Route'], row['Efficiency']),
                   xytext=(10, 10), textcoords='offset points', 
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Label only top regional ports to avoid clutter
    top_regional = regional_ports.nlargest(4, 'Total_Volume')
    for _, row in top_regional.iterrows():
        ax.annotate(row['AustralianPort'], 
                   (row['Route'], row['Efficiency']),
                   xytext=(5, 5), textcoords='offset points', 
                   fontsize=9, fontweight='bold')
    
    # Add strategic quadrants with better positioning
    if len(significant_ports) > 0:
        route_median = significant_ports['Route'].median()
        efficiency_median = significant_ports['Efficiency'].median()
        ax.axvline(route_median, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(efficiency_median, color='gray', linestyle='--', alpha=0.5)
        
        # Position quadrant labels to avoid overlap
        ax.text(route_median + 5, efficiency_median + efficiency_median*0.3, 
                'EXPAND\n(High Network\nHigh Efficiency)', 
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgreen', alpha=0.7))
        
        ax.text(route_median - 8, efficiency_median + efficiency_median*0.3, 
                'OPTIMIZE\n(Low Network\nHigh Efficiency)', 
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor='yellow', alpha=0.7))
    
    # Add legend
    ax.legend(loc='lower right', fontsize=10)
    
    # Add investment insights
    ax.text(0.02, 0.98, 
            'KEY INSIGHTS:\n• Sydney: Dominant hub requiring capacity investment\n• Melbourne: Efficient secondary hub for expansion\n• Regional ports: Optimize route networks',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    return fig