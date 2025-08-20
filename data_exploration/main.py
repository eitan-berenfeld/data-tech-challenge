import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

from utilities import load_and_clean_data, create_datetime_index
from analysis_functions import (
    analyze_traffic_routes, analyze_port_flow_efficiency, 
    analyze_hub_utilization, analyze_geographical_patterns
)
from visualizations import (
    create_investment_priority_analysis,
    create_market_expansion_opportunities,
    create_operational_efficiency_analysis
)

def main():
    """
    Understanding the Data: Traffic Routes Analysis (1985-1989)
    a) Identify the most and least trafficked routes
    b) Analyze trends and/or geographical patterns  
    c) Create visualizations to demonstrate trends & patterns
    """
    
    # Configuration
    DATA_FILE = 'TechChallenge_Data.csv'
    START_YEAR = 1985
    START_MONTH = 1
    END_YEAR = 1989
    END_MONTH = 12
    
    # Load and prepare data
    df = load_and_clean_data(DATA_FILE)
    df = create_datetime_index(df)
    
    # Export cleaned data as a CSV file
    cleaned_data_file = 'cleaned_traffic_data.csv'
    df.to_csv(cleaned_data_file, index=False)

    # Create output directory
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # a) IDENTIFY MOST AND LEAST TRAFFICKED ROUTES
    print("a) MOST AND LEAST TRAFFICKED ROUTES")
    
    # Most trafficked routes by year
    most_trafficked = analyze_traffic_routes(
        df, START_YEAR, START_MONTH, END_YEAR, END_MONTH, 
        'Most', 'Passengers', 'Year'
    )
    print("MOST TRAFFICKED ROUTES BY YEAR:")
    print(most_trafficked[['Year', 'Total_Route', 'Total_Value']].to_string(index=False))
    
    # Least trafficked routes by year
    least_trafficked = analyze_traffic_routes(
        df, START_YEAR, START_MONTH, END_YEAR, END_MONTH, 
        'Least', 'Passengers', 'Year'
    )
    print("\nLEAST TRAFFICKED ROUTES BY YEAR:")
    print(least_trafficked[['Year', 'Total_Route', 'Total_Value']].to_string(index=False))
    
    # b) ANALYZE TRENDS AND GEOGRAPHICAL PATTERNS
    print("\n\nb) TRENDS AND GEOGRAPHICAL PATTERNS")
    
    # Country performance analysis
    country_performance = analyze_geographical_patterns(
        df, START_YEAR, START_MONTH, END_YEAR, END_MONTH, 
        'country_performance', top_n=10
    )
    print("TOP COUNTRIES BY TOTAL PASSENGER VOLUME:")
    print(country_performance[['Country', 'Passengers_Total', 'Route']].to_string(index=False))
    
    # Regional growth analysis
    regional_growth = analyze_geographical_patterns(
        df, START_YEAR, START_MONTH, END_YEAR, END_MONTH, 
        'regional_growth', top_n=10
    )
    print("\nCOUNTRY GROWTH RATES (1985-1989):")
    if not regional_growth.empty:
        print(regional_growth[['Country', 'growth_rate', 'Passengers_Total']].to_string(index=False))
    
    # Port connectivity analysis
    port_connectivity = analyze_geographical_patterns(
        df, START_YEAR, START_MONTH, END_YEAR, END_MONTH, 
        'port_connectivity', top_n=10
    )
    print("\nPORT CONNECTIVITY ANALYSIS:")
    print(port_connectivity[['AustralianPort', 'Country', 'ForeignPort', 'Passengers_Total']].to_string(index=False))
    
    # c) CREATE VISUALIZATIONS    
    plt.ioff()  # Turn off interactive mode
    
    # Route traffic analysis Visualization
    fig1 = create_investment_priority_analysis(df, START_YEAR, START_MONTH, END_YEAR, END_MONTH)
    fig1.savefig(f'{output_dir}/01_route_traffic_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    
    # Geographical patterns Visualization
    fig2 = create_market_expansion_opportunities(df, START_YEAR, START_MONTH, END_YEAR, END_MONTH)
    fig2.savefig(f'{output_dir}/02_geographical_patterns.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    
    # Hub and operational analysis Visualization
    fig3 = create_operational_efficiency_analysis(df, START_YEAR, START_MONTH, END_YEAR, END_MONTH)
    fig3.savefig(f'{output_dir}/03_hub_operational_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"Visualizations saved to: {output_dir}/")

if __name__ == "__main__":
    main()