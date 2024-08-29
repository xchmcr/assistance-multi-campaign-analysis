import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Function to load and preprocess dataset 1
def load_dataset1():
    file_path = r'C:\Users\Migue\assistance-multi-campaign-analysis\sample_data-data.csv'
    df = pd.read_csv(file_path)
    df['Spend'] = df['Spend'].replace('[\$,]', '', regex=True).astype(float)
    df['Week'] = pd.to_datetime(df['Week'])
    return df

# Function to load dataset 2
def load_dataset2():
    file_path = r'C:\Users\Migue\assistance-multi-campaign-analysis\sample_data-gadata.csv'
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Function to load dataset 3
def load_dataset3():
    file_path = r'C:\Users\Migue\assistance-multi-campaign-analysis\sample_data-radiodata.csv'
    df = pd.read_csv(file_path)
    df['Week Of'] = pd.to_datetime(df['Week Of'])
    return df

# Function to create weekly report for dataset 1
def create_weekly_report(df):
    weekly_report = df.groupby('Week').agg({
        'Spend': 'sum',
        'Leads (from CRM)': 'sum',
        'Ascend Application': 'sum',
        'Approved for Services': 'sum'
    }).reset_index()
    
    weekly_report['CPI'] = weekly_report['Spend'] / weekly_report['Leads (from CRM)']
    weekly_report['CPA Ascend App'] = weekly_report['Spend'] / weekly_report['Ascend Application']
    weekly_report['CPApproved'] = weekly_report['Spend'] / weekly_report['Approved for Services']
    
    weekly_report.replace([float('inf'), -float('inf')], 'N/A', inplace=True)
    return weekly_report

# Function to create channel summary for dataset 1
def create_channel_summary(df):
    channel_summary = df.groupby('Channel').agg({
        'Spend': 'sum',
        'Leads (from CRM)': 'sum',
        'Ascend Application': 'sum',
        'Approved for Services': 'sum'
    }).reset_index()
    
    channel_summary['CPL'] = channel_summary['Spend'] / channel_summary['Leads (from CRM)']
    channel_summary['CPA Ascend App'] = channel_summary['Spend'] / channel_summary['Ascend Application']
    channel_summary['CPA Approved'] = channel_summary['Spend'] / channel_summary['Approved for Services']
    
    # Add grand total row
    grand_total = channel_summary.sum(numeric_only=True).to_frame().T
    grand_total['Channel'] = 'Grand Total'
    channel_summary = pd.concat([channel_summary, grand_total]).reset_index(drop=True)
    
    channel_summary.replace([float('inf'), -float('inf')], None, inplace=True)
    return channel_summary

# Streamlit app
def main():
    st.title("Advertising Data Analysis for Elderly Care Services")

    # Load datasets
    df1 = load_dataset1()
    df2 = load_dataset2()
    df3 = load_dataset3()

    # Sidebar for user input
    st.sidebar.header("User Input")
    selected_dataset = st.sidebar.selectbox("Select Dataset", ["Dataset 1", "Dataset 2", "Dataset 3"])
    
    if selected_dataset == "Dataset 1":
        df = df1
        date_column = 'Week'
    elif selected_dataset == "Dataset 2":
        df = df2
        date_column = 'Date'
    else:
        df = df3
        date_column = 'Week Of'

    # Date range selection
    min_date = df[date_column].min().date()
    max_date = df[date_column].max().date()
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

    # Filter data based on date range
    mask = (df[date_column].dt.date >= start_date) & (df[date_column].dt.date <= end_date)
    filtered_df = df.loc[mask]

    # Display raw data
    st.subheader("Raw Data")
    st.write(filtered_df)

    # Create visualizations
    st.subheader("Data Visualizations")

    if selected_dataset == "Dataset 1":
        # Weekly report
        st.subheader("Weekly Report")
        weekly_report = create_weekly_report(filtered_df)
        st.write(weekly_report)

        # Line and Bar Combination Chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=weekly_report['Week'], y=weekly_report['Spend'], name="Total Cost", marker_color='green'), secondary_y=False)
        fig.add_trace(go.Scatter(x=weekly_report['Week'], y=weekly_report['CPI'], name="Cost per Lead", marker_color='blue'), secondary_y=True)
        fig.update_layout(title_text="Weekly Performance: Total Cost vs Cost per Lead")
        fig.update_xaxes(title_text="Week")
        fig.update_yaxes(title_text="Total Cost", secondary_y=False)
        fig.update_yaxes(title_text="Cost per Lead", secondary_y=True)
        st.plotly_chart(fig)

        # Channel summary
        st.subheader("Channel Summary")
        channel_summary = create_channel_summary(filtered_df)
        st.write(channel_summary)

    elif selected_dataset == "Dataset 2":
        # Table Visualization for Media Buying Data
        st.subheader("Media Buying Data Analysis")
        media_buying_data = filtered_df.groupby('Session source - GA4').agg({
            'Sessions - GA4, event based': 'sum',
            'Event count - GA4': 'sum',
            'Event value - GA4 (USD)': 'sum'
        }).reset_index()
        media_buying_data['CPS'] = media_buying_data['Event value - GA4 (USD)'] / media_buying_data['Sessions - GA4, event based']
        media_buying_data['CPE'] = media_buying_data['Event value - GA4 (USD)'] / media_buying_data['Event count - GA4']
        st.write(media_buying_data)

        # Add a useful chart (example: bar chart of sessions by source)
        fig = px.bar(media_buying_data, x='Session source - GA4', y='Sessions - GA4, event based', 
                     title='Sessions by Source')
        st.plotly_chart(fig)

    else:  # Dataset 3
        # Region filter
        regions = filtered_df['Market'].unique()
        selected_regions = st.multiselect('Select Regions', regions, default=regions)
        
        # Filter data based on selected regions
        region_filtered_df = filtered_df[filtered_df['Market'].isin(selected_regions)]

        # Stacked Bar Chart for Breakdown Over Time
        fig = px.bar(region_filtered_df, x='Week Of', y='$ SPENT', color='Market',
                     title='Cost Breakdown by Region Over Time',
                     labels={'$ SPENT': 'Spend', 'Week Of': 'Date'})
        fig.update_layout(barmode='stack', xaxis_title="Date", yaxis_title="Spend")
        st.plotly_chart(fig)

        # Radio Station Performance Table
        st.subheader("Radio Station Performance")
        
        # Check available columns
        available_columns = filtered_df.columns
        st.write("Available columns:", available_columns)

        # Define column mappings
        column_mappings = {
            'Station': 'Station',
            'Spend': '$ SPENT',
            'UNQ': 'UNQ >=' if 'UNQ >=' in available_columns else 'UNQ',
            'Submitted Apps': 'Submitted Apps' if 'Submitted Apps' in available_columns else None,
            'Approved Apps': 'Approved Apps' if 'Approved Apps' in available_columns else None
        }

        # Create the performance table
        station_performance = filtered_df.groupby('Station').agg({
            column_mappings['Spend']: 'sum',
            column_mappings['UNQ']: 'sum'
        }).reset_index()

        # Calculate metrics
        station_performance['CPUC'] = station_performance[column_mappings['Spend']] / station_performance[column_mappings['UNQ']]
        
        if column_mappings['Submitted Apps']:
            station_performance['Submitted Apps'] = filtered_df.groupby('Station')[column_mappings['Submitted Apps']].sum()
            station_performance['CPSubmitted'] = station_performance[column_mappings['Spend']] / station_performance['Submitted Apps']
            station_performance['CNV1%'] = station_performance['Submitted Apps'] / station_performance[column_mappings['UNQ']] * 100
        
        if column_mappings['Approved Apps']:
            station_performance['Approved Apps'] = filtered_df.groupby('Station')[column_mappings['Approved Apps']].sum()
            station_performance['CPApproved'] = station_performance[column_mappings['Spend']] / station_performance['Approved Apps']
            station_performance['CNV2%'] = station_performance['Approved Apps'] / station_performance[column_mappings['UNQ']] * 100

        # Rename columns for display
        station_performance = station_performance.rename(columns={
            column_mappings['Spend']: 'Spend',
            column_mappings['UNQ']: 'UNQ'
        })

        # Display the table
        st.write(station_performance)

    # ... (rest of the code remains the same)

if __name__ == "__main__":
    main()