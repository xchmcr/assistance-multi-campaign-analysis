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

# Function to create weekly aggregated data for dataset 1
def create_weekly_data(df):
    weekly_data = df.groupby('Week').agg({
        'Spend': 'sum',
        'Leads (from CRM)': 'sum',
        'Ascend Application': 'sum',
        'Approved for Services': 'sum'
    }).reset_index()
    
    weekly_data['CPL'] = weekly_data['Spend'] / weekly_data['Leads (from CRM)']
    weekly_data['CPA Ascend App'] = weekly_data['Spend'] / weekly_data['Ascend Application']
    weekly_data['CPA Approved'] = weekly_data['Spend'] / weekly_data['Approved for Services']
    
    weekly_data.replace([float('inf'), -float('inf')], None, inplace=True)
    return weekly_data

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
        # Statistic labels
        col1, col2, col3 = st.columns(3)
        total_ascend = filtered_df['Ascend Application'].sum()
        total_spend = filtered_df['Spend'].sum()
        cpa_ascend = total_spend / total_ascend if total_ascend > 0 else 0
        
        total_approved = filtered_df['Approved for Services'].sum()
        cpa_approved = total_spend / total_approved if total_approved > 0 else 0
        
        conversion_rate = total_approved / total_ascend if total_ascend > 0 else 0
        
        col1.metric("Ascend Apps", f"{total_ascend:.0f}", f"${cpa_ascend:.2f} per app")
        col2.metric("Approved Apps", f"{total_approved:.0f}", f"${cpa_approved:.2f} per app")
        col3.metric("Conversion Rate", f"{conversion_rate:.2%}")

        # Line and Bar Combination Chart
        weekly_data = create_weekly_data(filtered_df)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=weekly_data['Week'], y=weekly_data['Spend'], name="Total Cost", marker_color='green'), secondary_y=False)
        fig.add_trace(go.Scatter(x=weekly_data['Week'], y=weekly_data['CPL'], name="Cost per Lead", marker_color='blue'), secondary_y=True)
        fig.update_layout(title_text="Weekly Performance: Total Cost vs Cost per Lead")
        fig.update_xaxes(title_text="Week")
        fig.update_yaxes(title_text="Total Cost", secondary_y=False)
        fig.update_yaxes(title_text="Cost per Lead", secondary_y=True)
        st.plotly_chart(fig)

        # Weekly report table
        st.subheader("Weekly Report")
        st.write(weekly_data)

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
        fig.update_layout(barmode='stack')
        st.plotly_chart(fig)

        # Radio Station Performance Table
        st.subheader("Radio Station Performance")
        station_performance = filtered_df.groupby('Station').agg({
            '$ SPENT': 'sum',
            'UNQ': 'sum',
            'Submitted Apps': 'sum',
            'Approved Apps': 'sum'
        }).reset_index()
        
        station_performance['CPUC'] = station_performance['$ SPENT'] / station_performance['UNQ']
        station_performance['CPSubmitted'] = station_performance['$ SPENT'] / station_performance['Submitted Apps']
        station_performance['CPApproved'] = station_performance['$ SPENT'] / station_performance['Approved Apps']
        station_performance['CNV1%'] = station_performance['Submitted Apps'] / station_performance['UNQ'] * 100
        station_performance['CNV2%'] = station_performance['Approved Apps'] / station_performance['UNQ'] * 100
        
        station_performance = station_performance.rename(columns={'$ SPENT': 'Spend'})
        st.write(station_performance)

if __name__ == "__main__":
    main()