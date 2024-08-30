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
    df['$ SPENT'] = df['$ SPENT'].replace('[\$,]', '', regex=True).astype(float)
    return df

# Function to load dataset 4
def load_dataset4():
    file_path = r'C:\Users\Migue\assistance-multi-campaign-analysis\sample_data-crm_analysis.csv'
    df = pd.read_csv(file_path)
    df['1/1/22'] = pd.to_datetime(df['1/1/22'])
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
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

# Function to create CRM visualizations
def create_crm_visualizations(df):

    # 2. Medicaid Status Distribution
    medicaid_counts = df['Medicaid Status'].value_counts()
    fig2 = px.pie(values=medicaid_counts.values, names=medicaid_counts.index, title='Medicaid Status Distribution')
    st.plotly_chart(fig2)

    # 3. Age Distribution
    fig3 = px.histogram(df, x='Age', title='Age Distribution')
    st.plotly_chart(fig3)


    # 5. Channel Distribution
    channel_counts = df['Channel'].value_counts()
    fig5 = px.bar(x=channel_counts.index, y=channel_counts.values, title='Channel Distribution')
    st.plotly_chart(fig5)

    # 6. Source Distribution
    source_counts = df['Source'].value_counts()
    fig6 = px.bar(x=source_counts.index, y=source_counts.values, title='Source Distribution')
    st.plotly_chart(fig6)



# Streamlit app
def main():
    st.title("Advertising Data Analysis for Elderly Care Services")

    # Load datasets
    df1 = load_dataset1()
    df2 = load_dataset2()
    df3 = load_dataset3()
    df4 = load_dataset4()

    # Sidebar for user input
    st.sidebar.header("User Input")
    selected_dataset = st.sidebar.selectbox("Select Dataset", ["Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4"])
    
    if selected_dataset == "Dataset 1":
        df = df1
        date_column = 'Week'
    elif selected_dataset == "Dataset 2":
        df = df2
        date_column = 'Date'
    elif selected_dataset == "Dataset 3":
        df = df3
        date_column = 'Week Of'
    else:  # Dataset 4
        df = df4
        date_column = '1/1/22'

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

    elif selected_dataset == "Dataset 3":
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
        
        # Aggregate data by 'Station'
        station_performance = filtered_df.groupby('Station').agg({
            'Ord Spots': 'sum',
            'Spots Ran': 'sum',
            'UNQ >= :01': 'sum',
            '$ SPENT': 'sum'
        }).reset_index()

        # Calculate derived metrics
        station_performance['CPUC'] = station_performance['$ SPENT'] / station_performance['UNQ >= :01']

        # Rename columns for display
        station_performance = station_performance.rename(columns={
            'Station': 'Station',
            '$ SPENT': 'Spend',
            'UNQ >= :01': 'UNQ',
            'Ord Spots': 'Ordered Spots',
            'Spots Ran': 'Spots Ran'
        })

        # Select and order columns
        columns_to_display = ['Station', 'Spend', 'UNQ', 'Ordered Spots', 'Spots Ran', 'CPUC']
        station_performance = station_performance[columns_to_display]

        # Format columns
        station_performance['Spend'] = station_performance['Spend'].apply(lambda x: f'${x:,.2f}')
        station_performance['CPUC'] = station_performance['CPUC'].apply(lambda x: f'${x:,.2f}')

        # Display the table
        st.dataframe(station_performance.style.format({
            'UNQ': '{:,.0f}',
            'Ordered Spots': '{:,.0f}',
            'Spots Ran': '{:,.0f}'
        }))

    else:  # Dataset 4
        st.subheader("CRM Analysis Data")
        st.write("This dataset contains CRM analysis information.")
        create_crm_visualizations(filtered_df)

if __name__ == "__main__":
    main()