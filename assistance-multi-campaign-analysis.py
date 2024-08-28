import streamlit as st
import pandas as pd
import plotly.express as px
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
        weekly_data = create_weekly_data(filtered_df)
        
        # Line chart for Spend over time
        fig_spend = px.line(weekly_data, x='Week', y='Spend', title='Weekly Spend Over Time')
        st.plotly_chart(fig_spend)

        # Bar chart for Leads, Ascend Applications, and Approved Services
        fig_metrics = px.bar(weekly_data, x='Week', y=['Leads (from CRM)', 'Ascend Application', 'Approved for Services'],
                             title='Weekly Metrics Comparison')
        st.plotly_chart(fig_metrics)

        # Line chart for CPL, CPA Ascend App, and CPA Approved
        fig_cpa = px.line(weekly_data, x='Week', y=['CPL', 'CPA Ascend App', 'CPA Approved'],
                          title='Cost Per Action Metrics Over Time')
        st.plotly_chart(fig_cpa)

    elif selected_dataset == "Dataset 2":
        # Pie chart for Session source distribution
        fig_source = px.pie(filtered_df, names='Session source - GA4', title='Distribution of Session Sources')
        st.plotly_chart(fig_source)

        # Bar chart for Event count by Session campaign
        fig_campaign = px.bar(filtered_df.groupby('Session campaign - GA4')['Event count - GA4'].sum().reset_index(), 
                              x='Session campaign - GA4', y='Event count - GA4', 
                              title='Event Count by Session Campaign')
        st.plotly_chart(fig_campaign)

        # Line chart for Sessions over time
        daily_sessions = filtered_df.groupby('Date')['Sessions - GA4, event based'].sum().reset_index()
        fig_sessions = px.line(daily_sessions, x='Date', y='Sessions - GA4, event based', title='Daily Sessions Over Time')
        st.plotly_chart(fig_sessions)

    else:  # Dataset 3
        # Bar chart for Spots Ran by Station
        fig_spots = px.bar(filtered_df.groupby('Station')['Spots Ran'].sum().reset_index(), 
                           x='Station', y='Spots Ran', title='Spots Ran by Station')
        st.plotly_chart(fig_spots)

        # Scatter plot for Ordered Buy Rate vs Actual Spend
        fig_spend = px.scatter(filtered_df, x='$ ORD', y='$ SPENT', color='Station',
                               title='Ordered Buy Rate vs Actual Spend by Station')
        st.plotly_chart(fig_spend)

        # Line chart for Unique Listeners over time
        weekly_listeners = filtered_df.groupby('Week Of')['UNQ >= :01'].sum().reset_index()
        fig_listeners = px.line(weekly_listeners, x='Week Of', y='UNQ >= :01', title='Weekly Unique Listeners')
        st.plotly_chart(fig_listeners)

if __name__ == "__main__":
    main()