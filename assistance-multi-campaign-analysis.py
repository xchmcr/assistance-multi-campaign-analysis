import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

def load_dataset1():
    file_url = 'https://raw.githubusercontent.com/xchmcr/assistance-multi-campaign-analysis-repo/workingbranch/sample_data-data.csv'
    df = pd.read_csv(file_url)
    df['Spend'] = df['Spend'].replace('[\$,]', '', regex=True).astype(float)
    df['Week'] = pd.to_datetime(df['Week'])
    return df

def load_dataset2():
    file_url = 'https://raw.githubusercontent.com/xchmcr/assistance-multi-campaign-analysis-repo/workingbranch/sample_data-gadata.csv'
    df = pd.read_csv(file_url)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def load_dataset3():
    file_url = 'https://raw.githubusercontent.com/xchmcr/assistance-multi-campaign-analysis-repo/workingbranch/sample_data-radiodata.csv'
    df = pd.read_csv(file_url)
    df['Week Of'] = pd.to_datetime(df['Week Of'])
    df['$ SPENT'] = df['$ SPENT'].replace('[\$,]', '', regex=True).astype(float)
    return df

def load_dataset4():
    file_url = 'https://raw.githubusercontent.com/xchmcr/assistance-multi-campaign-analysis-repo/workingbranch/sample_data-crm_analysis.csv'
    df = pd.read_csv(file_url)
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

# Function to display conversation types and intentions
def display_conversation_types():
    # Define conversation types and their intentions
    conversation_data = {
        'Employment Inquiries': [
            "Looking for employment, Spanish speaking, connecting with Stephania.",
            "Wants to apply for homecare position.",
            "Looking for employment in Bridgeport area.",
            "Looking for employment, lives in Norwalk, CT.",
            "Looking for live-in case, lives in Hartford, CT.",
            "Looking for employment.",
            "Looking for employment in Alabama.",
            "Wants a job.",
            "Looking for AFL in Mass.",
            "Looking for work but located in Alabama."
        ],
        'Medical and Care-Related Inquiries': [
            "Wanted to know where we get our medical supplies from.",
            "Wanted to get paid for taking care of a sister who is only 24 and had surgery.",
            "MIL is 86 years old, working on changing the address.",
            "Son is autistic, 22 years old, has Medicaid.",
            "Mother on Medicaid, interested in the program, calling back for Med ID.",
            "Wants husband to be the caregiver, but it's against program policy.",
            "Wife's grandmother lives together, over 65, not on Medicaid, over income.",
            "Son takes care of him, only works PT, 62 years old, on Medicare.",
            "Client is only 55, needs info on PCA waiver.",
            "Mom lives in Virginia, has stage 4 cancer.",
            "Mom on Medicaid, family not sure if she wants to do it for fear of losing Husky insurance.",
            "Registered nurse calling on behalf of his patient, way over income but interested in talking about private pay."
        ],
        'Incomplete or Failed Communications': [
            "Call could not be completed - error message.",
            "No answer, left voicemail.",
            "Sent an email - no answer when called.",
            "Contact number provided has a digit missing.",
            "Sent an email - unable to call the number.",
            "Vmail, left voice and text message.",
            "Missed call, left voicemail.",
            "Called back, left message.",
            "No answer, no voicemail, text sent, no response.",
            "Phone number not valid."
        ],
        'Spam or Wrong Numbers': [
            "As contacted individual on 8/26 but the number provided is of YMCA.",
            "This is spam, I called and it said to block the number.",
            "Wrong number.",
            "Phone number not valid.",
            "Wrong contact information."
        ],
        'Inquiries about Program Eligibility': [
            "Wanted to check if she could get services without being on Medicaid.",
            "Grandparents in FL, wants to move them here, not sure if they are on Medicaid.",
            "Wanted to know how much we pay.",
            "House $250,000, $2700 total income, suggested to fill out Medicaid application.",
            "Over asset, mom owns home, also resides in Mass.",
            "Caller wanted her friend to take care but they don’t live together.",
            "Over income, informed to apply for Medicaid and get a spend-down amount.",
            "Son takes care of him, only works PT, 62 years old, on Medicare.",
            "Wanted to take care of his son but the son is only 30 years old."
        ]
    }

    # Display as text
    st.subheader("Conversation Types and Intentions")
    for category, intentions in conversation_data.items():
        st.write(f"**{category}**")
        for intention in intentions:
            st.write(f"- {intention}")
        st.write("")

    # Optionally, display as a chart
    chart_data = {category: len(intentions) for category, intentions in conversation_data.items()}
    chart_df = pd.DataFrame(list(chart_data.items()), columns=['Conversation Type', 'Count'])
    fig = px.bar(chart_df, x='Conversation Type', y='Count', title="Count of Inquiries by Conversation Type")
    st.plotly_chart(fig)

# Streamlit app
def main():
    st.title("Advertising Data Analysis for Care Services Campaign")


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

        # Bar chart of sessions by source
        fig = px.bar(media_buying_data, x='Session source - GA4', y='Sessions - GA4, event based', title='Sessions by Source')
        st.plotly_chart(fig)

    elif selected_dataset == "Dataset 3":
        # Region filter
        regions = filtered_df['Market'].unique()
        selected_regions = st.multiselect('Select Regions', regions, default=regions)
        
        # Filter data based on selected regions
        region_filtered_df = filtered_df[filtered_df['Market'].isin(selected_regions)]

        # Stacked Bar Chart for Breakdown Over Time
        fig = px.bar(region_filtered_df, x='Week Of', y='$ SPENT', color='Market', title='Cost Breakdown by Region Over Time', labels={'$ SPENT': 'Spend', 'Week Of': 'Date'})
        fig.update_layout(barmode='stack', xaxis_title="Date", yaxis_title="Spend")
        st.plotly_chart(fig)

        # Radio Station Performance Table
        st.subheader("Radio Station Performance")
        station_performance = region_filtered_df.groupby('Station').agg({
            'Ord Spots': 'sum',
            'Spots Ran': 'sum',
            'UNQ >= :01': 'sum',
            '$ SPENT': 'sum'
        }).reset_index()

        station_performance['CPUC'] = station_performance['$ SPENT'] / station_performance['UNQ >= :01']
        station_performance = station_performance.rename(columns={
            'Station': 'Station',
            '$ SPENT': 'Spend',
            'UNQ >= :01': 'UNQ',
            'Ord Spots': 'Ordered Spots',
            'Spots Ran': 'Spots Ran'
        })
        columns_to_display = ['Station', 'Spend', 'UNQ', 'Ordered Spots', 'Spots Ran', 'CPUC']
        station_performance = station_performance[columns_to_display]

        station_performance['Spend'] = station_performance['Spend'].apply(lambda x: f'${x:,.2f}')
        station_performance['CPUC'] = station_performance['CPUC'].apply(lambda x: f'${x:,.2f}')

        st.dataframe(station_performance.style.format({
            'UNQ': '{:,.0f}',
            'Ordered Spots': '{:,.0f}',
            'Spots Ran': '{:,.0f}'
        }))

    else:  # Dataset 4
        st.subheader("CRM Analysis Data")
        create_crm_visualizations(filtered_df)
        display_conversation_types()

if __name__ == "__main__":
    main()

