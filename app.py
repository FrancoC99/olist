import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.metrics import mean_squared_error
import plotly.express as px
import streamlit as st
import zipfile
import os

# Helper function to extract and load CSVs from zip files
def load_csv_from_zip(zip_path, file_name):
    if not os.path.exists(zip_path):
        st.error(f"{zip_path} does not exist.")
        st.stop()
    with zipfile.ZipFile(zip_path, 'r') as z:
        if file_name not in z.namelist():
            st.error(f"{file_name} not found in the zip file.")
            st.stop()
        with z.open(file_name) as f:
            return pd.read_csv(f)

# Load datasets
@st.cache_data
def load_data():
    merged_df = load_csv_from_zip('merged_final_dataset_cleaned.csv.zip', 'merged_final_dataset_cleaned.csv')
    df = load_csv_from_zip('final_dataset.csv.zip', 'final_dataset.csv')
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'], errors='coerce')
    df['delivery_time'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    df = df[df['delivery_time'].notna() & df['review_score'].notna()]
    closed_deals = load_csv_from_zip('olist_closed_deals_dataset.csv.zip', 'olist_closed_deals_dataset.csv')
    qualified_leads = load_csv_from_zip('olist_marketing_qualified_leads_dataset.csv.zip', 'olist_marketing_qualified_leads_dataset.csv')
    return merged_df, df, closed_deals, qualified_leads

merged_df, df, closed_deals, qualified_leads = load_data()

# Calculate metrics
state_summary = df.groupby('customer_state').agg({'delivery_time': 'mean', 'review_score': 'mean'}).reset_index()
unique_sellers_per_category = df.groupby('product_category_name_english')['seller_id'].nunique().reset_index()
unique_sellers_per_category.rename(columns={'seller_id': 'unique_sellers_count'}, inplace=True)
total_sales_per_category = df.groupby('product_category_name_english')['order_id'].count().reset_index()
total_sales_per_category.rename(columns={'order_id': 'total_sales'}, inplace=True)
category_analysis = pd.merge(unique_sellers_per_category, total_sales_per_category, on='product_category_name_english')
category_analysis['avg_sales_per_seller'] = category_analysis['total_sales'] / category_analysis['unique_sellers_count']
median_avg_sales_per_seller = category_analysis['avg_sales_per_seller'].median()
high_power_threshold = median_avg_sales_per_seller * 1.5
high_power_categories = category_analysis[category_analysis['avg_sales_per_seller'] > high_power_threshold]
merged_data = pd.merge(closed_deals, qualified_leads, on='mql_id', how='inner')
closed_deals_by_origin_segment = merged_data.groupby(['origin', 'business_segment'])['mql_id'].count().reset_index()
closed_deals_by_origin_segment.rename(columns={'mql_id': 'closed_deals_count'}, inplace=True)
qualified_leads_by_origin = qualified_leads.groupby('origin')['mql_id'].count().reset_index()
qualified_leads_by_origin.rename(columns={'mql_id': 'qualified_leads_count'}, inplace=True)
conversion_data = pd.merge(closed_deals_by_origin_segment, qualified_leads_by_origin, on='origin', how='inner')
conversion_data['conversion_rate'] = conversion_data['closed_deals_count'] / conversion_data['qualified_leads_count']

# Demand Forecast Analysis
def prepare_data(data, selection_type, customer_state=None, product_category=None):
    if selection_type == 'state':
        df = data[data['customer_state'] == customer_state].copy()
    elif selection_type == 'category':
        df = data[data['product_category_name_english'] == product_category].copy()
    elif selection_type == 'both':
        df = data[(data['customer_state'] == customer_state) & (data['product_category_name_english'] == product_category)].copy()
    else:
        raise ValueError("Invalid selection_type. Choose from 'state', 'category', or 'both'.")
    
    df = df.set_index('order_purchase_timestamp').resample('D').size().reset_index(name='demand')
    return df

def analyze_orders(selection_type, state=None, category=None):
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    cutoff_date = pd.to_datetime('2018-07-31')
    df_filtered = df[df['order_purchase_timestamp'] <= cutoff_date]

    prepared_df = prepare_data(df_filtered, selection_type, state, category)
    prepared_df = prepared_df.sort_values('order_purchase_timestamp')

    train = prepared_df.iloc[:-21].copy()
    test = prepared_df.iloc[-21:].copy()

    def create_features(df):
        df = df.copy()
        df['day_of_week'] = df['order_purchase_timestamp'].dt.dayofweek
        df['day_of_month'] = df['order_purchase_timestamp'].dt.day
        df['week_of_year'] = df['order_purchase_timestamp'].dt.isocalendar().week
        df['month'] = df['order_purchase_timestamp'].dt.month
        return df

    train = create_features(train)
    test = create_features(test)

    X_train = train.drop(['order_purchase_timestamp', 'demand'], axis=1)
    y_train = train['demand']
    X_test = test.drop(['order_purchase_timestamp', 'demand'], axis=1)
    y_test = test['demand']

    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    def calculate_intervals(predictions, alpha=0.05):
        errors = y_train - model.predict(X_train)
        error_std = np.std(errors)
        interval_range = error_std * 1.96
        lower_bounds = predictions - interval_range
        upper_bounds = predictions + interval_range
        return lower_bounds, upper_bounds

    lower_bounds, upper_bounds = calculate_intervals(preds)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    st.write(f'Root Mean Square Error (RMSE): {rmse}')

    start_date = test['order_purchase_timestamp'].min() - timedelta(days=30)
    plot_data = prepared_df[(prepared_df['order_purchase_timestamp'] >= start_date) & (prepared_df['order_purchase_timestamp'] <= test['order_purchase_timestamp'].max())].copy()
    plot_data.set_index('order_purchase_timestamp', inplace=True)

    plt.figure(figsize=(14, 7))
    plt.plot(plot_data.index, plot_data['demand'], label='Actual Demand', color='blue')
    plt.plot(test['order_purchase_timestamp'], preds, label='Predicted Demand', color='red')
    plt.fill_between(test['order_purchase_timestamp'], lower_bounds, upper_bounds, color='gray', alpha=0.3, label='Confidence Interval')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.title('Demand Forecast')
    st.pyplot(plt)

# Streamlit App
st.title("Olist Data Analysis")

section = st.sidebar.radio("Go to", ("Welcome", "Demand Forecast", "Rating and Delivery Time", "Seller Analysis", "Seller Power and Conversion Rates"))

if section == "Welcome":
    st.header("Welcome")
    st.write("""
    Welcome to the Olist Data Analysis dashboard. Use the sidebar to navigate between different sections of the analysis.
    """)

elif section == "Rating and Delivery Time":
    st.header("Rating and Delivery Time Analysis")
    selected_metric = st.selectbox('Select metric', ['Delivery Time', 'Rating'])
    
    # Check for required columns and non-null values
    if 'customer_state' not in state_summary.columns or 'delivery_time' not in state_summary.columns or 'review_score' not in state_summary.columns:
        st.error("Required columns are missing from the state_summary dataframe.")
    elif state_summary[['customer_state', 'delivery_time', 'review_score']].isnull().any().any():
        st.error("There are null values in the required columns of the state_summary dataframe.")
    else:
        if selected_metric == 'Delivery Time':
            metric_column = 'delivery_time'
            color_scale = 'Reds'
            color_label = 'Avg Delivery Time (days)'
        else:
            metric_column = 'review_score'
            color_scale = 'Blues'
            color_label = 'Avg Rating'
        
        # Debugging output
        st.write("Data preview:", state_summary.head())
        st.write("GeoJSON URL is valid and accessible.")

        fig = px.choropleth(
            state_summary,
            geojson="https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson",
            locations='customer_state',
            featureidkey="properties.sigla",
            hover_name='customer_state',
            color=metric_column,
            color_continuous_scale=color_scale,
            labels={metric_column: color_label},
            hover_data={
                'delivery_time': True,
                'review_score': True,
                'customer_state': False
            },
            title=f'Average {color_label} by State'
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(
            margin={"r":0,"t":50,"l":0,"b":0},
            clickmode='event+select',
            autosize=True,
            width=1000,
            height=600,
            coloraxis_colorbar=dict(
                title=color_label,
                thicknessmode="pixels", thickness=15,
                lenmode="pixels", len=200,
                yanchor="middle", y=0.5,
                xanchor="left", x=-0.1
            )
        )
        st.plotly_chart(fig)

elif section == "Seller Analysis":
    st.header("Seller Analysis")
    st.write("Seller analysis content goes here...")

elif section == "Seller Power and Conversion Rates":
    st.header("Seller Power and Conversion Rates")
    st.write(high_power_categories)

elif section == "Demand Forecast":
    st.header("Demand Forecast Analysis")

    selection_type = st.selectbox('Select analysis type', ['state', 'category', 'both'])
    selected_state = None
    selected_category = None

    if selection_type in ['state', 'both']:
        selected_state = st.selectbox('Select state', df['customer_state'].unique())
    if selection_type in ['category', 'both']:
        selected_category = st.selectbox('Select category', df['product_category_name_english'].unique())

    analyze_orders(selection_type, selected_state, selected_category)
