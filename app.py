import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import io
import base64
from datetime import timedelta
from sklearn.metrics import mean_squared_error
import plotly.express as px
import streamlit as st
import zipfile
import os

# Estilos CSS para cambiar los colores y la barra de navegación superior
st.markdown("""
    <style>
    .stApp {
        background-color: #E1ECF4;  /* Azul claro para el resto del diseño */
    }
    .top-bar {
        background-color: #2B3A67;  /* Azul oscuro para la barra superior */
        display: flex;
        justify-content: space-around;
        padding: 10px;
    }
    .top-bar a {
        color: white;
        text-decoration: none;
        font-size: 20px;
    }
    .top-bar a:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

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
    plot_data = prepared_df[(prepared_df['order_purchase_timestamp'] >= start_date) & (prepared_df['order_purchase_timestamp'] <= test['order_purchase_timestamp'].max())]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(plot_data['order_purchase_timestamp'], plot_data['demand'], label='Historical')
    ax.plot(test['order_purchase_timestamp'], y_test, label='Test')
    ax.plot(test['order_purchase_timestamp'], preds, label='Forecast')
    ax.fill_between(test['order_purchase_timestamp'], lower_bounds, upper_bounds, color='gray', alpha=0.2, label='95% Prediction Interval')
    ax.legend()
    st.pyplot(fig)

    results = test[['order_purchase_timestamp']].copy()
    results['forecast'] = preds
    results['lower_bound'] = lower_bounds
    results['upper_bound'] = upper_bounds
    results_filtered = results[results['order_purchase_timestamp'] >= start_date]

    st.write(results_filtered)

# Streamlit App
st.title("Olist Consulting Dashboard")

# Obtener la sección seleccionada de los parámetros de la URL
query_params = st.experimental_get_query_params()
default_section = query_params.get("section", ["demand_forecast"])[0]

# Mapear la sección seleccionada a la opción correspondiente del selectbox
section_to_option = {
    "demand_forecast": "Demand Forecast",
    "rating_and_delivery_time": "Rating and Delivery Time",
    "seller_analysis": "Seller Analysis",
    "seller_power_and_conversion_rates": "Seller Power and Conversion Rates"
}
option = section_to_option.get(default_section, "Demand Forecast")

# Barra de navegación superior
st.markdown(f"""
    <div class="top-bar">
        <a href="?section=demand_forecast">Demand Forecast</a>
        <a href="?section=rating_and_delivery_time">Rating and Delivery Time</a>
        <a href="?section=seller_analysis">Seller Analysis</a>
        <a href="?section=seller_power_and_conversion_rates">Seller Power and Conversion Rates</a>
    </div>
    """, unsafe_allow_html=True)

# Pestañas de la aplicación
if option == "Rating and Delivery Time":
    st.header("Rating and Delivery Time Analysis")

    if state_summary.empty:
        st.error("No data available for the analysis.")
    else:
        # Verificar si hay estados en el dataframe
        if state_summary['customer_state'].nunique() == 0:
            st.error("No states found in the data.")
        else:
            # Mostrar el resumen por estado
            st.dataframe(state_summary)

            # Gráfico de coropletas
            fig = px.choropleth(state_summary,
                                locations='customer_state',
                                locationmode='USA-states',
                                color='delivery_time',
                                hover_name='customer_state',
                                hover_data=['review_score'],
                                color_continuous_scale='Blues',
                                labels={'delivery_time': 'Avg Delivery Time'})
            fig.update_layout(title_text='Average Delivery Time by State', geo_scope='usa')
            st.plotly_chart(fig)

elif option == "Demand Forecast":
    st.header("Demand Forecast")
    selection_type = st.selectbox("Choose a selection type", ["state", "category", "both"])
    selected_state = None
    selected_category = None

    if selection_type == "state":
        selected_state = st.selectbox("Select a state", df['customer_state'].unique())
    elif selection_type == "category":
        selected_category = st.selectbox("Select a category", df['product_category_name_english'].unique())
    elif selection_type == "both":
        selected_state = st.selectbox("Select a state", df['customer_state'].unique())
        selected_category = st.selectbox("Select a category", df['product_category_name_english'].unique())

    if st.button("Analyze Orders"):
        if selection_type == "state":
            analyze_orders(selection_type='state', state=selected_state)
        elif selection_type == "category":
            analyze_orders(selection_type='category', category=selected_category)
        elif selection_type == "both":
            analyze_orders(selection_type='both', state=selected_state, category=selected_category)

elif option == "Seller Analysis":
    st.header("Seller Analysis")
    st.write("Unique Sellers per Category")
    st.dataframe(unique_sellers_per_category)
    st.write("Total Sales per Category")
    st.dataframe(total_sales_per_category)

elif option == "Seller Power and Conversion Rates":
    st.header("Seller Power and Conversion Rates")
    st.write("High Power Categories")
    st.dataframe(high_power_categories)
    st.write("Conversion Data")
    st.dataframe(conversion_data)

