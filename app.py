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
# ... (código anterior)

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
# ... (resto del código de análisis de la demanda)

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

# Función para mostrar la sección seleccionada
def show_section(section):
    if section == "demand_forecast":
        st.header("Demand Forecast Analysis", anchor="demand-forecast")
        state = st.selectbox('Select a customer state', df['customer_state'].unique())
        category = st.selectbox('Select a product category', df['product_category_name_english'].unique())
        forecast_option = st.radio('Forecast Option', ['Only by Category', 'Only by State', 'By Both State and Category'], index=1)
        
        # Map forecast_option to selection_type
        if forecast_option == 'Only by Category':
            selection_type = 'category'
        elif forecast_option == 'Only by State':
            selection_type = 'state'
        elif forecast_option == 'By Both State and Category':
            selection_type = 'both'

        if st.button('Go'):
            analyze_orders(selection_type, state, category)

    elif section == "rating_and_delivery_time":
        st.header("Rating and Delivery Time Analysis", anchor="rating-and-delivery-time")
        selected_metric = st.selectbox('Select metric', ['Delivery Time', 'Rating'])
        
        # Check for required columns and non-null values
        if 'customer_state' not in state_summary.columns or 'delivery_time' not in state_summary.columns or 'review_score' not in state_summary.columns:
            st.error("Required columns are missing from the state_summary dataframe.")
        elif state_summary[['customer_state', 'delivery_time', 'review_score']].isnull().any().any():
            st.error("There are null values in the required columns of the state_summary dataframe.")
        else:
            if selected_metric == 'Delivery Time':
                color_scale = 'Reds'
                color_label = 'Avg Delivery Time (days)'
            else:
                color_scale = 'Blues'
                color_label = 'Avg Rating'
            
            # Debugging output
            st.write("Data preview:", state_summary.head())
            st.write("GeoJSON URL is valid and accessible.")

            # Verificación adicional de los datos
            if selected_metric.lower() not in state_summary.columns:
                st.error(f"Column {selected_metric.lower()} does not exist in state_summary.")
            else:
                st.write("Data passed to px.choropleth:", state_summary[['customer_state', selected_metric.lower()]].head())
                
                fig = px.choropleth(
                    state_summary,
                    geojson="https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson",
                    locations='customer_state',
                    featureidkey="properties.sigla",
                    hover_name='customer_state',
                    color=selected_metric.lower(),
                    color_continuous_scale=color_scale,
                    labels={selected_metric.lower(): color_label},
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

    elif section == "seller_analysis":
        st.header("Seller Analysis", anchor="seller-analysis")
        selected_state = st.selectbox('Select a customer state', merged_df['customer_state_summary'].unique())
        selected_category = st.selectbox('Select a product category', merged_df['product_category_name_english_summary'].unique())
        ranking_filter = st.radio('Select ranking filter', ['Top 10 Best Sellers', 'Top 10 Worst Sellers'])
        if st.button('Go'):
            filtered_data = merged_df
            if selected_state:
                filtered_data = filtered_data[filtered_data['customer_state_summary'] == selected_state]
            if selected_category:
                filtered_data = filtered_data[filtered_data['product_category_name_english_summary'] == selected_category]
            if ranking_filter == 'Top 10 Best Sellers':
                filtered_data = filtered_data.nlargest(10, 'revenue_final')
            elif ranking_filter == 'Top 10 Worst Sellers':
                filtered_data = filtered_data.nsmallest(10, 'revenue_final')

            fig = px.scatter(
                filtered_data,
                x='delivery_time_summary',
                y='review_score_summary',
                size='revenue_final',
                color='revenue_final',
                hover_name='seller_id_summary',
                labels={'delivery_time_summary': 'Delivery Time (days)', 'review_score_summary': 'Review Score', 'revenue_final': 'Revenue'},
                title='Seller Analysis'
            )
            st.plotly_chart(fig)

    elif section == "seller_power_and_conversion_rates":
        st.header("Seller Power and Conversion Rates Analysis", anchor="seller-power-and-conversion-rates")
        st.dataframe(high_power_categories)
        st.dataframe(conversion_data)

show_section(default_section)
