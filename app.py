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

    elif option == "Rating and Delivery Time":
    st.header("Rating and Delivery Time Analysis", anchor="rating-and-delivery-time")
    selected_metric = st.selectbox('Select metric', ['Delivery Time', 'Rating'])

    # Ensure columns exist and are numeric (using renamed column name)
    for col in ['delivery_time', 'review_score']:
        if col not in state_summary.columns or state_summary[col].dtype != 'float64':
            st.error(f"Data issue: Column '{col}' missing or not numeric in state_summary.")
            st.stop()

    color_scale = 'Reds' if selected_metric == 'Delivery Time' else 'Blues'
    color_label = f'Avg {selected_metric} (days)' if selected_metric == 'Delivery Time' else 'Avg Rating'

    # Explicitly rename column to match what Plotly expects
    state_summary = state_summary.rename(columns={
        'customer_state': 'sigla', 
        selected_metric.lower(): color_label  # Make this the exact label used in the plot
    })

    st.write("Data preview (after renaming):", state_summary[['sigla', color_label]].head())

    # Create the GeoJSON URL
    geojson_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"

    # Create the Plotly chart (use renamed columns)
    fig = px.choropleth(
        state_summary,
        geojson=geojson_url,
        locations='sigla',
        featureidkey="properties.sigla",
        color=color_label,
        color_continuous_scale=color_scale,
        labels={color_label: color_label},
        hover_data={'delivery_time': True, 'review_score': True, 'sigla': False},
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

            def get_top_n_unique(data, column, n=5):
                return data.drop_duplicates(subset=['seller_id']).nlargest(n, column)[['seller_id', column]]

            top_sellers_revenue = get_top_n_unique(filtered_data, 'revenue_final')
            top_sellers_delivery_time = get_top_n_unique(filtered_data, 'delivery_time_summary')
            top_sellers_rating = get_top_n_unique(filtered_data, 'avg_rating')
            filtered_data['overall_score'] = (
                (filtered_data['revenue_final'].rank(ascending=False) +
                filtered_data['delivery_time_summary'].rank(ascending=True) +
                filtered_data['avg_rating'].rank(ascending=False)) / 3
            )
            top_sellers_overall = get_top_n_unique(filtered_data, 'overall_score')

            st.subheader("Top 5 by Revenue")
            st.write(top_sellers_revenue)
            st.subheader("Top 5 by Delivery Time")
            st.write(top_sellers_delivery_time)
            st.subheader("Top 5 by Rating")
            st.write(top_sellers_rating)
            st.subheader("Top 5 Overall")
            st.write(top_sellers_overall)

    elif section == "seller_power_and_conversion_rates":
        st.header("Seller Power and Conversion Rates", anchor="seller-power-and-conversion-rates")
        num_top_categories = st.selectbox('Select number of top categories', [5, 10, 15, 20], index=1)
        selected_segment = st.selectbox('Select a business segment', conversion_data['business_segment'].unique())
        if st.button('Analyze'):
            top_categories = category_analysis.sort_values(by='avg_sales_per_seller', ascending=False).head(num_top_categories)
            fig1 = px.bar(
                top_categories,
                x='product_category_name_english',
                y='avg_sales_per_seller',
                title=f'Top {num_top_categories} Categories with Highest Seller Power',
                labels={'avg_sales_per_seller': 'Average Sales per Seller'},
                color='avg_sales_per_seller',
                color_continuous_scale='Viridis'
            )
            fig1.add_hline(y=high_power_threshold, line_dash="dash", line_color="red", annotation_text="High Power Threshold")
            st.plotly_chart(fig1)

            if selected_segment:
                filtered_data = conversion_data[conversion_data['business_segment'] == selected_segment]
                fig2 = px.bar(
                    filtered_data,
                    x='origin',
                    y='conversion_rate',
                    title=f'Conversion Rates for Business Segment: {selected_segment}',
                    labels={'conversion_rate': 'Conversion Rate'},
                    color='conversion_rate',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig2)

# Mostrar la sección seleccionada
show_section(default_section)
