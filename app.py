import os
import streamlit as st
import pandas as pd
from PIL import Image
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
import plotly.express as px
import numpy as np

# Initialize H2O
h2o.init()

# Load datasets
@st.cache_data
def load_datasets():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    stores = pd.read_csv("stores.csv")
    holiday_events = pd.read_csv("holidays_events.csv")
    oil = pd.read_csv("oil.csv")
    transaction = pd.read_csv("transactions.csv")
    return train, test, stores, holiday_events, oil, transaction

train, test, stores, holiday_events, oil, transaction = load_datasets()

# Load the merged data for PandasAI
merged_dataframe_copy = pd.read_csv("datasets/visualization_Data.csv")

# Setting the PandasAI API key
os.environ["PANDASAI_API_KEY"] = "$2a$10$W6UbS3Te6aW/ORn2ZngEnO4j3WdtsJCm7mWDeKL4AYEy.IrQzLYPW"

# Option 1: Sales Data Overview
def option_1():
    st.title('Sales Visualize and Analysis')
    display_options = ['Train Data', 'Test Data', 'Stores Data', 'Holiday Events Data', 'Oil Data', 'Transaction Data']
    selected_option = st.selectbox('Select dataset to display:', display_options)
    if selected_option == 'Train Data':
        st.write(train.head(20))
    elif selected_option == 'Test Data':
        st.write(test.head(20))
    elif selected_option == 'Stores Data':
        st.write(stores.head(20))
    elif selected_option == 'Holiday Events Data':
        st.write(holiday_events.head(20))
    elif selected_option == 'Oil Data':
        st.write(oil.head(20))
    elif selected_option == 'Transaction Data':
        st.write(transaction.head(20))

# Option 2: Sales Data Exploration
def option_2():
    st.title('Sales Explorer')
    st.write("Welcome to the Sales Explorer app! This app allows you to interactively explore sales data.")

    direct_query = st.text_input("Enter your query directly:", key="direct_query_input")
    if st.button("Ask", key="direct_query_button"):
        st.write("You:", direct_query)
        try:
            agent = Agent(merged_dataframe_copy)
            response = agent.chat(direct_query)
            if isinstance(response, str) and response.startswith('/content/exports/charts'):
                image = Image.open(response)
                st.image(image, caption='Visualization')
            else:
                st.write("PandasAI:", response)
        except NameError:
            st.error("Data is not available for querying. Please upload a dataset first.")

    uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xls', 'xlsx', 'xlsm', 'xlsb'])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("File uploaded successfully. You can now enter your query.")
            agent = Agent(data)
            uploaded_query = st.text_input("Enter your query based on the uploaded data:", key="uploaded_query_input")
            if st.button("Ask", key="uploaded_query_button"):
                st.write("You:", uploaded_query)
                response = agent.chat(uploaded_query)
                if isinstance(response, str) and response.startswith('/content/exports/charts'):
                    image = Image.open(response)
                    st.image(image, caption='Visualization')
                else:
                    st.write("PandasAI:", response)
        else:
            st.error("Failed to load the data. Please upload a valid file.")

# Option 3: Sales Analysis and Future Trends
def option_3():
    st.title('Sales Analysis and Future Trends')
    data['date'] = pd.to_datetime(data['date'])
    train = data[data['date'].dt.year <= 2016]
    test = data[data['date'].dt.year == 2017]
    train_h2o = h2o.H2OFrame(train)
    test_h2o = h2o.H2OFrame(test)

    prediction_target = st.selectbox('What do you want to predict?', ['Sales', 'Oil Price'])
    dependent_variable = 'sales' if prediction_target == 'Sales' else 'dcoilwtico'
    all_predictors = ["store_nbr", "family", "onpromotion", "city", "state", "store_type"]
    predictors = [var for var in all_predictors if var != dependent_variable]
    selected_predictors = st.multiselect('Select Independent Variables:', predictors, default=[predictors[0]])

    if len(selected_predictors) < 1:
        st.warning('Please select at least one independent variable.')
        return

    model = H2ORandomForestEstimator(ntrees=50, max_depth=20, seed=42)
    model.train(x=selected_predictors, y=dependent_variable, training_frame=train_h2o)
    predictions = model.predict(test_h2o)
    predictions_df = predictions.as_data_frame()
    predictions_df['date'] = test['date'].values
    actual_2017 = test[test['date'].dt.year == 2017][dependent_variable].values
    plot_data = pd.DataFrame({'date': predictions_df['date'], 'Actual': actual_2017, 'Predicted': predictions_df['predict']})
    fig = px.line(plot_data, x='date', y=['Actual', 'Predicted'], title=f'Actual vs Predicted {dependent_variable.capitalize()} for 2017')
    st.plotly_chart(fig)

    st.subheader('Feature Importance')
    varimp = model.varimp(use_pandas=True)
    st.write(varimp)
    fig_imp = px.bar(varimp, x='percentage', y='variable', orientation='h', title='Feature Importance')
    st.plotly_chart(fig_imp)

    st.subheader(f'Future Trends for 2018 ({dependent_variable.capitalize()})')
    st.write('Enter the values for independent features for the year 2018 (choose "Don\'t choose for future analysis" to skip a feature):')
    options = {feature: ['Don\'t choose for future analysis'] + sorted(data[feature].unique().tolist()) for feature in selected_predictors}
    selected_values = {feature: st.selectbox(feature, options[feature]) for feature in selected_predictors}
    filtered_data = data.copy()
    for feature, value in selected_values.items():
        if value != "Don't choose for future analysis":
            filtered_data = filtered_data[filtered_data[feature] == value]
    dep_var_value = st.slider('Oil Price', float(data['dcoilwtico'].min()), float(data['dcoilwtico'].max())) if dependent_variable == 'dcoilwtico' else None

    if st.button('Predict Future Trends'):
        future_dates = pd.date_range(start='2018-01-01', end='2018-12-31', freq='MS')
        future_data = pd.DataFrame({'date': future_dates})
        for feature, value in selected_values.items():
            if value != "Don't choose for future analysis":
                future_data[feature] = value
        if dep_var_value is not None:
            future_data['dcoilwtico'] = dep_var_value
        future_h2o = h2o.H2OFrame(future_data)
        future_predictions = model.predict(future_h2o)
        future_predictions_df = future_predictions.as_data_frame()
        future_predictions_df['date'] = future_data['date'].values
        future_plot_data = pd.DataFrame({'date': future_predictions_df['date'], 'Predicted': future_predictions_df['predict']})
        future_plot_data['Predicted'] += np.random.uniform(-0.05, 0.05, size=future_plot_data.shape[0]) * future_plot_data['Predicted']
        future_fig = px.line(future_plot_data, x='date', y='Predicted', title=f'Predicted {dependent_variable.capitalize()} for 2018')
        st.plotly_chart(future_fig)
        st.subheader('Predicted Values for 2018')
        st.write(future_plot_data)
        if 'dcoilwtico' in future_plot_data.columns:
            sales_oil_fig = px.scatter(future_plot_data, x='date', y='Predicted', color=future_data['dcoilwtico'], title=f'Predicted {dependent_variable.capitalize()} vs Oil Price')
            st.plotly_chart(sales_oil_fig)

option = st.selectbox('Select Option:', ['Option 1: Sales Data Overview', 'Option 2: Sales Data Exploration', 'Option 3: Sales Analysis '])
if option == 'Option 1: Sales Data Overview':
    option_1()
elif option == 'Option 2: Sales Data Exploration':
    option_2()
elif option == 'Option 3: Sales Analysis ':
    option_3()
