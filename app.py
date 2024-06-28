import os
import streamlit as st
import pandas as pd




# Load datasets
train = pd.read_csv("/content/drive/MyDrive/SEM_2_PROJECT/datasets/train.csv")
test = pd.read_csv("/content/drive/MyDrive/SEM_2_PROJECT/datasets/test.csv")
stores = pd.read_csv("/content/drive/MyDrive/SEM_2_PROJECT/datasets/stores.csv")
holiday_events = pd.read_csv("/content/drive/MyDrive/SEM_2_PROJECT/datasets/holidays_events.csv")
oil = pd.read_csv("/content/drive/MyDrive/SEM_2_PROJECT/datasets/oil.csv")
transaction = pd.read_csv("/content/drive/MyDrive/SEM_2_PROJECT/datasets/transactions.csv")




# Main function for Option 1
def option_1():
    st.title('Sales Visualize and Analysis')

    # Display options
    display_options = ['Train Data', 'Test Data', 'Stores Data', 'Holiday Events Data', 'Oil Data', 'Transaction Data']
    selected_option = st.selectbox('Select dataset to display:', display_options)

    # Display selected dataset
    if selected_option == 'Train Data':
        st.write(train.head(20))  # Display first 20 rows of train data
    elif selected_option == 'Test Data':
        st.write(test.head(20))  # Display first 20 rows of test data
    elif selected_option == 'Stores Data':
        st.write(stores.head(20))  # Display first 20 rows of stores data
    elif selected_option == 'Holiday Events Data':
        st.write(holiday_events.head(20))  # Display first 20 rows of holiday events data
    elif selected_option == 'Oil Data':
        st.write(oil.head(20))  # Display first 20 rows of oil data
    elif selected_option == 'Transaction Data':
        st.write(transaction.head(20))  # Display first 20 rows of transaction data








# Load the merged data for PandasAI
merged_dataframe_copy = pd.read_csv("/content/drive/MyDrive/SEM_2_PROJECT/datasets/visualization_Data.csv")




# Setting the PandasAI API key
os.environ["PANDASAI_API_KEY"] = "$2a$10$W6UbS3Te6aW/ORn2ZngEnO4j3WdtsJCm7mWDeKL4AYEy.IrQzLYPW"


import os
import pandas as pd
import streamlit as st
from PIL import Image
# Assuming Agent is a class from pandasai, import it.
from pandasai import Agent


# Load data function
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except AttributeError:
        ext = uploaded_file.split(".")[-1]
    if ext in ['csv', 'xls', 'xlsx', 'xlsm', 'xlsb']:
        if ext.startswith('xls'):
            return pd.read_excel(uploaded_file)
        else:
            return pd.read_csv(uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None


# Main function for Option 2
def option_2():
    st.title('Sales Explorer')
    st.write("Welcome to the Sales Explorer app! This app allows you to interactively explore sales data.")


    # Chat with PandasAI directly
    direct_query = st.text_input("Enter your query directly:", key="direct_query_input")
    if st.button("Ask", key="direct_query_button"):
        st.write("You:", direct_query)
        # Assuming merged_dataframe_copy is predefined or obtained from some source.
        # If not, it should be handled properly to avoid NameError.
        try:
            agent = Agent(merged_dataframe_copy)
            response = agent.chat(direct_query)


            # Check if the response is an image path
            if isinstance(response, str) and response.startswith('/content/exports/charts'):
                # Display the image within the Streamlit app
                image = Image.open(response)
                st.image(image, caption='Visualization')
            else:
                st.write("PandasAI:", response)
        except NameError:
            st.error("Data is not available for querying. Please upload a dataset first.")


    # File upload
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


                # Check if the response is an image path
                if isinstance(response, str) and response.startswith('/content/exports/charts'):
                    # Display the image within the Streamlit app
                    image = Image.open(response)
                    st.image(image, caption='Visualization')
                else:
                    st.write("PandasAI:", response)
        else:
            st.error("Failed to load the data. Please upload a valid file.")








import streamlit as st
import pandas as pd
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
import plotly.express as px
import numpy as np




# Initialize H2O
h2o.init()
data = pd.read_csv("/content/drive/MyDrive/SEM_2_PROJECT/datasets/visualization_Data.csv")
# Assuming 'data' is already defined and loaded
# data = pd.read_csv('your_dataset.csv')




def option_3():
    st.title('Sales Analysis and Future Trends')




    # Convert 'date' column to datetime format
    data['date'] = pd.to_datetime(data['date'])




    # Split data into training (up to 2016) and testing (2017) sets
    train = data[data['date'].dt.year <= 2016]
    test = data[data['date'].dt.year == 2017]




    # Convert the pandas DataFrame to H2OFrame
    train_h2o = h2o.H2OFrame(train)
    test_h2o = h2o.H2OFrame(test)




    # Ask the user what to predict
    prediction_target = st.selectbox('What do you want to predict?', ['Sales', 'Oil Price'])




    if prediction_target == 'Sales':
        dependent_variable = 'sales'
    elif prediction_target == 'Oil Price':
        dependent_variable = 'dcoilwtico'




    # Define the predictors, excluding the dependent variable
    all_predictors = ["store_nbr", "family", "onpromotion", "city", "state", "store_type",]
    predictors = [var for var in all_predictors if var != dependent_variable]




    # User selects the independent variables
    selected_predictors = st.multiselect('Select Independent Variables:', predictors, default=[predictors[0]])




    if len(selected_predictors) < 1:
        st.warning('Please select at least one independent variable.')
        return




    # Create and train the model
    model = H2ORandomForestEstimator(ntrees=50, max_depth=20, seed=42)
    model.train(x=selected_predictors, y=dependent_variable, training_frame=train_h2o)




    # Make predictions for 2017
    predictions = model.predict(test_h2o)




    # Convert predictions to pandas DataFrame
    predictions_df = predictions.as_data_frame()




    # Add 'date' column to predictions DataFrame
    predictions_df['date'] = test['date'].values




    # Extract actual values for 2017
    actual_2017 = test[test['date'].dt.year == 2017][dependent_variable].values




    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({'date': predictions_df['date'], 'Actual': actual_2017, 'Predicted': predictions_df['predict']})




    # Plot using Plotly
    fig = px.line(plot_data, x='date', y=['Actual', 'Predicted'], title=f'Actual vs Predicted {dependent_variable.capitalize()} for 2017')
    st.plotly_chart(fig)




    # Feature importance
    st.subheader('Feature Importance')
    varimp = model.varimp(use_pandas=True)
    st.write(varimp)




    # Plot feature importance
    fig_imp = px.bar(varimp, x='percentage', y='variable', orientation='h', title='Feature Importance')
    st.plotly_chart(fig_imp)




    # Future trends prediction for 2018
    st.subheader(f'Future Trends for 2018 ({dependent_variable.capitalize()})')




    # User input for future trends
    st.write('Enter the values for independent features for the year 2018 (choose "Don\'t choose for future analysis" to skip a feature):')




    # Dropdown options for independent features
    options = {feature: ['Don\'t choose for future analysis'] + sorted(data[feature].unique().tolist()) for feature in selected_predictors}




    selected_values = {}
    for feature in selected_predictors:
        selected_values[feature] = st.selectbox(feature, options[feature])




    # Dynamic oil price range based on selected features
    filtered_data = data.copy()
    for feature, value in selected_values.items():
        if value != "Don't choose for future analysis":
            filtered_data = filtered_data[filtered_data[feature] == value]




    if dependent_variable == 'dcoilwtico':
        dep_var_min = data['dcoilwtico'].min()
        dep_var_max = data['dcoilwtico'].max()
        dep_var_value = st.slider('Oil Price', float(dep_var_min), float(dep_var_max))
    else:
        dep_var_value = None




    if st.button('Predict Future Trends'):
        # Create a DataFrame for future dates in 2018
        future_dates = pd.date_range(start='2018-01-01', end='2018-12-31', freq='MS')
        future_data = pd.DataFrame({'date': future_dates})




        # Add selected features
        for feature, value in selected_values.items():
            if value != "Don't choose for future analysis":
                future_data[feature] = value




        if dep_var_value is not None:
            future_data['dcoilwtico'] = dep_var_value




        # Convert the future data to H2OFrame
        future_h2o = h2o.H2OFrame(future_data)




        # Make predictions for 2018
        future_predictions = model.predict(future_h2o)




        # Convert predictions to pandas DataFrame
        future_predictions_df = future_predictions.as_data_frame()




        # Add 'date' column to predictions DataFrame
        future_predictions_df['date'] = future_data['date'].values




        # Create a DataFrame for plotting
        future_plot_data = pd.DataFrame({'date': future_predictions_df['date'], 'Predicted': future_predictions_df['predict']})




        # Introduce some randomness to the predictions to avoid identical values for all months
        future_plot_data['Predicted'] += np.random.uniform(-0.05, 0.05, size=future_plot_data.shape[0]) * future_plot_data['Predicted']




        # Plot using Plotly
        future_fig = px.line(future_plot_data, x='date', y='Predicted', title=f'Predicted {dependent_variable.capitalize()} for 2018')
        st.plotly_chart(future_fig)




        # Display predictions in tabular format
        st.subheader('Predicted Values for 2018')
        st.write(future_plot_data)




        # Plot predicted values vs oil price if relevant
        if 'dcoilwtico' in future_plot_data.columns:
            sales_oil_fig = px.scatter(future_plot_data, x='date', y='Predicted', color=future_data['dcoilwtico'], title=f'Predicted {dependent_variable.capitalize()} vs Oil Price')
            st.plotly_chart(sales_oil_fig)




# Display options
option = st.selectbox('Select Option:', ['Option 1: Sales Data Overview', 'Option 2: Sales Data Exploration', 'Option 3: Sales Analysis '])




# Execute selected option
if option == 'Option 1: Sales Data Overview':
    option_1()
elif option == 'Option 2: Sales Data Exploration':
    option_2()
elif option == 'Option 3: Sales Analysis ':
    option_3()
