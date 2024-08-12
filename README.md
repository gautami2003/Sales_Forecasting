# Sales Analysis and Prediction App

This Streamlit application provides tools for visualizing, exploring, and analyzing sales data. It allows users to interactively examine datasets, make predictions using machine learning models, and analyze future trends. The app is built using Python with the following key libraries: Pandas, Streamlit, H2O, and Plotly.

## Features

### 1. Sales Data Overview
This section allows users to quickly visualize and explore different datasets used in the analysis. Users can choose from various datasets such as Train Data, Test Data, Stores Data, Holiday Events Data, Oil Data, and Transaction Data.

### 2. Sales Data Exploration
In this section, users can interactively explore the sales data by asking queries. The app can respond to direct queries or queries based on uploaded datasets. It also supports visualizations of query results.

### 3. Sales Analysis and Future Trends
Users can analyze the sales data and predict future trends using a Random Forest model. This section allows for the selection of independent variables, visualization of model predictions, and analysis of feature importance. Future trends for the year 2018 can be predicted based on user input.

## Installation

1. **Clone the repository*:
   ```bash
   git clone https://github.com/gautami2003/Sales_Forecasting.git
   cd Sales_Forecasting

   Install the necessary dependencies:

    ```bash
      pip install -r requirements.txt

Run the Streamlit app:

    ```bash
    streamlit run app.py

**Datasets*
The application uses several datasets for analysis:

train.csv: Contains the training data.
test.csv: Contains the test data.
stores.csv: Contains store-related information.
holidays_events.csv: Contains holiday events data.
oil.csv: Contains oil price data.
transactions.csv: Contains transaction data.
These datasets must be placed in the root directory or properly referenced in the code.

**Usage*
Upon launching the app, users will be presented with three main options:

Sales Data Overview: View the first 20 records of the selected dataset.
Sales Data Exploration: Interactively explore and query the sales data, either through direct text input or by uploading a custom dataset.
Sales Analysis and Future Trends: Analyze the sales data, visualize predictions, and explore future trends for 2018.
Model
The app uses an H2O Random Forest model for making predictions. The model is trained on the sales data and can predict sales or oil prices based on selected independent variables. Feature importance is also provided to help users understand the impact of each variable on the predictions.

*API Key*
The app utilizes the PandasAI API for processing queries. Ensure to set the PANDASAI_API_KEY in your environment variables before running the app.

*Contributing*
Feel free to submit issues or pull requests if you would like to contribute to this project.
