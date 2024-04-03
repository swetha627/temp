#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from gluonts.mx import DeepAREstimator
from gluonts.dataset.common import ListDataset
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
import plotly.graph_objects as go
import os
import argparse
import warnings

warnings.filterwarnings('ignore')



class DeepAR:
    def __init__(self, datafile):
        _, file_ext = os.path.splitext(datafile)
        if file_ext == ".csv":
            self.data_df = pd.read_csv(datafile)
        elif file_ext == ".xslx" or file_ext == ".xls" or file_ext == ".xlsx" or file_ext == ".xlsm" or file_ext == ".xlsb" or file_ext == ".odf" or file_ext == ".ods" or file_ext == ".odt":
            self.data_df = pd.read_excel(datafile)
        elif file_ext == ".json":
            self.data_df = pd.read_json(datafile)
        elif file_ext == ".h5" or file_ext == ".hdf":
            self.data_df = pd.read_hdf(datafile)
        elif file_ext == ".html":
            self.data_df = pd.read_html(datafile)
        else:
            print("File Extension not supported. Please use dataset with file ext. - .csv, .xls, .xlsx, .xlsm, .xlsb, .odf, .ods, .odt, .json, .html, .h5, .hdf")
    
    
    def aggregateData(self, order_date_col, forecasting_col):
        product_daily_sales = self.data_df.groupby(order_date_col)[forecasting_col].sum().reset_index()
        
        print("Aggregated Data : \n", product_daily_sales.head())
        print("------------------------------------------------------------------------")
        
        return product_daily_sales
    
    
    def trainingAndInference(self, product_daily_sales, order_date_col, forecasting_col):
        training_split_point = product_daily_sales[order_date_col].iloc[-30]  
        training_data_list = [{
            "start": product_daily_sales[order_date_col].iloc[0],
            "target": product_daily_sales[product_daily_sales[order_date_col] < training_split_point][forecasting_col].values
        }]

        # Create testing data (last 30 days)
        testing_data_list = [{
            "start": training_split_point,
            "target": product_daily_sales[product_daily_sales[order_date_col] >= training_split_point][forecasting_col].values
        }]

        # Convert to GluonTS ListDatasets
        training_data = ListDataset(training_data_list, freq="D")
        testing_data = ListDataset(testing_data_list, freq="D")

        # Define and train the model
        model = DeepAREstimator(
            prediction_length=30,
            num_layers=5,
            freq="D",
            trainer=Trainer(epochs=60)  
        )

        predictor = model.train(training_data=training_data)
        forecasts = list(predictor.predict(testing_data))

        return forecasts, testing_data
    
    
    def kpi_calc(self, forecasts, testing_data):
        tss = list(testing_data)
        rmses = []
        maes = []
        mapes = []

        for forecast, ts in zip(forecasts, tss):
            forecast_values = forecast.mean
            actual_values = ts['target'][-len(forecast_values):]

            # RMSE
            mse = np.mean((forecast_values - actual_values) ** 2)
            rmse = np.sqrt(mse)
            rmses.append(rmse)

            # MAE
            mae = np.mean(np.abs(forecast_values - actual_values))
            maes.append(mae)

            # MAPE
            mape = np.mean(np.abs((forecast_values - actual_values) / (actual_values + np.finfo(float).eps)))
            mapes.append(mape)

        # Calculate the average metrics
        average_rmse = np.mean(rmses)
        average_mae = np.mean(maes)
        average_mape = np.mean(mapes) * 100  

        # Print out the average error metrics
        print("------------------------------------------------------------------------")
        print(f'Average RMSE: {average_rmse}')
        print(f'Average MAE: {average_mae}')
        print(f'Average MAPE: {average_mape}%')
        
        return tss
        
    def visualization(self, forecasts, tss):
        forecast_means = [f.mean for f in forecasts]
        forecast_dates = pd.date_range(start='2015-12-01', end='2015-12-30', freq='D')
        actual_dates = pd.date_range(start='2015-12-01', end='2015-12-30', freq='D')
        # Create the Actual vs Forecasted Values plot
        actual_trace = go.Scatter(x=actual_dates, y=tss[0]['target'], mode='lines', name='Actual')
        forecast_trace = go.Scatter(x=forecast_dates, y=forecast_means[0], mode='lines', name='Forecast')

        fig1 = go.Figure()
        fig1.add_trace(actual_trace)
        fig1.add_trace(forecast_trace)
        fig1.update_layout(title='Actual vs Forecasted Values',
                          xaxis_title='Date',
                          yaxis_title='Quantity',
                          xaxis=dict(tickformat='%Y-%m-%d', dtick='D'))
        fig1.show()
        
        first_forecast = forecasts[0]
        lower_bound = first_forecast.quantile(0.25)  # 25th percentile
        upper_bound = first_forecast.quantile(0.75)  # 75th percentile

        # Create the Forecast with Percentile Intervals plot
        mean_forecast_trace = go.Scatter(x=forecast_dates, y=first_forecast.mean, mode='lines', name='Mean Forecast', line=dict(color='green'))
        percentile_25_trace_lower = go.Scatter(x=forecast_dates, y=lower_bound, mode='lines', name='25th Percentile', line=dict(color='blue'))
        percentile_75_trace_upper = go.Scatter(x=forecast_dates, y=upper_bound, mode='lines', fill='tonexty', fillcolor='rgba(255, 127, 14, 0.5)', name='75th Percentile', line=dict(color='orange'))

        fig2 = go.Figure()
        fig2.add_trace(mean_forecast_trace)
        fig2.add_trace(percentile_25_trace_lower)
        fig2.add_trace(percentile_75_trace_upper)
        fig2.update_layout(title='Forecast with Percentile Intervals',
                          xaxis_title='Date',
                          yaxis_title='Quantity',
                          xaxis=dict(tickformat='%Y-%m-%d', dtick='D'))
        fig2.show()


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='The path of the data to be processed.')
    parser.add_argument('--date_col', type=str, help='The date and time column of the data.')
    parser.add_argument('--forecasting_col', type=str, help='The forecasting column of the data, eg. quantity.')
    
    args = parser.parse_args()
    
    deepar_obj = DeepAR(args.data)
    
    product_daily_sales = deepar_obj.aggregateData(args.date_col, args.forecasting_col)

    forecasts, testing_data = deepar_obj.trainingAndInference(product_daily_sales, args.date_col, args.forecasting_col)
    
    tss = deepar_obj.kpi_calc(forecasts, testing_data)
    
    deepar_obj.visualization(forecasts, tss)
    

