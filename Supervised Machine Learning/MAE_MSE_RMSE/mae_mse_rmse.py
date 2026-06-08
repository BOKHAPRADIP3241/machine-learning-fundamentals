"""
MAE - Mean Absolute Error
     step1: Calculate the absolute error for each data point (|y_true - y_pred|)
     step2: Calculate the mean of the absolute errors (sum of absolute errors / number of data points)

MSE - Mean Squared Error
        step1: Calculate the squared error for each data point ((y_true - y_pred)^2)
        step2: Calculate the mean of the squared errors (sum of squared errors / number of data points)
RMSE - Root Mean Squared Error
        step1: Calculate the squared error for each data point ((y_true - y_pred)^2)
        step2: Calculate the mean of the squared errors (sum of squared errors / number of data points)
        step3: Take the square root of the mean squared error
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

#real_scores
real_scores = [90,60,80,100]

#model Guess
predicted_scores = [85,70,70,95]

mae = mean_absolute_error(real_scores, predicted_scores)
mse = mean_squared_error(real_scores, predicted_scores)
rmse = np.sqrt(mse)

print("MAE:", mae)
print("MSE:", mse)      
print("RMSE:", rmse)