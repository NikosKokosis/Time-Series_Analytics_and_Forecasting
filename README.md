- Time-Series_Analytics_and_Forecasting/
  - Daily_Data/
    - Holt/
      - Holt's_linear_trend_method.ipynb
    - RNNs/
      - LSTMs_Models.ipynb
      - Decomp_EXP_Simple_LSTM.ipynb
      - RNNArchi_Vs_MLPArchi.ipynb
      - RNN_GRU_LSTM_Multivariate.ipynb
    - SES/
      - Simple_Exponential_Smoothing.ipynb
  - Quarterly_Data/
    - Holt's_Winter/
      - Holt_Winter.ipynb


# Time Series Forecasting Analysis in M4 Competition

## Project Overview

This project delves into time series data analysis and prediction using the M4 competition dataset from Kaggle. The dataset encompasses various time intervals, such as hourly, daily, weekly, monthly, quarterly, and yearly. We aim to explore and evaluate different forecasting techniques, from traditional statistical methods to modern machine learning models, on this dataset.

In our analysis, we focus on two main categories of data: daily data and quarterly data. Each category presents unique challenges and requires specific forecasting approaches.

## Daily Data

In this section, we analyze daily time series data using various methods:

### [Simple Exponential Smoothing (SES)](Daily_Data/SES/Simple_Exponential_Smoothing.ipynb)
SES is a straightforward statistical method for predicting future values by weighted averaging of past data points. While useful for time series data with continuous levels and no significant trends or seasonality, SES may not perform well with complex patterns.

### [Holt's Linear Trend Method](Daily_Data/Holt/Holt's_linear_trend_method.ipynb)
Holt's method, also known as Double Exponential Smoothing, extends SES to capture trends in time series data. It is effective for data with trends but no seasonality.

### [LSTM Models with Different Input Sequence Lengths](Daily_Data/RNNs/LSTMs_Models.ipynb)
We experiment with LSTM models using different input sequence lengths to determine the optimal sequence length for prediction. Our findings shed light on the adaptability and performance of LSTM models for daily data.

- Model 1 (with the shorter input sequence) exhibits a more reliable generalization ability, although with limited pattern detection and considerable deviation from the actual values.
- Model 2 (with the longer input sequence) shows sensitivity and tendencies towards overfitting, resulting in entirely inaccurate predictions.

These findings provide valuable insights into the advantages and disadvantages of the models, paving the way for further adjustments and improvements to achieve enhanced predictive accuracy.

### [Adding Trend to SimpleRNN and LSTM Model](Daily_Data/RNNs/Decomp_EXP_Simple_LSTM.ipynb)
To enhance the LSTM and SimpleRNN models, we introduced trend values from the dataset as additional input features. This modification aims to improve pattern recognition and prediction accuracy.

- Model 1 - SimpleRNN<br>
The SimpleRNN model exhibited fluctuations and variations in validation loss during training, indicating overfitting. Its predictions deviated significantly from actual values, raising reliability concerns.

- Model 2 - LSTM<br>
The LSTM model showed close convergence between training and validation loss, avoiding overfitting. Its predictions were more accurate and closely approximated actual values.

### [Forecasting Returns: SimpleRNN vs Multi-Layer Perceptron (MLP)](Daily_Data/RNNs/RNNArchi_Vs_MLPArchi.ipynb)
We compare Simple Recurrent Neural Networks (RNN) and Multi-Layer Perceptrons (MLP) in forecasting percentage changes. Our analysis reveals the strengths and limitations of each architecture.It is evident that the recursive architecture of SimpleRNN has managed to approximate the actual sequence of percentage changes much better than the simple architecture of the MLP neural network, which predicts almost a straight line.

We can draw the following conclusions:

- The architecture of recurrent networks has been more helpful in approximating the patterns of real values, as the simple MLP architecture predicts an almost linear increasing trend.
- During the initial time steps, both models seem to relatively approach the values, with better approximation achieved by SimpleRNN.
- In the later values, while both models diverge, SimpleRNN appears to follow the pattern more closely.

### [Multivariate Forecasting: SimpleRNN vs LSTM vs GRU](Daily_Data/RNNs/RNN_GRU_LSTM_Multivariate.ipynb)
We explore the performance of SimpleRNN, LSTM, and GRU models for multivariate forecasting, considering the interaction between different input features. This analysis provides insights into the suitability of each model for handling multiple input variables.
- All three models (SimpleRNN, LSTM, and GRU) tend to follow a similar pattern in predicting future values, closely resembling the actual pattern.

- The primary distinction among these models is the magnitude of their predictions, with the GRU providing the closest approximation, supported by a lower RMSE.

- Efficiency in predicting the first time step is a common feature among all three models, showcasing their effectiveness in forecasting a near-future moment.

- However, as the forecasting horizon extends, there is significant uncertainty about prediction accuracy, with subsequent predictions often deviating from actual values.

## Quarterly Data

In the context of quarterly time series data, we explore advanced exponential smoothing techniques:

### [Holt-Winter Method](Quarterly_Data/Holt's_Winter/Holt_Winter.ipynb)
The Holt-Winter's Multiplicative Seasonality With Trend (Triple Exponential Smoothing) is applied to quarterly data with clear seasonality. This method accounts for seasonal patterns, trends, and observed values.

## Final Conclusions

Our research on time series data analysis yields significant conclusions:

### Statistical Methods vs. Machine Learning

Traditional statistical methods like SES and Holt's method may not capture complex patterns effectively. Machine learning models offer flexibility and adaptability, making them suitable for non-stationary data.

### Input to the Model Matters

The choice of input data representation significantly impacts model performance. Experimenting with different input types highlights the importance of data preprocessing.

### Multi-Step Forecasting is Challenging

Multi-step forecasting presents challenges, as models tend to produce less accurate predictions as the time horizon extends.

### Evaluation Metrics and Predictive Patterns

Evaluation metrics, especially RMSE, indicate that the GRU model provides higher predictive accuracy. GRU also aligns predictions more closely with real data trends.

### Data Specificity Matters

Tailoring model selection and preprocessing to the unique characteristics of each dataset is crucial for accurate predictions.

In summary, our research underscores the dynamic nature of time series data analysis. While traditional statistical methods have their place, machine learning models offer enhanced adaptability and accuracy, particularly for complex time series data. Effective data preprocessing and model selection are key to unlocking valuable insights and predictions.