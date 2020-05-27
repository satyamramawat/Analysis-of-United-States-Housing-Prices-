# Analysis of United States Housing Prices - A DATA MINING PROJECT - CRISP DM

##### The United States is a country that covers 50 states along with North America, Alaska in the northwest. United states also known as the world’s market leader where most of the multi-national companies have its headquarters. More business leads to more population around the globe. In the U.S. many students come from different countries to pursue their higher studies and find a job opportunity, thus leads to housing crisis in the United States. With constrained dataset and information includes, a reasonable and composite data pre-processing, the inventive component designing technique is analyzed in this paper. The major focus on housing prices of the U.S., by the power of prediction with Regression modeling, considering the amenities.

<center>Regressor model used in the analysis </center>

| Model  | MAE | MSE | RMSE | Accuracy
| ------------- | ------------- |  ------------- |  ------------- | ------------- |
| Random Forest  | 0.37  | 0.37 | 0.61 | 0.64 |
| Decision Tree | 0.37  | 0.43 | 0.65 | 0.57 |
| Gradient Boosting | 0.58 | 0.56 | 0.75 | 0.44 |
| Linear Regression | 0.68 | 0.72 | 0.84 | 0.29 |
| Support Vector Machine | 0.67 | 0.72 | 0.85 | 0.28 |


# Conclusion
##### The objective of this work was to forecast the rent increased per month in the USA and various factors which affects the rent of the house in the USA which has been addressed successfully using machine learning algorithms.
##### The model was evaluated based on the data used for modeling. We have used Supervised Machine Learning Regression algorithms to predict the rise of housing rent in the USA. The Random Forest regressor outperformed all the individual model performance with better accuracy and fewer errors.


To learn more, Kindly go through <b>"Assignment_Report.pdf"</b>.

###### imports used
* import pandas as pd
* import numpy as np
* import seaborn as sns
* import matplotlib.pyplot as plt
* import plotly.express as px
* from sklearn.model_selection import  train_test_split,cross_val_score
* from sklearn.preprocessing import StandardScaler
* from sklearn.linear_model import LinearRegression
* from sklearn import metrics
* from sklearn.metrics import r2_score
* from sklearn.ensemble import RandomForestRegressor
* from sklearn.ensemble import GradientBoostingRegressor
* from sklearn.model_selection import ShuffleSplit
* from sklearn.svm import SVR
* from sklearn.tree import DecisionTreeRegressor
