#import Libraries
#Pandas is a Python library used for working with data sets. 
#It has functions for analyzing, cleaning, exploring, and manipulating data.
import pandas as pd
#NumPy is a Python library used for working with arrays. 
#It also has functions for working in domain of linear algebra, 
#fourier transform, and matrices. 
import numpy as np
#data visualization and graphical plotting library 
#(histograms, scatter plots, bar charts, etc) for Python 
import matplotlib.pyplot as plt
#Seaborn-It provides data visualizations that are typically more aesthetic and 
#statistically sophisticated.
import seaborn as sns
You can use this module in Scikit-Learn for various datasets, score functions, 
#and performance metrics. 
from sklearn import metrics

#read the Dataset (.csv) file by using read_csv()
house=pd.read_csv(r"C:\Users\Administrator\Desktop\Project\Project Linear Regression\USA_Housing.csv")
house_df=pd.DataFrame(house)
#Data set store into DataFrame, A DataFrame is a data structure that organizes
#data into a 2-dimensional table of rows and columns, much like a spreadsheet.
print(house_df.head())
print(house_df.describe())
print(house_df.columns)
print(house_df.info())

#Address is nonsignificant variable to pridict USA_Housing price.hence we have to drop this variable
house_df.drop(['Address'],axis=1,inplace=True)

###Analyse the data with pairplots
#sns.pairplot(house_df) 

#pairplot distubution is Normalize /showing histogram is bell shape
#Heatmap shows the correlation of dataFrame
sns.heatmap(house_df.corr(),annot=True)

#here diagnol row showing 100% correlation
#remain are no correlation but highest corelation between Price and Avg Area income i.e, 0.64

#define the features, X features are Independent Variables and y feature is dependent variable(target)
X=house_df.drop(['Price'],axis=1)
y=house_df['Price']

#training and spliting dataset
from sklearn.model_selection import train_test_split
#it will create two sets of data, 1-trains set and another test set data
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size= 0.3,random_state=10)
print(print(X_train.shape, X_test.shape, y_train.shape, y_test.shape))


######################### Linear Regression Model #############33
#import the linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#creating the obect to store linear regression function
lm=LinearRegression()
#After creating Variable the object of Linear regression we will fitting Model inside
lm.fit(X_train,y_train)
#Model prediction by using 
y_pred=lm.predict(X_test)             #prediction on test data
print(y_pred)
#the intercept (often labeled the constant) is the expected value of Y when all X=0.
print("Intercept:",lm.intercept_)
print()

#first see coefficient of DF, that mean one unit price increse mean that value equal to coefficient
coeff_df=pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print(coeff_df)

#print(plt.scatter(y_test,y_pred))
#this graph shows line shape so our model predicted very well
sns.distplot((y_test,y_pred),bins=50)
#This graph shows bell shape which is normalized which saya that our model is well predicted

#to get a descriptive statistics summary of a given dataframe
#print(house_df.describe())

#divide the number of correct predictions by the total number of predictions 
#made by the model
#from sklearn.metrics import classification_report
##print(classification_report(y_test_classes,y_pred_classes))

#print("Training Accuracy :", r2_score(y_train, y_pred))
print()
print("Linear Regression Test Accuracy :", r2_score(y_test, y_pred))

###################Lasso Regularization################
#Modifies overfitted or under-fitted models by adding a penalty equivalent to 
#the sum of the absolute values ​​of the coefficients. Lasso regression also performs 
#coefficient minimization, but instead of squaring the magnitudes of the coefficients,
#it takes the actual values ​​of the coefficients.

from sklearn.linear_model import Lasso 
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print("Lasso Model :", (lasso.coef_))
y_pred_train_lasso = lasso.predict(X_train)
y_pred_test_lasso = lasso.predict(X_test)
#R2 Score is a very popular metric used for evaluating the performance of 
#linear regression models.
print("R2_Score Training Accuracy :", r2_score(y_train, y_pred_train_lasso))
print()
print("R2_score Test Accuracy :", r2_score(y_test, y_pred_test_lasso))

#####################Regression Evaluation Metrics####################
from sklearn import metrics
#The MAE score is measured as the average of the absolute error values. 
#The Absolute is a mathematical function that makes a number positive. 
#Therefore, the difference between an expected value and a predicted value 
#can be positive or negative and will necessarily be positive when calculating the MAE.
print("Mean Absoulte Error(MAE):", metrics.mean_absolute_error(y_test, y_pred))
#Mean Squared Error (MSE) is more popular than MAE, 
#because MSE "punishes" larger errors, which tends to be useful in the real world.
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
#RMSE- It measures the average difference between values predicted by a model and 
#the actual values. It provides an estimation of how well the model is able 
#to predict the target value (accuracy).
print("Root Mean Squared Error :", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

