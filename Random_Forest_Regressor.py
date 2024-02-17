#importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

#loading the dataset
insurance1=pd.read_csv(r"C:\Users\Administrator\Desktop\Project\RandomForestRegressor\insurance.csv")
insurance=pd.DataFrame(insurance1)
print(insurance.info())
#check number of rows and cloumns
print(insurance.shape)

#To check number of null values in each column
print(insurance.isnull().sum())

#There are 396 null values in "age" column and 956 in "bmi" 
#column.Dropping this rows may lead to lose of crucial data set, 
#so we are going to fill these values with respective "mean" values.

from sklearn.impute import SimpleImputer

# Initialize the SimpleImputer with the strategy 'mean'
imputer = SimpleImputer(strategy='mean')

# Specify the columns with missing values that we want to impute
columns_with_missing_values = ['age', 'bmi']

# Apply the imputer to fill missing values with the mean
insurance[columns_with_missing_values] = imputer.fit_transform(insurance[columns_with_missing_values])
print(insurance.isnull().sum())

print("no. of rows",insurance.shape[0])
print("no. of columns",insurance.shape[1])
## no data is lost,while dealing with missing values

print(insurance.describe(include='all'))
print(insurance.info())

# we have to convert the object to int or float for upcoming machine learning algortham
#"sex","hereditary_diseases","city","job_title"

#values mapped to integers (0 and 1)
insurance['sex']=insurance['sex'].map({'female':0,'male':1})
print(insurance["sex"].unique())

insurance["hereditary_diseases"]=insurance["hereditary_diseases"].map({'NoDisease':0, 'Epilepsy':1, 'EyeDisease':2, 'Alzheimer':3, 'Arthritis':4,
       'HeartDisease':5, 'Diabetes':6, 'Cancer':7, 'High BP':8, 'Obesity':9})
print(insurance["hereditary_diseases"].unique())

#label encoding 
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder instance
label_encoder = LabelEncoder()

# Encode the "city" column
insurance['city'] = label_encoder.fit_transform(insurance['city'])
insurance['job_title'] = label_encoder.fit_transform(insurance['job_title'])
print(insurance['city'].unique())

#label encoding done
print(insurance['job_title'].unique())

print(insurance.info())

#Selecting Columns for Features and Target

# setting feature and target columns
X=insurance.drop(['claim'],axis=1)
y=insurance['claim']

######################### Heart Map ########################
#Correlation Matrix: Visualize the correlation between numerical features.
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create a custom colormap with modified color intensity
custom_cmap = sns.color_palette(['#ffffcc', '#a1dab4', '#41b6c4', '#225ea8'])
correlation_matrix = insurance.corr()

# Create a heatmap with the custom colormap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap=custom_cmap)
plt.title('Correlation Matrix')
plt.show()

########### Train-Test Split ##############
# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(y_train[0])

# Import RandomForest Regressor from scikit-learn (Sklearn) for Regressor tasks
from sklearn.ensemble import RandomForestRegressor
# Initialize and train a Random Forest Regressor model
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Make predictions using the Random Forest Regressor (RF) model on the test data
y_pred = rf.predict(X_test)
print(y_pred)

df1=pd.DataFrame({'Actual':y_test,'RF':y_pred})
################# Visuals of RandomForest Model##############
plt.subplot(223)
plt.plot(df1['Actual'].iloc[0:11], label='Actual')
plt.plot(df1['RF'].iloc[0:11], label="RF")
plt.title('Random Forest Predictions')
plt.show()

#random forest regressor is giving most accurate results

############################-Model Performnce-Evalution Metrics-#################
from sklearn import metrics
# Calculate and store the R-squared (R2) score for the Random Forest Regressor (RF) model
rf_r2_score = metrics.r2_score(y_test, y_pred)
print("RandomForest Model R2 Score",rf_r2_score)
# o/p: it excels in explaining approximately 96.64% of the variance in health insurance claim amounts. 

#The MAE score is measured as the average of the absolute error values. 
#The Absolute is a mathematical function that makes a number positive. 
#Therefore, the difference between an expected value and a predicted value 
#can be positive or negative and will necessarily be positive when calculating the MAE
# Calculate and store the Mean Absolute Error (MAE) for the Random Forest Regressor (RF) model
mae_rf = metrics.mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE) for the Random Forest Regressor",mae_rf)

#Mean Squared Error (MSE) is more popular than MAE, 
#because MSE "punishes" larger errors, which tends to be useful in the real world.
print("Mean Squared Error For Random Forest Regressor:", metrics.mean_squared_error(y_test, y_pred))
#RMSE- It measures the average difference between values predicted by a model and 
#the actual values. It provides an estimation of how well the model is able 
#to predict the target value (accuracy).
print("Root Mean Squared Error For Random Forest Regressor:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Model Performance Based on MAE
#The Random Forest Regressor (RF) stands out as the best-performing model,
# with the lowest MAE, signifying its accuracy in predicting health insurance claim amounts.
#####################prediction on random new customer###############
new_df = pd.DataFrame(X_train,index=[0])
new_df
#Use the trained Random Forest Regressor (RF) model to make predictions for a new customer
new_pred_rf = rf.predict(new_df)

#Print the predicted medical insurance cost for the new customer using Random Forest Regressor
print("Medical Insurance claim for New Customer (Random Forest Regressor) is:", new_pred_rf[0])
print(y_train[0])
#Random Forest Regressor (RF) model is predicting more accurately

# Create a dictionary with the sample data
sample_data = {
    'age': [50.0],
    'sex': [0],
    'weight': [55],
    'bmi': [24.3],
    'hereditary_diseases': [0],
    'no_of_dependents': [1],
    'smoker': [0],
    'city': [55],
    'bloodpressure': [72],
    'diabetes': [0],
    'regular_ex': [0],
    'job_title': [2]
}

# Create a DataFrame from the sample data
sample_df = pd.DataFrame(sample_data)

# Display the sample DataFrame
print(sample_df)

# Use the trained Random Forest Regressor (RF) model to make predictions for a new customer
new_pred_rf = rf.predict(sample_df)

# Print the predicted medical insurance cost for the new customer using Random Forest Regressor
print("Medical Insurance claim for New Customer (Random Forest Regressor) is:", new_pred_rf[0])

