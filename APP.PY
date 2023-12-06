# Streamlit app for predicting Telco Customers Churn

# Importing Libraries

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import seaborn as sns

sns.set()

import warnings
warnings.filterwarnings('ignore')

#App title
st.title('TELCO CUSTOMERS CHURN')


#add an upload button for data upload
uploaded_file=st.file_uploader("upload your input csv file",
                               type=["csv"])

#create a dataframe from the uploaded fle
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    #Get input from user on the number of rows to be displayed
    num_row = st.number_input('Enter the number of rows to display',
                              min_value=0, max_value=30, value=5)
    
st.header(" Data Sample")
 #show the top 5 rows of the dataframe
st.dataframe(data.head(num_row))

#Create a function to plot categorical variable
def plot_cat(data, cat_var):
    st.header("Plot of " + cat_var)
    fig, ax = plt.subplots()
    sns.set_style('darkgrid')
    sns.countplot(data=data, x=cat_var)
    plt.title(cat_var)
    plt.show()
    st.pyplot(fig)
    
# Get a list of all the columns in the data frame
columns = data.columns.tolist()

# Create a dropdown where user can select column to plot
cat_var = st.selectbox("Select column to plot", columns)

#Plot the selected column
plot_cat(data, cat_var)

#Removing all unimportant columns
data = data.drop(['Zip Code','Churn Reason','City','Churn Score','Churn Value','CLTV','CustomerID','Lat Long',
                  'Latitude','Longitude'], axis = 1)

#use ordinal encoding to convert categorical variables to numerical
data['Churn Label'].replace(to_replace='Yes', value=1, inplace=True)
data['Churn Label'].replace(to_replace='No',  value=0, inplace=True)

def encode_data(dataframe_series):
    if dataframe_series.dtype=='object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series

data = data.apply(lambda x: encode_data(x))


# show top 5 rows of the updated dataset
st.header(" Encoded Data Sample")
st.write("""
          After removing all unimportant columns; 
         'Zip Code', 'Churn Reason', 'City', 'Churn Score', 'Churn Value', 'CLTV', 'CustomerID', 'Lat Long', 
                  'Latitude', 
                  'Longitude'
         """)
st.dataframe(data.head(5))

# split data into target and features
X = data.drop(columns = ['Churn Label'])
y = data['Churn Label'].values

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state = 40, stratify=y)

# RANDOM FOREST

# Instantiate and fit the model
st.header("Gradient Boosting Classifier")
gbc=GradientBoostingClassifier()
gbc.fit(X_train,y_train)

y_pred=gbc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,mean_squared_error,r2_score
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print("Training Score: ",gbc.score(X_train,y_train)*100)
print(gbc.score(X_test,y_test))

st.write("""
          Accuracy score
         """)
st.text(accuracy_score(y_test,y_pred)*100)

# Display classification report
st.text("Classification Report")
classification_rep = classification_report(y_test, y_pred)
st.text(classification_rep)

# get user input on the number of rows to display
num_row_pred = st.number_input('Enter the number of rows to display',
                               min_value=1, max_value=50, value=5)

# show the top 5 rows of the dataframe
st.header('Prediction')
st.dataframe(data.head(num_row_pred))

# print classification report
st.header("Classification Report")
st.text("0 = Will not churn, 1 = Will churn")

class_report = classification_report(data['Churn Label'],
                                     data['Churn Label'])

st.text(class_report)