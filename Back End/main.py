import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplt
import seaborn as sb  # for data visualization based on matplotlib
from sklearn.model_selection import train_test_split as ttsp
from sklearn.ensemble import RandomForestClassifier
# importing parameters for the evaluation of the model's performance
from sklearn.metrics import accuracy_score, precision_score, classification_report, recall_score, matthews_corrcoef, confusion_matrix, f1_score
import datetime
import joblib as jb

# Load the dataset
cardData = pd.read_csv("Data set\dataset.csv")
X = cardData.drop(['Class'], axis=1)  # Input features (all columns except 'Class')
Y = cardData["Class"]  # Output labels (Class)

# Load the pre-trained model
model = jb.load("creditCardFraudDetectionModel.pkl")

# Function to detect fraud based on user input
def detectFraud(rfc, X):
    # Get current time
    current_time = datetime.datetime.now()
    
    # Ask for user inputs
    amt = float(input("Enter the transaction amount: "))
    last_transaction_time_input = input("Enter the time of the last transaction (YYYY-MM-DD HH:MM:SS): ")
    
    # Convert the entered last transaction time to datetime
    last_transaction_time = datetime.datetime.strptime(last_transaction_time_input, "%Y-%m-%d %H:%M:%S")
    
    # Calculate the time difference in seconds between the current time and the last transaction time
    time_diff = current_time - last_transaction_time    
    time_seconds = time_diff.total_seconds()
    
    # Initialize the feature vector with mean values from the dataset (except 'Class')
    mean_values = X.mean().values
    
    # Replace the first element (time difference) and the last element (amount transacted) in the feature vector
    mean_values[0] = time_seconds  # This represents time since the last transaction
    mean_values[29] = amt         # This represents the amount of the current transaction
    
    # Reshape the array to have one row (as required by the model for prediction)
    userTransactions = np.array(mean_values).reshape(1, -1)
    
    # Make the prediction using the trained model
    prediction = model.predict(userTransactions)
    
    # Output the result based on the prediction
    if prediction[0] == 1:
        print("Fraudulent transaction suspected!!!")
    else:
        print("Transaction is legitimate!!!")    

# Call the detectFraud function to test
detectFraud(model, X)

# Ask if the user wants to verify another transaction
choice = input("Do you want to verify another transaction? (yes/no): ")

while choice.lower() == "yes":
    detectFraud(model, X)
    choice = input("Do you want to verify another transaction? (yes/no): ")
