import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplt
import seaborn as sb #for data visualization based on matplotlib
from matplotlib import gridspec # to have multiple plots in the same window space
from sklearn.model_selection import train_test_split as ttsp
from sklearn.ensemble import RandomForestClassifier
# importing parameters for the evaluation of the model's performance 
from sklearn.metrics import accuracy_score, precision_score, classification_report, recall_score, matthews_corrcoef, confusion_matrix, f1_score
import datetime
import joblib as jb
from imblearn.over_sampling import SMOTE

cardData = pd.read_csv("Data set\dataset.csv") 
X = cardData.drop(['Class'], axis = 1) # takes all columns of the data set as input except the 'Class' column
Y = cardData["Class"] # the 'Class' column is to be recieved as an output
smote = SMOTE() #INSTANCE CREATION
newX, newY = smote.fit_resample(X, Y) # creating synthetic fraudlent entriws to balanace the data set 


def trainModel(X, Y):    

    fraud = cardData[cardData["Class"]==1] #transactions having value 1 for the class attribute are to be considered fradulent

    legit = cardData[cardData["Class"] == 0]

    outlierFraction = len(fraud)/float(len(legit)) #number of fraud transactions relative to the legitimate ones.

    # Correlation matrix
    corrmat = cardData.corr()
    fig = pyplt.figure(figsize = (12, 9))
    sb.heatmap(corrmat, vmax = .8, square = True)
    pyplt.show()

    # dividing the X and the Y from the dataset
    

    xData = newX.values
    yData = newY.values

    # splitting the dataset to training and testing parts

    xTrain, xTest, yTrain, yTest = ttsp(xData, yData, test_size=0.2, random_state=42) # using 20% of the data for testing and rest fro training the model

    print("Starting to train the model....")

    rfc = RandomForestClassifier(class_weight="balanced")  # instance creation 
    rfc.fit(xTrain, yTrain) # training the modprint("Training successful !!")
    print("Training successful !!!")
    results = rfc.predict(xTest) # Testing the model on the 20% of the data stored for it

    # Analysing the model's performance 

    outliers = len(fraud)
    errors = (results!=yTest).sum()  # counting the number of cases where the results predicted by the model dont match with the actual testing data

    print("Number of mismatches in the results predicted by the model : {}".format(errors))

    print("The model used here is the Random Forest classifier")

    accuracy = accuracy_score(yTest, results)
    print("The accuracy is {}".format(accuracy))

    precision = precision_score(yTest, results)
    print("The precision is {}".format(precision))

    recallScore = recall_score(yTest, results)
    print("The recall is {}".format(recallScore))

    f1 = f1_score(yTest, results)
    print("The F1-Score is {}".format(f1))

    correlation = matthews_corrcoef(yTest, results)
    print("The Matthews correlation coefficient is{}".format(correlation))


    # Plotting the confusion matrix 

    LABELS = ['Normal', 'Fraud']
    conf_matrix = confusion_matrix(yTest, results)
    pyplt.figure(figsize =(12, 12))
    sb.heatmap(conf_matrix, xticklabels = LABELS, 
                yticklabels = LABELS, annot = True, fmt ="d");
    pyplt.title("Confusion matrix")
    pyplt.ylabel('True class')
    pyplt.xlabel('Predicted class')
    pyplt.show()
    return rfc

rfc = trainModel(newX, newY)
jb.dump(rfc, "creditCardFraudDetectionModel.pkl")
print("Model saved successfully !!!")
