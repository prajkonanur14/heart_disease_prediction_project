###################### Heart Disease Prediction Project #######################
### Import Libraries ###
#General Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Machine Learning Libraries
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#Evaluation Metric Libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import jaccard_score, f1_score, accuracy_score
import warnings
warnings.simplefilter(action='ignore')


################################# Import Data #################################
filepath = 'framingham.csv'
df = pd.read_csv(filepath)


############################# Data Pre-Processing #############################
df = df.astype('float')
cigs_avg = np.mean(df['CigsPerDay'])
tot_chol_avg = np.mean(df['TotChol'])
bmi_avg = np.mean(df['BMI'])
heart_rate_avg = np.mean(df['HeartRate'])
glucose_avg = np.mean(df['Glucose'])
#Fill in missing values
df['Education'].replace(np.NaN, 1.0, inplace=True) #Education
df['BPMeds'].replace(np.NaN, 0, inplace=True) #BPMeds
df['CigsPerDay'].replace(np.NaN, cigs_avg, inplace=True) #CigsPerDay
df['TotChol'].replace(np.NaN, tot_chol_avg, inplace=True) #TotChol
df['BMI'].replace(np.NaN, bmi_avg, inplace=True) #BMI
df['HeartRate'].replace(np.NaN, heart_rate_avg, inplace=True) #HeartRate
df['Glucose'].replace(np.NaN, glucose_avg, inplace=True) #Glucose
#Drop 1st colunmn
df.drop('Unnamed: 0', axis=1, inplace=True)

#Set X (variables) and y
variables = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']


####################### K-Nearest Neighbors (KNN) Model #######################
#Function to find optimal k value
def optimal_k(x_train, x_test, y_train, y_test):
    knn_list = []
    for k in range(1,30):
        #Fit model
        neigh = KNeighborsClassifier(n_neighbors = k)
        neigh.fit(x_train, y_train)
        #Make prediction
        yhat = neigh.predict(x_test)
        #Calculate accuracy, jaccard, and f1 scores
        acc_score = np.round(accuracy_score(y_test, yhat), 4)
        jac_score = np.round(jaccard_score(y_test, yhat), 4)
        f1_score1 = np.round(f1_score(y_test, yhat), 4)
        #Append k and accuracy score to lists
        knn_list.append([k, acc_score, jac_score, f1_score1])
    return knn_list
#Function to plot optimal k values
def plot_k(knn_list):
    df = pd.DataFrame(knn_list, columns=['K', 'Accuracy Score', 'Jaccard Index',
                                         'F1 Score'])
    df_plot = df[['K', 'Accuracy Score']]
    fig, ax = plt.subplots(figsize = (10,6))
    ax.plot(df_plot, 
            marker = 'o',
            markersize = 3,
            color = 'firebrick',
            linestyle = 'dotted')
    ax.set_title('K values vs Accuracy Score')
    ax.set_xlabel('K Value')
    ax.set_ylabel('Accuracy Score')
    ax.grid()
    plt.xlim(0, 30)
    plt.ylim(0.7, 1)
        
#Normalize Data
X_knn = variables.values
X_knn = StandardScaler().fit(X_knn).transform(X_knn.astype(float))
y_knn = y.values
#Train-Test split
x_train_knn,x_test_knn,y_train_knn,y_test_knn = train_test_split(X_knn,y_knn,
                                                                     test_size=0.2,
                                                                     random_state=5)
solution = optimal_k(x_train_knn, x_test_knn, y_train_knn, y_test_knn)
#plot_k(solution)



############################# Decision Tree Model #############################
X_tree = variables.values
y_tree = y.values
#Train-Test split
x_train_tree,x_test_tree,y_train_tree,y_test_tree = train_test_split(X_tree,y_tree,
                                                                     test_size=0.2,
                                                                     random_state=5)
#Fit Decision Tree Model
dTree = DecisionTreeClassifier(criterion='entropy')
dTree.fit(x_train_tree, y_train_tree)
#Predict Decision Tree Model
yhat_tree = dTree.predict(x_test_tree)
#Evaluation Metrics
acc_score_tree = accuracy_score(y_test_tree, yhat_tree)
jac_index_tree = jaccard_score(y_test_tree, yhat_tree)
f1_score_tree = f1_score(y_test_tree, yhat_tree)


#################### Artificial Neural Network (ANN) Model ####################
















#df.to_csv('framingham.csv')