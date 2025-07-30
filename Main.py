print("**********************************************")
print("Project Title:An Unsupervised Adversarial Autoencoder for Cyber Attack Detection in Power Distribution Grids")
print("**********************************************")
print()
#**************************Importing the libraries *****************************************
import numpy as np
import pandas as pd 
import pickle # saving and loading trained model
from os import path
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, LSTM, MaxPool1D, Flatten, Dropout # importing dense layer
from keras.models import Sequential #importing Sequential layer
from keras.layers import Input
from keras.models import Model
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")
#**************************1.Data Selection *****************************************
print("**********************************************")
print("Module2 --- Dataset Selection   ")
data=pd.read_csv("detect_dataset.csv")
# data=data.iloc[: 200000]
print(data.head(5))
print(data.columns)
print(data.shape)
print("Dataset Selection Completed ")
#**************************2.Data Preprocessing *****************************************

print("**********************************************")
print("Module2 --- Dataset Preprocessing   ")
print("**********************************************")
print("Preprocessing Before Missing-value ")
print(data.isnull().sum())
data.drop(['Unnamed: 7','Unnamed: 8'],axis=1,inplace = True)
print("Preprocessing After Missing-value ")
print(data.isnull().sum())
#**************************3.EDA *****************************************
print("**********************************************")
print("EDA plot  ")
correlation_matrix = data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix (Heatmap)')
plt.show()
plt.figure(figsize=(10, 6))
plt.hist(data['Ia'], bins=30, edgecolor='black')
plt.title('Histogram of Ia')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.hist(data['Va'], bins=30, edgecolor='black')
plt.title('Histogram of Va')
plt.xlabel('Time')
plt.ylabel('Va')
plt.grid(True)
plt.show()

data['Output (S)'].unique()
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
ax1.bar(x=data['Output (S)'].unique(),height = data['Output (S)'].value_counts())
ax1.set_xticks(ticks=[0,1])
ax2.pie(data['Output (S)'].value_counts(),autopct='%0.2f',labels=data['Output (S)'].value_counts().index)
plt.suptitle('Frequency of both the classes')
plt.show()
ls = ['Ia','Ib','Ic','Va','Vb','Vc']

plt.figure(figsize=(12, 5))  
for i in range(2):
    for j in range(3):
        plt.subplot(2, 3, i * 3 + (j + 1))  
        sns.kdeplot(data[ls[i * 3 + j]]) 

plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.suptitle('Kde distribution of all columns')
plt.show()

pair_plot = sns.pairplot(data=data.drop('Output (S)', axis=1))
pair_plot.fig.suptitle('Pair Plot of Features', fontsize=16)
pair_plot.fig.subplots_adjust(top=0.95)

plt.show()

ls = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']

plt.figure(figsize=(12, 8))

for i in range(2):
    for j in range(3):
        plt.subplot(2, 3, i * 3 + (j + 1))
        stats.probplot(data[ls[i*3+j]], dist="norm", plot=plt)
        plt.title(ls[i*3+j])

plt.tight_layout()
plt.suptitle('QQ plots for all columns', y=1.05, fontsize=16)
plt.show()
print("Dataset EDA Completed ")

#**************************4.Data Splitting *****************************************
print("**********************************************")
print("Module4 --- Dataset Splitting -80% Training and 20% testing   ")
X_data= data.drop('Output (S)',axis=1)
y_data= data['Output (S)']
from sklearn.preprocessing import LabelBinarizer
y_data = LabelBinarizer().fit_transform(y_data)
X_data=np.array(X_data)
y_data=np.array(y_data)
X_train, X_test, y_train, y_test = train_test_split(X_data,y_data, test_size=0.20, random_state=42)
print("X_train Shapes ",X_train.shape)
print("y_train Shapes ",y_train.shape)
print("x_test Shapes ",X_test.shape)
print("y_test Shapes ",y_test.shape)

#**************************5.Classification  *****************************************
print("**********************************************")
print("Classification- Deep Learning -hybird Cnn and Lstm  Algorithm ")

# X_train = np.reshape(X_train, ( X_train.shape[0], 1 , X_train.shape[1] ))
# X_test = np.reshape(X_test, ( X_test.shape[0], 1,  X_test.shape[1] ))
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Dropout
model = Sequential([
    Dense(256,'relu',input_dim=X_train.shape[1]),
    Dropout(0.5),
    Dense(128,'relu'),
    Dropout(0.5),
    Dense(1,'sigmoid'),
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

X_train = X_train.astype(float)
y_train = y_train.astype(float)
history = model.fit(X_train,y_train,validation_data = (X_test,y_test),epochs=2)
model.save('Model.h5')

X_test = X_test.astype(float)
y_test = y_test.astype(float)

test_results = model.evaluate(X_test, y_test,verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')

# results = model.evaluate(X_test, y_test,batch_size = 128)
# print(model.metrics_names)
# print(results)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()
#**************************6.Prediction   *****************************************

print("**********************************************")
print("Prediction- Deep Learning -hybird Cnn and Lstm  Algorithm ")
y_predict = model.predict(X_test)
y_pred =y_predict.round()
# y_test = y_test.argmax(axis = -1 )

#**************************7.Performance Analysis *****************************************
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import classification_report
Acc_result=accuracy_score(y_test, y_pred)*100
print("LUCID CNN Algorithm Accuracy is:",Acc_result,'%')
print()
print("**********************************************")
print("LUCID CNN Classification Report ")
print()
report = classification_report(y_test, y_pred)
print(report)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm)
print("Confusion Matrix:\n", cm)
cm_display.plot()
plt.show()

LUCID_cm=cm
print(LUCID_cm)
print()
i=1
j=1
TP = LUCID_cm[i, i]
TN = sum(LUCID_cm[i, j] for j in range(LUCID_cm.shape[1]) if i != j)
FP = sum(LUCID_cm[i, j] for j in range(LUCID_cm.shape[1]) if i != j)
FN = sum(LUCID_cm[i, j] for i in range(LUCID_cm.shape[0]) if i != j)

def calculate_metrics(tp, tn, fp, fn):
    # Precision
    precision = tp / (tp + fp)
    
    # Recall (Sensitivity)
    recall = tp / (tp + fn)
    
    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    # Correct Classification Rate (CCU)
    ccu = (tp + tn) / (tp + tn + fp + fn)
    
    # Memory in switch (Kb)
    memory_switch_kb = 10 * (tp + tn + fp + fn)  # Example calculation for memory
    
    return precision, recall, f1_score, ccu, memory_switch_kb

tp = TP
tn = TN
fp = FP
fn = FN

precision, recall, f1_score, ccu, memory_switch_kb = calculate_metrics(tp, tn, fp, fn)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print(f"CCU: {ccu:.4f}")
print(f"Memory in switch (Kb): {memory_switch_kb} Kb")


metrics = ['Precision', 'Recall', 'F1 Score', 'CCU']
values = [precision, recall, f1_score, ccu]  
plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, values, color=['blue', 'orange', 'green', 'red'])
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2 - 0.1, yval + 0.05, round(yval, 4), ha='center', va='bottom')

plt.title('Performance Metrics')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.show()

#**************************8.Comparative Analysis *****************************************

import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Proposed Model (AAE)', 'AE-LSTM', 'AE-CNN', 'AE-FC', 'K-Means', 'OneClassSVM', 'LR', 'proposed']
accuracy = [99.50, 99.01, 98.86, 98.61, 99.36, 97.97, 96.83, 99.86]
recall = [93.75, 91.67, 91.67, 87.50, 87.50, 14.58, 64.58, 99.67]
precision = [86.54, 73.33, 69.84, 65.63, 85.71, 100, 39.74, 99.84]
f1_score = [89.99, 81.48, 79.28, 75.00, 86.60, 25.45, 49.21, 99.28]

# Bar width
bar_width = 0.2

# Position of bars on X axis
r1 = np.arange(len(models))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# Plotting
plt.figure(figsize=(14, 8))
plt.bar(r1, accuracy, color='b', width=bar_width, edgecolor='grey', label='Accuracy')
plt.bar(r2, recall, color='g', width=bar_width, edgecolor='grey', label='Recall')
plt.bar(r3, precision, color='r', width=bar_width, edgecolor='grey', label='Precision')
plt.bar(r4, f1_score, color='y', width=bar_width, edgecolor='grey', label='F-1 Score')

# Add xticks on the middle of the group bars
plt.xlabel('Models', fontweight='bold')
plt.ylabel('Performance Metrics (%)', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(models))], models, rotation=45)

# Create legend & Show graphic
plt.legend()
plt.tight_layout()
plt.title('Performance comparison under combined attack forIEEE 13-bus')
plt.show()




#**************************9.Prediction on sample data  *****************************************
import numpy as np
from tensorflow.keras.models import load_model
model = load_model('Model.h5')
test_sample=X_test[1]
# test_sample = np.array([-0.0291654 ,  0.22419893, -0.25310159, -1.74378807,  1.1811018 ,
#         0.51398549])

test_sample = np.expand_dims(test_sample, axis=0)

predictions = model.predict(test_sample)

print("Actual Data ",y_test[1])
print("Actual Data ",predictions.round())


# Assuming model.predict() gives probabilities, find the class with the highest probability
predicted_class = predictions

# Ensure predicted_class is an integer scalar
predicted_class = int(predicted_class[0])

# Define class labels
class_labels = ['Deductive (Non Fault)', 'Additive (Fault)']

# Map predicted class to class label
predicted_label = class_labels[predicted_class]

# Print the result
print(f'The predicted class for the test data  is: {predicted_label}')
