import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv("StudentPerformanceFactors.csv")

print("First five rows of the dataset:")
print(df.head())
print("\nDataset shape:", df.shape)
df.columns = df.columns.str.strip() #remove spaces from columns
print("\nColumn names:")
print(df.columns.tolist())

if "Exam_Score" not in df.columns:
    raise ValueError("Exam_Score column not found")

df["Exam_Score"]=pd.to_numeric(df["Exam_Score"],errors="coerce")  #errors="coerce" convert invalid score into missing value
df=df.dropna(subset=["Exam_Score"]).copy()

X=df.drop("Exam_Score",axis=1)
y=df["Exam_Score"]

X = pd.get_dummies(X, dtype=int)
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)  #split dataset into 80(training) 20(testing)

#-----------------------------------------------------------------------------------------------------------------------------------------
print("\nLINEAR REGRESSION")
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)
#evaluation metrics
mae = mean_absolute_error(y_test,linear_predictions)
mse = mean_squared_error(y_test,linear_predictions)
r2 = r2_score(y_test,linear_predictions)

print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("R2 Score:", round(r2, 4))

#-----------------------------------------------------------------------------------------------------------------------------------------
# separate training and testing split data for the classification algorithms
y_classification = (df["Exam_Score"] >= 70).astype(int)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = (train_test_split(X,y_classification,test_size=0.20,random_state=42,stratify=y_classification))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_cls)
X_test_scaled = scaler.transform(X_test_cls)

#-----------------------------------------------------------------------------------------------------------------------------------------
print("\n")
print("\nLOGISTIC REGRESSION")
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_cls, y_train_cls)
logistic_predictions = logistic_model.predict(X_test_cls)
logistic_accuracy = accuracy_score(y_test_cls, logistic_predictions)
print("Accuracy:", round(logistic_accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test_cls,logistic_predictions,zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test_cls, logistic_predictions))

#-----------------------------------------------------------------------------------------------------------------------------------------
print("\n")
print("\nK-NEAREST NEIGHBORS")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train_cls)
knn_predictions = knn_model.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test_cls, knn_predictions)
print("Accuracy:", round(knn_accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test_cls,knn_predictions,zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test_cls, knn_predictions))

#-----------------------------------------------------------------------------------------------------------------------------------------
print("\n")
print("\nSUPPORT VECTOR MACHINE")
svm_model = SVC(kernel="rbf")
svm_model.fit(X_train_scaled, y_train_cls)
svm_predictions = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test_cls, svm_predictions)
print("Accuracy:", round(svm_accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test_cls,svm_predictions,zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test_cls, svm_predictions))

#-----------------------------------------------------------------------------------------------------------------------------------------" 
