# Internship Project - Human Action Detection
# Name: Deepak Gupta
# Domain: Machine Learning
# Step 1: Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import statsmodels.api as sm

from sklearn.metrics import classification_report, r2_score, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

df = pd.read_csv("dataset/mhealth_raw_data.csv")

print(df.head())
print(df.head())        # first 5 rows
print(df.tail())        # last 5 rows
print(df.shape)         # rows & columns
print(df.columns)       # column names
print(df.info())        # data types
print(df.describe())    # stats (mean, min, max)


# step 2 data importing


plt.figure(figsize=(10,6))
df['Activity'].value_counts().plot(kind='bar')

plt.title("Activity Distribution")
plt.xlabel("Activity")
plt.ylabel("Count")

plt.show()

# Separate majority and minority class

data_activity_0 = df[df["Activity"] == 0]
data_activity_else = df[df["Activity"] != 0]
# Downsample majority class

data_activity_0 = data_activity_0.sample(n=min(40000, len(data_activity_0)))
# Combine both datasets

df = pd.concat([data_activity_0, data_activity_else])
plt.figure(figsize=(10,6))
df['Activity'].value_counts().plot(kind='bar')
plt.title("Activity Distribution")
plt.xlabel("Activity")
plt.ylabel("Count")

plt.show()
print(len(df))
 
# # step 3 EDA

# Activity Labels
activity_label = {
    0: "None",
    1: "Standing still (1 min)",
    2: "Sitting and relaxing (1 min)",
    3: "Lying down (1 min)",
    4: "Walking (1 min)",
    5: "Climbing stairs (1 min)",
    6: "Waist bends forward (20x)",
    7: "Frontal elevation of arms (20x)",
    8: "Knees bending (crouching) (20x)",
    9: "Cycling (1 min)",
    10: "Jogging (1 min)",
    11: "Running (1 min)",
    12: "Jump front & back (20x)"
}

# Select one subject
subject1 = df[df['subject'] == 'subject1']

# Readings (a = ankle, g = gyro)
readings = ['a', 'g']

# Loop through activities normal graph

# for i in range(1, 13):
#     for r in readings:
#         print(f"\n===== {activity_label[i]} - {r} =====")

#         plt.figure(figsize=(14,4))
# # LEFT SENSOR (ankle)
#         plt.subplot(1,2,1)
#         plt.plot(subject1[subject1['Activity'] == i].reset_index(drop=True)['alx'], label='alx')
#         plt.plot(subject1[subject1['Activity'] == i].reset_index(drop=True)['aly'], label='aly')
#         plt.plot(subject1[subject1['Activity'] == i].reset_index(drop=True)['alz'], label='alz')

#         plt.title("Left ankle sensor")
#         plt.legend()


# # RIGHT SENSOR
#         plt.subplot(1,2,2)
#         plt.plot(subject1[subject1['Activity'] == i].reset_index(drop=True)['arx'], label='arx')
#         plt.plot(subject1[subject1['Activity'] == i].reset_index(drop=True)['ary'], label='ary')
#         plt.plot(subject1[subject1['Activity'] == i].reset_index(drop=True)['arz'], label='arz')

#         plt.title("Right ankle sensor")
#         plt.legend()

#         plt.title("Right wrist sensor")
#         plt.legend()

#         plt.show()

for i in range(1,13):
    for r in readings:
        print(f"\n===== {activity_label[i]} - {r} =====")

        plt.figure(figsize=(14,4))

        # LEFT SENSOR
        plt.subplot(1,2,1)
        plt.hist(subject1[subject1['Activity'] == i].reset_index(drop=True)['alx'], alpha=0.7, label='alx')
        plt.hist(subject1[subject1['Activity'] == i].reset_index(drop=True)['aly'], alpha=0.7, label='aly')
        plt.hist(subject1[subject1['Activity'] == i].reset_index(drop=True)['alz'], alpha=0.7, label='alz')

        plt.title("Left ankle sensor")
        plt.legend()

        # RIGHT SENSOR
        plt.subplot(1,2,2)
        plt.hist(subject1[subject1['Activity'] == i].reset_index(drop=True)['arx'], alpha=0.7, label='arx')
        plt.hist(subject1[subject1['Activity'] == i].reset_index(drop=True)['ary'], alpha=0.7, label='ary')
        plt.hist(subject1[subject1['Activity'] == i].reset_index(drop=True)['arz'], alpha=0.7, label='arz')

        plt.title("Right wrist sensor")
        plt.legend()

        plt.show()

df['Activity'] = df['Activity'].replace(
    [0,1,2,3,4,5,6,7,8,9,10,11,12],
    [
        'None',
        'Standing still (1 min)',
        'Sitting and relaxing (1 min)',
        'Lying down (1 min)',
        'Walking (1 min)',
        'Climbing stairs (1 min)',
        'Waist bends forward (20x)',
        'Frontal elevation of arms (20x)',
        'Knees bending (crouching) (20x)',
        'Cycling (1 min)',
        'Jogging (1 min)',
        'Running (1 min)',
        'Jump front & back (20x)'
    ]
)

print(df.Activity.value_counts())
# -------- STEP 4: ENCODING --------



# Balance data (same as video) pue chart
# -------- BALANCE DATA --------
data_activity_0 = df[df["Activity"] == 0]
data_activity_else = df[df["Activity"] != 0]

data_activity_0 = data_activity_0.sample(n=min(40000, len(data_activity_0)))

df = pd.concat([data_activity_0, data_activity_else])
plt.figure(figsize=(12,8))

round(df['Activity'].value_counts() / df.shape[0] * 100, 2).plot.pie(autopct='%1.1f%%')
plt.axis('equal')
plt.show()

# outline removal (same as video)

df1 = df.copy()
for feature in df1.select_dtypes(include=[np.number]).columns:# last column (Activity) skip
    lower_range = np.quantile(df1[feature], 0.01)
    upper_range = np.quantile(df1[feature], 0.99)

    print(feature, "range:", lower_range, "to", upper_range)

    df1 = df1.drop(
        df1[(df1[feature] > upper_range) | (df1[feature] < lower_range)].index,
        axis=0
    )

    print("shape:", df1.shape)
    
    print(df1)

## step 4 data preprocessing
le = LabelEncoder()
df['subject'] = le.fit_transform(df['subject'])

# Label Encoding for Activity
df['Activity'] = le.fit_transform(df['Activity'])

# Boxplot (outliers check)
df.plot(kind='box', subplots=True, layout=(5,5), figsize=(20,15))

plt.show()

# ##Step 5: Model Building
# -------- PREPROCESSING --------
# ## Step 5: Model Building

# -------- FUNCTION (PEHLE DEFINE KAR) --------
def resultsSummarizer(y_true, y_pred, cm_en=True):
    cm = confusion_matrix(y_true, y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    if cm_en:
        plt.figure(figsize=(15,15))

       
        sns.heatmap(cm, annot=True, cmap="Blues")
        plt.title('Confusion Matrix')
        plt.show()

    print("Accuracy Score : " + '{:.4%}'.format(acc))
    print("Precision Score: " + '{:.4%}'.format(prec))
    print("Recall Score   : " + '{:.4%}'.format(rec))
    print("F1 Score       : " + '{:.4%}'.format(f1))

# -------- PREPROCESSING --------
X = df.drop(['Activity', 'subject'], axis=1)
y = df['Activity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------- MODEL 1: Logistic Regression --------
print("\n===== Logistic Regression =====")

lr = LogisticRegression(max_iter=300)

lr.fit(X_train_scaled, y_train)

# Train accuracy
print("Train Accuracy:", lr.score(X_train_scaled, y_train))

# Test accuracy
print("Test Accuracy:", lr.score(X_test_scaled, y_test))

# Prediction
y_pred = lr.predict(X_test_scaled)

# Final output
resultsSummarizer(y_test, y_pred)


# -------- KNN MODEL --------
print("\n===== KNN =====")

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

# Train accuracy
print("Train Accuracy:", knn.score(X_train_scaled, y_train))

# Test accuracy
print("Test Accuracy:", knn.score(X_test_scaled, y_test))

# Prediction
y_pred_knn = knn.predict(X_test_scaled)

# Output (video style)
resultsSummarizer(y_test, y_pred_knn)
## output final in loop for different n values (1 to 10)
for n in range(1, 6):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train_scaled, y_train)

    y_pred = knn.predict(X_test_scaled)

    print("\nNo of Neighbors:", n)
    resultsSummarizer(y_test, y_pred, cm_en=False)


# -------- Decision Tree (ALAG SE) --------
print("\n===== Decision Tree =====")

dt = DecisionTreeClassifier(max_depth=14)

dt.fit(X_train_scaled, y_train)

print("Train Accuracy:", dt.score(X_train_scaled, y_train))
print("Test Accuracy:", dt.score(X_test_scaled, y_test))

y_pred_dt = dt.predict(X_test_scaled)

resultsSummarizer(y_test, y_pred_dt)