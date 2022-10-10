import pandas
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import os

filename = "venv/participants_dataset.csv"

# imports the table
# dlearn = pandas.rxead_csv("/lib/participants_dataset.csv")
dlearn = pandas.read_csv(filename)

# replaces the NaN in BMI by the most frequent value, less accurate
# dlearn["bmi"].fillna(dlearn["bmi"].value_counts().index[0], inplace=True)

# replaces the NaN in BMI by the average if the bmi, more accurate
dlearn['bmi'] = dlearn['bmi'].fillna(dlearn['bmi'].mean())

# we also worked on an AI that would replace the missing BMIs based on the other data (AI-ception let's goo)
# the file that we use is the 'corrected' version, but we also left two other methods we used,
#     finding the average and the most frequent value, but it was missing ID's at the end so we coudldn't submit :(

# divides the big table into 2, the one to learn (dlearn) and the one to test (dlearn_submit)
dlearn_submit = dlearn[dlearn['label'].isna()]

# replaces the String by int for each column
dlearn = dlearn.replace({"gender":"Male"},1)
dlearn = dlearn.replace({"gender":"Female"},0)
dlearn = dlearn.replace({"gender":"Other"},2)
dlearn = dlearn.replace({"ever_married":"Yes"},1)
dlearn = dlearn.replace({"ever_married":"No"},0)
dlearn = dlearn.replace({"work_type":"Self-employed"},0)
dlearn = dlearn.replace({"work_type":"Private"},1)
dlearn = dlearn.replace({"work_type":"Govt_job"},2)
dlearn = dlearn.replace({"work_type":"children"},3)
dlearn = dlearn.replace({"work_type":"Never_worked"},3)
dlearn = dlearn.replace({"Residence_type":"Urban"},0)
dlearn = dlearn.replace({"Residence_type":"Rural"},1)
dlearn = dlearn.replace({"smoking_status":"never smoked"},0)
dlearn = dlearn.replace({"smoking_status":"formerly smoked"},1)
dlearn = dlearn.replace({"smoking_status":"smokes"},2)
dlearn = dlearn.replace({"smoking_status":"Unknown"},3)

# creates X variable for both tables
X = dlearn.drop("ID", axis=1)

# creates the X and Y variables for the learning part
X1 = X.loc[(X["label"] == 1)| (X["label"] == 0)]
Y1 = X1["label"]
X1 = X1.iloc[:,:-1]

# implements the classifier, trains de AI and prints the f1 score for funsies
rfc = RandomForestClassifier(max_depth=10, random_state=5)
rfc.fit(X1, Y1)
y_pred = rfc.predict(X1)
print(f1_score(Y1, y_pred, average='macro'))

# creates the X and Y variables for the testing part
X2 = X.loc[(X["label"].isna())]
Y2 = X2["label"]
X2 = X2.iloc[:,:-1]

# does the testing and casts to int
Y2 = rfc.predict(X2)
Y2 = Y2.astype(int)
dlearn_submit['label'] = Y2

# exports the file as a zipzip
compression_opts = dict(method='zip',
                               archive_name='out.csv')
dlearn_submit.to_csv('out.zip', index=False,
              compression=compression_opts)
