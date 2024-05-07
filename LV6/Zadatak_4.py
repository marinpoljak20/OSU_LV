import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import train_test_split

# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 1.1, 1.5],
              'gamma': [0.001, 0.01, 0.1, 1, 1.1, 1.5]}

SVM_model = svm.SVC(kernel='rbf')

svm_gscv = GridSearchCV(SVM_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

svm_gscv.fit(X_train_n, y_train)

print(svm_gscv.best_params_)