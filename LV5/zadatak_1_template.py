import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

#a
y1 = (y_train == 1)
y0 = (y_train == 0)
plt.scatter(X_train[y0,0], X_train[y0,1], cmap='Pastel1', marker='.')
plt.scatter(X_train[y1,0], X_train[y1,1], cmap='hot', marker='.')

y1_ = (y_test == 1)
y0_ = (y_test == 0)
plt.scatter(X_test[y0_,0], X_test[y0_,1], cmap='hsv', marker='x')
plt.scatter(X_test[y1_,0], X_test[y1_,1], cmap='viridis', marker='x')

#b
LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

#c
b = LogRegression_model.intercept_[0]
w1, w2 = LogRegression_model.coef_.T
c = -b/w2
m = -w1/w2

xmin, xmax = -4, 4
ymin, ymax = -4, 4
xd = np.array([xmin, xmax])
yd = m*xd + c
plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.fill_between(xd, yd, ymin, color='tab:orange', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='tab:blue', alpha=0.2)
plt.show()

#d
y_test_p = LogRegression_model.predict(X_test)
cm = confusion_matrix(y_test, y_test_p)
print("Matrica zabune: ", cm)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

print("Tocnost: ", accuracy_score(y_test, y_test_p))
print("Preciznost: ", precision_score(y_test, y_test_p))
print("Recall: ", recall_score(y_test, y_test_p))

#e
y1_ = (y_test == y_test_p)
y0_ = (y_test != y_test_p)
plt.scatter(X_test[y0_,0], X_test[y0_,1], c='black', marker='.')
plt.scatter(X_test[y1_,0], X_test[y1_,1], c='green', marker='.')
plt.show()

