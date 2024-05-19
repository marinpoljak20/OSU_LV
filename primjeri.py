import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Učitavanje Iris dataset-a
iris = load_iris()
X = iris.data
y = iris.target

# a) Prikaži odnos duljine latice i čašice za Virginicu (zelenom bojom) i Setosu (sivom bojom)
virginica = X[y == 2]
setosa = X[y == 0]

plt.scatter(virginica[:, 0], virginica[:, 1], c='green', label='Virginica')
plt.scatter(setosa[:, 0], setosa[:, 1], c='grey', label='Setosa')

plt.xlabel('Duljina latice')
plt.ylabel('Širina čašice')
plt.title('Odnošenje duljine latice i širine čašice za Virginicu i Setosu')
plt.legend()

plt.show()

# b) Prikaži najveću vrijednost širine čašice za sve tri klase cvijeta
max_sepal_width = max(X[:, 1])
labels = iris.target_names

plt.bar(labels, [max_sepal_width] * len(labels), color=['blue', 'orange', 'green'])

plt.xlabel('Klasa cvijeta')
plt.ylabel('Najveća vrijednost širine čašice')
plt.title('Najveća vrijednost širine čašice za sve tri klase cvijeta')

plt.show()

# c) Izračunaj koliko jedinki klase Setosa ima veću širinu čašice od prosječne širine čašice te klase
setosa_width = X[y == 0][:, 1]
mean_setosa_width = setosa_width.mean()
num_greater_than_mean = sum(setosa_width > mean_setosa_width)

print("Broj jedinki klase Setosa s većom širinom čašice od prosječne širine čašice te klase:", num_greater_than_mean)