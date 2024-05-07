import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("LV2\data.csv", delimiter=',', names=True)

#a
print("Mjerenja izvrsena na", len(data), "osoba.")

#b
plt.scatter(data["Height"], data["Weight"], marker=".", linewidths=0.1)
plt.xlabel("Visina [cm]")
plt.ylabel("Masa [kg]")
plt.title("Odnos visine i mase")
plt.show()

#c
plt.scatter(data["Height"][0::50], data["Weight"][0::50], marker=".", linewidths=0.1)
plt.xlabel("Visina [cm]")
plt.ylabel("Masa [kg]")
plt.title("Odnos visine i mase")
plt.show()

#d
print("Min:", np.min(data["Height"]))
print("Max:", np.max(data["Height"]))
print("Mean:", np.mean(data["Height"]))

#e
ind_m = (data[:,0] == 1)
ind_z = (data[:,0] == 0)
print("Muskarci:")
print("Min:", np.min(data["Height"], where=ind_m))
print("Max:", np.max(data["Height"], where=ind_m))
print("Mean:", np.mean(data["Height"], where=ind_m))
print("Zene:")
print("Min:", np.min(data["Height"][ind_z]))
print("Max:", np.max(data["Height"][ind_z]))
print("Mean:", np.mean(data["Height"][ind_z]))
