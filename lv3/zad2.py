import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')
data['Make'] = data['Make'].astype('category')
data['Model'] = data['Model'].astype('category')
data['Vehicle Class'] = data['Vehicle Class'].astype('category')
data['Transmission'] = data['Transmission'].astype('category')
data['Fuel Type'] = data['Fuel Type'].astype('category')

# pod a)
plt.figure()
data['Fuel Consumption City (L/100km)'].plot(kind ='hist', bins=20)

# pod b)
#1. nacin
data.plot.scatter(x='Fuel Consumption City (L/100km)',
                  y='CO2 Emissions (g/km)',
                  c='Fuel Type',
                  cmap ="nipy_spectral")

#2. nacin
plt.scatter(x=data['Fuel Consumption City (L/100km)'], y=data['CO2 Emissions (g/km)'])
plt.title('Odnos')
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.show()

# pod c)
data.plot.box(column='Fuel Consumption Hwy (L/100km)', by="Fuel Type")

# pod d)
plt.figure()
plt.title("Counts")
grouped_data = data.groupby("Fuel Type")['Fuel Type'].count()
grouped_data.plot.bar()

plt.bar(prezivjeli_po_spolu.index, prezivjeli_po_spolu.values, color=['yellow', 'green'])
plt.xlabel('Spol')
plt.ylabel('Postotak preživjelih')
plt.title('Postotak preživjelih po spolu')
plt.show()

# pod e)
plt.figure()
plt.title("CO2 prema broju cilindara")
grouped_data = data.groupby("Cylinders")['CO2 Emissions (g/km)'].mean()
grouped_data.plot.bar()


plt.show()
