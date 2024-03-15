import matplotlib.pyplot as plt
import numpy as np


'''x = np.array([1,2,3,3,1])
y= np.array([1,2,2,1,1])
plt . plot (x , y , 'b', linewidth =1 , marker =".", markersize =10 )
plt . axis ([0 ,4 ,0 , 4])
plt . xlabel ('x os')
plt . ylabel ('y os')
plt . title (  'Primjer')
plt . show ()'''

a=np.loadtxt('data.csv', skiprows=1, delimiter=',')
print(len(a))
plt . figure ()
visina=np.array(a[:,1])
tezina=np.array(a[:,2])
plt.scatter(visina, tezina)


plt.figure ()
visina1=np.array(a[:,1:50])
tezina1=np.array(a[:,2:50])
plt.scatter(visina, tezina)

plt.show()


