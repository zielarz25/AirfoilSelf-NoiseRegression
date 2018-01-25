
import pandas as pd
import numpy as np


#wczytanie danych z pliku
csv = pd.read_csv('airfoil_self_noise.dat',sep='\t').as_matrix()


# dane: 5 atrybutow o indeksach 0-4
# data = csv[:,2:60].astype(float)
data = csv[:,0:5].astype(float)
#data = data[:,np.newaxis]
#labelem jest ostatnia kolumna (5)
target = csv[:,5].astype(float)                     #wektor
target = target[:,np.newaxis]                       #macierz
print (data[0,:])
print (target[0,:])