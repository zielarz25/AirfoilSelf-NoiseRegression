from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# wczytanie danych z pliku
csv = pd.read_csv('airfoil_self_noise.dat',sep='\t').as_matrix()

# dane: 5 atrybutow o indeksach 0-4
feature_number = 4                                  # numer cechy do wyswietlenia na wykresie
data = csv[:,feature_number].astype(float)
data = data[:,np.newaxis]
#labelem jest ostatnia kolumna (5)
target = csv[:,5].astype(float)                     # wektor
target = target[:,np.newaxis]                       # macierz

print("********************************")
print("1 cecha:")

# standaryzacja danych
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

# podział na zbiór uczący i testujacy
from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = \
    train_test_split(data, target, test_size=0.3)


# utworzenie obiektu regresji
regr = linear_model.LinearRegression()
# wytreniowanie regresji na podstawie danych trenujacych
regr.fit(data_train, target_train)
# zwrocenie przewidywanych wynikow na podstawie zbioru testujacego
pred = regr.predict(data_test)


# print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(target_test, regr.predict(data_test)))
print('r2: %.2f' % r2_score(target_test, regr.predict(data_test)))

plt.figure(1)
plt.subplot('221')
plt.scatter(data_train,target_train)
plt.title('Training dataset')
plt.xlabel('x{}'.format(feature_number))
plt.ylabel('target')
plt.grid()

plt.subplot('222')
plt.scatter(data_test,target_test)
plt.title('Testing dataset')
plt.xlabel('x{}'.format(feature_number))
plt.ylabel('target')
plt.grid()

plt.subplot('223')
plt.scatter(data_train, target_train)
plt.plot(data_train, regr.predict(data_train), color='green',
         linewidth=3)
plt.title('Learned linear regression')
plt.xlabel('x{}'.format(feature_number))
plt.ylabel('target')
plt.tight_layout()
plt.grid()

plt.subplot('224')
plt.scatter(data_test, target_test)
plt.plot(data_test, regr.predict(data_test), color='green', linewidth=3)
plt.title('Learned linear regression')
plt.xlabel('x{}'.format(feature_number))
plt.ylabel('target')
plt.tight_layout()
plt.grid()


print ("******************************")
print ("Wszystkie cechy")

data = csv[:,0:5].astype(float)

target = csv[:,5].astype(float)                     # wektor
target = target[:,np.newaxis]                       # macierz


from sklearn.preprocessing import PolynomialFeatures
polyFeat= PolynomialFeatures(degree=6)
polyFeat= polyFeat.fit(data, target)
data= polyFeat.transform(data)


scaler.fit(data)
data = scaler.transform(data)

data_train, data_test, target_train, target_test = \
    train_test_split(data, target, test_size=0.3)
regr.fit(data_train, target_train)
pred = regr.predict(data_test)

# print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(target_test, pred))
print('r2: %.2f' % r2_score(target_test, pred))

# from sklearn.model_selection import cross_val_score
# cross_val= cross_val_score(regr, data_test, target_test, cv=8)
# print ("Cross val score: {}".format(cross_val))
plt.show()
