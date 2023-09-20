import scipy.stats as sps
import numpy as np
import pandas as pd
import itertools
import xlsxwriter

O = [1, 2, 0.05, -0.5]
m = 4
ro = 0.15
levels_x1 = np.array([-1, -0.5, 0.5, 1])
levels_x2 = np.array([-1, -0.5, 0, 0.5, 1])


n = levels_x1.size*levels_x2.size

# входные данные
x = np.array([[levels_x1[i], levels_x2[j]] for i, j in 
              itertools.product(range(levels_x1.size), range(levels_x2.size))])



# истиные значенияя отклика
u = np.array([O[0] + O[1] * x[i][0] + O[2] * x[i][0] ** 2 + O[3] * x[i][1] ** 2
             for i in range(n)])



# среднее истинных значений отклика
u_mean = np.mean(u)
omega_sqr = ((u - u_mean) @ (u - u_mean)) / (n - 1)





# мощность сигнала
#omega_sqr = np.cov(u)
#print((((u-u_mean)@(u-u_mean)) / (n - 1)) * ro)

# дисперсия
dispersion = ro*omega_sqr

# помехи
e = sps.norm(0, dispersion).rvs(size=n)  # генерируем выборки размера n из нормального распределения

# отклик с помехами
y = np.array([u[i] + e[i] for i in range(n)])

X = np.array([[1, x[i][0], x[i][0] ** 2, x[i][1] ** 2]
             for i in range(n)])


# оценка параметров = H * y
H = np.linalg.inv(X.T @ X) @ X.T

# оценка параметров
O_estimated = H @ y
print("Оценка параметров: ", end="")
print(O_estimated)


# оценка модели
y_estimated = X @ O_estimated.T

# ошибка
y_error = y - y_estimated

# оценка дисперсии
dispersion_estimated = (y_error @ y_error) / (n - m)

print("Дисперсия: %f\nОценка дисперсии: %f" % 
      (dispersion, dispersion_estimated))

# F-тест
F = dispersion_estimated / dispersion
FT = sps.f.isf(0.05,n-m,1e+10)

print("F: %f\nFT: %f" % (F, FT))
if (F < FT) :
    print("Модель адекватна")
else:
    print("Модель неадекватна")




result = pd.DataFrame({'x1': np.array([i[0] for i in x]), 'x2': np.array([i[1] for i in x]), 
                        'u': u, 'y' : y, 'y estimated': y_estimated, 'y error' : y_error})



# вывод в эксель
writer = pd.ExcelWriter('result.xlsx', engine='xlsxwriter')
result.to_excel(writer, sheet_name='welcome', index=False)
writer.close()