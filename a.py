import numpy as np
import sys

m = 4

points = []
y_arr = []
n = 0

with open("points.txt") as f:
    sys.stdin = f
    n = int(f.readline())
    points = np.ndarray((n, 2))
    for i in range(n):
        splt = f.readline().split()
        points[i] = (float(splt[0]), float(splt[1]))


with open("experiments.txt") as f:
    sys.stdin = f
    y_arr = np.array([float(line) for line in f])

y = np.matrix(y_arr).T

X_arr = np.ndarray((n, m))

for i in range(n) :
    X_arr[i][0] = 1
    X_arr[i][1] = points[i][0]
    X_arr[i][2] = points[i][0] * points[i][0]
    X_arr[i][3] = points[i][1] * points[i][1]


X = np.matrix(X_arr)


H = np.linalg.inv(X.T * X) * X.T
P = np.identity(n) - X * H

theta = H * y
print(theta)
