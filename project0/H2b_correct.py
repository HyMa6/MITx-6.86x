import numpy as np

X = np.array([
    [-4, 2],
    [-2, 1],
    [-1,-1],
    [2,2],
    [1,-2]
])

Y = np.array([[1], [1], [-1], [-1], [-1]])


def perceptron_sgd(X, Y):
    #w = np.random.randint(10, size=2)
    w=np.array([-2,3])
    print("w=", w)
    epochs = 10
    b=-3
    for t in range(epochs):
        for i, x in enumerate(X):
            if (np.dot((X[i]+b), w)*Y[i]) <= 0:
                w = w + X[i]*Y[i]
                b=b +Y[i]

        print("w=", w)
        print("b=", b)

perceptron_sgd(X, Y)




