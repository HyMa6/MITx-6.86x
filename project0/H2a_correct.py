import numpy as np

X1 = np.array([[-4, 2]])
X3 = np.array([[-1, -1]])
X4 = np.array([[2, 2]])

Y1=([1])
Y3=([-1])
Y4=([-1])
w=np.array([0, 0])
b=0

def perceptron_sgd(X, Y, w, b):
    epochs = 1
    for t in range(epochs):
        for i, x in enumerate(X):

            if (np.dot((X[i]+b), w)*Y[i]) <= 0:

                w = w + X[i]*Y[i]
                b=b+Y[i]
    print(w, b)

    return w, b
def main ():
    perceptron_sgd(X3, Y3, w, b)
    perceptron_sgd(X1, Y1, w=[1,1], b=-1)
    perceptron_sgd(X4, Y4, w=[-3, 3], b=0)
    perceptron_sgd(X3, Y3, w=[-5, 1], b=-1)

main()




