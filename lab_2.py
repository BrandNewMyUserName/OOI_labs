import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
np.set_printoptions(precision=2, suppress=True)

def function(x1, x2, x3)->np.dtype('float64'):
    return np.sin(x1) + np.sin(x2) - np.cos(x3)


def sigmoid(x) ->np.dtype('float64'):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x) ->np.dtype('float64'):
    return x*(1-x)

def normalization(X):
    scaler_x = MinMaxScaler()

    X_normalize = np.array(scaler_x.fit_transform(X.reshape(-1, 1)).reshape(X.shape))

    return X_normalize

def train_validation_split(X_values, train_size):
    np.random.shuffle(X_values)
    X_values = normalization(X_values)
    Y_values = np.array([function(x, y, z) for x, y, z in X_values])

    X_train = X_values[:train_size]
    X_valid = X_values[train_size:]

    Y_train = Y_values[:train_size]
    Y_valid = Y_values[train_size:]

    return X_train, X_valid, Y_train, Y_valid

def forward_move(W_1, W_2, X, average):
    # Обчислення нейронів прихованого шару
    S_1 = np.array(np.dot(W_1, X))
    y_1 = np.array(sigmoid(S_1))

    # Обчислення нейронів вихідного шару
    S_2 = np.array(np.dot(y_1, W_2))
    y_2 = np.array(sigmoid(S_2))

    d = np.zeros(2)
    d[0] = y_2[0]
    d[1] = 1 if y_2[0] > average else 0

    return y_1, y_2, d

def backpropagation(X, Y, neurons_in_hiden_layer, n):
    print("Навчання нейромережі:")
    W_1 = np.random.rand(neurons_in_hiden_layer, 3)
    W_2 = np.random.rand(neurons_in_hiden_layer, 2)
    num_of_image = list(range(0, len(X)))
    average = np.mean(Y)
    epoch = 0
    eps = 1
    threshold = 0.015

    for epoch in range(1000):
        eps = 0
        epoch +=1
        np.random.shuffle(num_of_image)
        for i in range(len(X)):
            x = X[num_of_image[i]]
            y = Y[num_of_image[i]]

            # Прямий хід
            y_1, y_2, d = forward_move(W_1, W_2, x, average)
            y_rounded = 1 if y > average else 0

            # Зворотній хід

            # Корекція вагових коефіцієнтів 2 шару
            delta_2 = np.array([(d[0] - y) * d_sigmoid(y_2[0]), (d[1] - y_rounded ) * d_sigmoid(y_2[1])])

            for t in range(len(W_2)):
                for j in range(len(W_2[t])):
                    W_2[t][j] += -n*delta_2[j]*y_1[t]

            # Корекція вагових коефіцієнтів 1 шару
            delta_1 = np.dot(W_2, delta_2) * d_sigmoid(y_1)

            for t in range(len(W_1)):
                for j in range(len(W_1[t])):
                    W_1[t][j] += -n*delta_1[t]*x[j]

            eps +=(d[0] - y)**2

        eps/=len(X)
        if(eps < threshold):
            break

    print(f"Загальна кількість ітерацій: {epoch}\nПохибка на останній ітерації: {eps:.5f}\n\n")
    return W_1, W_2

def validation(X, Y, W_1, W_2, neurons_in_hiden_layer, output):
    if output == True:
        print("Контрольна вибірка:")
        print(f"Кількість нейронів у прихованому шарі: {neurons_in_hiden_layer} | Розмір тестувальної вибірки: {len(X)}")

        print("=======================================================")
        print("  X1\t\t X2\t\t\t X3\t\t\t d1\t\t\t d2")
        print("=======================================================")


    total_error = error = 0
    average = np.mean(Y)

    y_1 = np.zeros(neurons_in_hiden_layer)
    y_2 = np.zeros(2)

    for k in range(len(X)):
        #Прямий хід
        y_1, y_2, d = forward_move(W_1, W_2, X[k], average)

        d[1] = 1 if y_2[0] > average else 0
        d[0] = y_2[0]
        error = (d[0] - Y[k])**2
        total_error += error
        if output == True:
            print(f"  {'%.5f'%X[k][0]}\t {'%.5f'%X[k][1]}\t {'%.5f'%X[k][2]}\t {'%.5f'%d[0]}\t {d[1]}\n")
            print("-------------------------------------------------------")


    total_error /= len(X)
    if output == True:
        print(f"Помилка : {total_error}")


if __name__ == '__main__':
    # Кількість наборів
    size = 30
    # Кількість нейронів у прихованому шарі
    neurons_in_hiden_layer = 12
    # Кількість наборів, які подаються на вхід для навчання та валідації
    train_set_size = 26
    # Коефіцієнт навчання
    learning_rate = 0.1

    # Ініціалізація значень X для кожного образу
    X = np.zeros((size, 3))
    X[0] = [8, 5, 3]
    #X[0] = [1, 2, 3]
    variable = 1
    for i in range(1, size):
        if i % 7 == 1:
            X[i] = X[0] + np.array([variable, 0, 0])
        elif i % 7 == 2:
            X[i] = X[0] + np.array([0, variable, 0])
        elif i % 7 == 3:
            X[i] = X[0] + np.array([0, 0, variable])
        elif i % 7 == 4:
            X[i] = X[0] + np.array([-variable, 0, 0])
        elif i % 7 == 5:
            X[i] = X[0] + np.array([0, -variable, 0])
        elif i % 7 == 6:
            X[i] = X[0] + np.array([0, 0, -variable])
        else:
            X[i] = X[0] + np.array([-variable, 0, -variable])
            variable += 1
            i-=1


    X_train, X_valid, Y_train, Y_valid = train_validation_split(X, train_set_size)
    W_1, W_2 = backpropagation(X_train, Y_train, neurons_in_hiden_layer, learning_rate)
    validation(X_valid, Y_valid, W_1, W_2, neurons_in_hiden_layer, True)

