import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

from utilities import *
from tqdm import tqdm



X_train, y_train, X_test, y_test = load_data_csv()

# print(y_train)
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='summer')
# plt.show()

# for value in y_train:
#     if value == 1:
#         print(value)


""" --- Initialisation --- """
def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)


""" --- Model --- """
def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A


""" --- Cost --- """
def log_loss(A, y):
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))


""" --- Gradient --- """
def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)


""" --- Update --- """
def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)


""" --- Predict --- """
def predict(X, W, b):
    A = model(X, W, b)
    return A >= 0.5


""" --------------------- ASSEMBLAGE FINAL --------------------- """
def artificial_neuron(X_train, y_train, X_test, y_test, learning_rate=0.01, n_iter=100):
    # initialisation W, b
    W, b = initialisation(X_train)

    # Stocker les valeurs du coût et de la présicion pour le dataset de train
    train_loss = []
    train_acc = []

    # Stocker les valeurs du coût et de la présicion pour le dataset de test
    test_loss = []
    test_acc = []

    for i in tqdm(range(n_iter)):
        # Fonction d'activation
        A = model(X_train, W, b)

        if i % 10 == 0:
            # -- Train
            # Ajout de l'évolution du loss dans notre list
            train_loss.append(log_loss(A, y_train))
            # Calculer les prédictions de notre matrice X_train avec les paramètres W et b à l'instant T
            y_pred = predict(X_train, W, b)
            # Ajout de l'évolution de la précision des prédictions du modèle par rapport au vraies valeurs d'output
            train_acc.append(accuracy_score(y_train, y_pred))

            # -- Test
            A_test = model(X_test, W, b)
            test_loss.append(log_loss(A_test, y_test))
            y_pred = predict(X_test, W, b)
            test_acc.append(accuracy_score(y_test, y_pred))

        # mise a jour
        dW, db = gradients(A, X_train, y_train)
        W, b = update(dW, db, W, b, learning_rate)

    # -- Affichage des différents graphes
    plt.figure(figsize=(12, 4))
    # Graphe loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss')
    plt.plot(test_loss, label='test loss')
    plt.legend()
    # Graphe accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train acc')
    plt.plot(test_acc, label='test acc')
    plt.legend()

    plt.show()

    return W, b



# W, b = artificial_neuron(X_train, y_train, X_test, y_test, learning_rate=0.01, n_iter=10000)
# print(W, b)

# -- Good Model -> valeurs paramètres W et biais b obtenus lorsque nous avons de bonnes courbes d'apprentissage (loss qui diminue et accuracy qui augmente)
W_good_model = np.array([
    -0.52727749,
    -0.7948972,
    -0.00961989,
    0.02243971,
    0.35557086,
    0.31093344
])

b_good_model = np.array([-0.35174418])

# -- Test with personalised data
# Suposed not injury -> suposed return False
nb_session = 3
nb_rest_day = 3
total_kms = 15
max_km_one_day = 7
nb_hard_session = 0
nb_strengh_training = 2

# Suposed injury  -> suposed return True
# nb_session = 1.0
# nb_rest_day = 0.0
# total_kms = 50.0
# max_km_one_day = 50.0
# nb_hard_session = 1.0
# nb_strengh_training = 0.0

my_data = np.array([
    nb_session,
    nb_rest_day,
    total_kms,
    max_km_one_day,
    nb_hard_session,
    nb_strengh_training
])

print(predict(my_data, W_good_model, b_good_model))
