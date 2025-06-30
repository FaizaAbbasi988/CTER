import numpy as np
from sklearn import metrics

from sklearn.metrics import mean_squared_error

def RSA(N, T, LB, UB, Dim, F_obj):
    Best_P = np.zeros(Dim)        # Best positions
    Best_F = 0         # Best fitness
    X = initialization(N, Dim, UB, LB)  # Initialize the positions of solutions
    Xnew = np.zeros((N, Dim))
    Conv = np.zeros(T)            # Convergence array

    t = 1                         # Starting iteration
    Alpha = 0.1                   # The best value 0.1
    Beta = 0.005                  # The best value 0.005
    Ffun = np.zeros(N)            # Old fitness values
    Ffun_new = np.zeros(N)        # New fitness values

    for i in range(N):
        Ffun[i] = F_obj(X[i, :])   # Calculate the fitness values of solutions
        if Ffun[i] > Best_F:
            Best_F = Ffun[i]
            Best_P = X[i, :]

    while t < T + 1:              # Main loop - Update the Position of solutions
        ES = 2 * np.random.choice([-1, 1]) * (1 - (t / T))  # Probability Ratio
        for i in range(1, N):
            for j in range(Dim):
                R = Best_P[j] - X[np.random.randint(0, N), j] / (Best_P[j] + np.finfo(float).eps)
                P = Alpha + (X[i, j] - np.mean(X[i, :])) / (Best_P[j] * (UB - LB) + np.finfo(float).eps)
                Eta = Best_P[j] * P
                if t < T / 4:
                    Xnew[i, j] = Best_P[j] - Eta * Beta - R * np.random.rand()
                elif T / 4 <= t < 2 * T / 4:
                    Xnew[i, j] = Best_P[j] * X[np.random.randint(0, N), j] * ES * np.random.rand()
                elif 2 * T / 4 <= t < 3 * T / 4:
                    Xnew[i, j] = Best_P[j] * P * np.random.rand()
                else:
                    Xnew[i, j] = Best_P[j] - Eta * np.finfo(float).eps - R * np.random.rand()

            Flag_UB = Xnew[i, :] > UB  # Check if they exceed (up) the boundaries
            Flag_LB = Xnew[i, :] < LB  # Check if they exceed (down) the boundaries
            Xnew[i, :] = (Xnew[i, :] * ~(Flag_UB + Flag_LB)) + UB * Flag_UB + LB * Flag_LB
            Ffun_new[i] = F_obj(Xnew[i, :])
            if Ffun_new[i] > Ffun[i]:
                X[i, :] = Xnew[i, :]
                Ffun[i] = Ffun_new[i]
            if Ffun[i] > Best_F:
                Best_F = Ffun[i]
                Best_P = X[i, :]

        Conv[t - 1] = Best_F  # Update the convergence curve

        if t % 4 == 0:  # Print the best solution fitness after every 50 iterations
            print(f"At iteration {t}, the best solution fitness is {Best_F}")

        t += 1

    return Best_F, Best_P, Conv


# Step 1: Prepare the training data and testing data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Step 2: Set the relevant parameters of ELM
n_hidden_neurons = 50  # Initial number of hidden neurons
# elm = ELMClassifier(n_hidden=500)
# Example usage:
def F_obj(x):
    elm_model.fit(X_train, y_train)
    y_pred = elm_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    accuracy_elm = metrics.accuracy_score(y_test, (y_pred > 0.5).astype(int))
    print("rmse:", rmse)
    print("ELM Accuracy:", accuracy_elm)
    return accuracy_elm

def initialization(N, Dim, UB, LB):
    # Define your initialization method here
    return np.random.uniform(LB, UB, size=(N, Dim))

# Parameters
N = 20   # Population size
T = 20  # Number of iterations
LB = -1.0  # Lower bound for solution space
UB = 1.0   # Upper bound for solution space
Dim = 10   # Dimensionality of the problem

# Call RSA with your fitness function (F_obj)
Best_F, Best_P, Conv = RSA(N, T, LB, UB, Dim, F_obj)
