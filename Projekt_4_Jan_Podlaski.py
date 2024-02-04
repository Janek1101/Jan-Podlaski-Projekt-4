import numpy as np 
import matplotlib.pyplot as plt

def f(x: list, gamma, beta, sigma):
    S = x[0]
    E = x[1]
    I = x[2]

    ds = -(beta * I * S)
    de = (beta * I * S) - sigma * E
    di = sigma * E - gamma * I
    dr = gamma * I

    return np.array([ds, de, di, dr])

def Runge_Kutta_4(x, dt, gamma, sigma, beta):
    f1 = dt * f(x, gamma, beta, sigma)
    f2 = dt * f(x + f1 / 2., gamma, beta, sigma)
    f3 = dt * f(x + f2 / 2., gamma, beta, sigma)
    f4 = dt * f(x + f3, gamma, beta, sigma)

    next_x = x + 1/6 * (f1 + 2 * f2 + 2 * f3 + f4)
    return next_x

def zadanie_1(gamma, sigma, beta):
    t0 = 0
    te = 50
    dt = .01
    time = np.arange(t0, te, dt)

    x = np.zeros([time.shape[0], 4])
    x[0, 0] = 0.99
    x[0, 1] = 0.01
    x[0, 2] = 0.
    x[0, 3] = 0.

    for i in range(time.shape[0] - 1):
        x[i+1] = Runge_Kutta_4(x[i], dt, gamma, sigma, beta)

    t = time.shape[0]
    S = [x[j, 0] for j in range(t)]
    E = [x[j, 1] for j in range(t)]
    I = [x[j, 2] for j in range(t)]
    R = [x[j, 3] for j in range(t)]

    plt.plot(time, S, label='Podatny')
    plt.plot(time, E, label='Narażony')
    plt.plot(time, I, label='Zakaźny')
    plt.plot(time, R, label='Oddalony')
    plt.xlabel('Czas[t]')
    plt.ylabel('Znormalizowana populacja')
    plt.legend()
    plt.show()

def zadanie_2(gamma, sigma, beta):
    t0 = 0
    te = 50
    dt = .01
    time = np.arange(t0, te, dt)

    x = np.zeros([time.shape[0], 4])
    x[0, 0] = 0.99
    x[0, 1] = 0.01
    x[0, 2] = 0.
    x[0, 3] = 0.

    for i in range(time.shape[0] - 1):
        x[i+1] = Runge_Kutta_4(x[i], dt, gamma, sigma, beta)

    t = time.shape[0]
    S = [x[j, 0] for j in range(t)]
    E = [x[j, 1] for j in range(t)]
    I = [x[j, 2] for j in range(t)]
    R = [x[j, 3] for j in range(t)]

    plt.plot(time, S, label='Podatny')
    plt.plot(time, E, label='Narażony')
    plt.plot(time, I, label='Zakaźny')
    plt.plot(time, R, label='Oddalony')
    plt.xlabel('Znormalizowana populacja')
    plt.ylabel('Czas[t]')
    plt.legend()
    plt.show()

def zadanie_3(gamma, sigma, beta):
    t0 = 0
    te = 50
    dt = .01
    time = np.arange(t0, te, dt)

    x = np.zeros([time.shape[0], 4])
    x[0, 0] = 0.99
    x[0, 1] = 0.01
    x[0, 2] = 0.
    x[0, 3] = 0.

    for i in range(time.shape[0] - 1):
        x[i+1] = Runge_Kutta_4(x[i], dt, gamma, sigma, beta)

    t = time.shape[0]
    S = [x[j, 0] for j in range(t)]
    E = [x[j, 1] for j in range(t)]
    I = [x[j, 2] for j in range(t)]
    R = [x[j, 3] for j in range(t)]

    print(f'R0 = {(beta/gamma) * x[0, 0]}')

    plt.plot(time, S, label='Podatny')
    plt.plot(time, E, label='Narażony')
    plt.plot(time, I, label='Zakaźny')
    plt.plot(time, R, label='Oddalony')
    plt.xlabel('Znormalizowana populacja')
    plt.ylabel('Czas[t]')
    plt.legend()
    plt.show()

def main():
    zadanie_1(gamma = 0.1, sigma = 1, beta = 1)
    zadanie_2(gamma = 0.1, sigma = 1, beta = 0.5)

    # R0 > 1
    zadanie_3(gamma = 0.7, sigma = 1, beta = 3)

    # R0 < 1
    zadanie_3(gamma = 3, sigma = 1, beta = 0.3)

if __name__ == '__main__':
    main()