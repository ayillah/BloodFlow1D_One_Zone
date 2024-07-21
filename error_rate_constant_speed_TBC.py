import numpy as np
import matplotlib.pyplot as plt

def error_rate(Nx, error):
    """Compute the error rate."""

    # Compute the logs of the nodes and the errors
    logNx = np.log(np.array(Nx))
    logError = np.log(np.array(error))
    ones = np.ones(len(logNx))

    V = np.array([ones, logNx]).transpose()

    # Solve least squares system
    A = np.matmul(V.transpose(), V)
    b = np.matmul(V.transpose(), logError)

    c = np.linalg.solve(A, b)

    return c[1]


if __name__ == '__main__':

    nodes = [32, 64, 128, 256, 512]
    error_norms = [0.049092564556246825, 0.007140174141234464, 0.0009230413460485456, 0.0001161118132243963, 1.4520701499859056e-05]

    #nodes = [61, 121, 181, 241, 301]
    #error_norms = [0.008316380427135036, 0.0010939410751682453, 0.00032782118354812706, 0.0001390640619430164, 7.144655009769631e-05]

    p = error_rate(nodes, error_norms)

    print('Error rate = {:6f} '.format(p))

    plt.loglog(nodes, error_norms, 'r-o')
    plt.xlabel('log(Nx)')
    plt.ylabel('log(error_norms)')
    plt.grid()
    plt.show()