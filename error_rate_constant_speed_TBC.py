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

    nodes = [32, 64, 128, 256, 512, 1024]
    #error_norms = [0.049092564556246825, 0.007140174141234464, 0.0009230413460485456, 0.0001161118132243963, 1.4520701499859056e-05, 1.8149584754760637e-06]
    
    error_norms = [0.07983136011284686, 0.014362169768465667, 0.0017808785402766666, 0.00022096227080161318, 2.7574119437930348e-05, 3.4446326258257683e-06]
    p = error_rate(nodes, error_norms)

    print('Error rate = {:6f} '.format(p))

    plt.loglog(nodes, error_norms, 'r-o')
    plt.xlabel('log(Nx)')
    plt.ylabel('log(error_norms)')
    plt.grid()
    plt.show()