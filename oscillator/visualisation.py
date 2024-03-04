import matplotlib.pyplot as plt

def plot_danino(t, sol):
    """
    Plot the solutions of the delayed differential equations.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, sol[:, 0], 'r', label='A', linewidth=2)
    plt.plot(t, sol[:, 1], 'g', label='I', linewidth=2)
    plt.plot(t, sol[:, 2], 'b', label='Hi', linewidth=2)
    plt.plot(t, sol[:, 3], 'k', label='He', linewidth=2)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('Dynamics of A, I, Hi, and He')
    plt.grid(True)
    plt.show()