import matplotlib.pyplot as plt


def plot(x,y):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y)
    plt.show()
