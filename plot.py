import matplotlib.pyplot as plt


def plot(x,y,ax,format):
    if format == "loglog":
        ax.loglog(x, y)
    else:
        ax.plot(x,y)

