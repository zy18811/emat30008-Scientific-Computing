import matplotlib.pyplot as plt
import sys

def plot(x,y,ax,format):
    if format == "loglog":
        ax.loglog(x, y)
    elif format == "linear":
        ax.plot(x,y)
    else:
        sys.exit("format: \"%s\" invalid. Please input a valid format." % format)


