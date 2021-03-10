import sys

def plotter(x,y,ax,format,label):
    if format == "loglog":
        ax.loglog(x, y,label=label)
    elif format == "linear":
        ax.plot(x,y,label=label)
    else:
        sys.exit("format: \"%s\" invalid. Please input a valid format." % format)


