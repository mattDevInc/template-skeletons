import numpy as np
import plotly.express as plt

# https://www.arnoldfw.com/pdf/3d_curves.pdf

# some of those might need swapping around
# for t [0, 6pi]
def make_spiral (t, a) :
    x = 4 * np.sin(t)
    y = 4 * np.cos(t)
    z = a ** (- t/10)
    return np.column_stack([x, y, z])

def make_wrapped_circle (t) :
    x = 2 * np.cos(t)
    y = 2 * np.sin(t)
    z = np.cos(2 * t)
    return np.column_stack([x, y, z])

def make_curve (t) :
    x = 2 * t**3 - 3 * t
    y = t**2 + 4 * t
    z = -t**3 + 2 * t**2 - t
    return np.column_stack([x, y, z])

def main () :
    ts = np.linspace(0, 6 * np.pi, 100)
    pts = make_spiral(ts, 10)
    x_ = [p[0] for p in pts]
    y_ = [p[1] for p in pts]
    z_ = [p[2] for p in pts]
    

    fig = plt.line_3d(x = x_, y = y_, z = z_, title = "squiggle")
    #fig.show()

    tss = np.linspace(0, 20, 100)
    ptss = make_curve(tss)
    xx_ = [p[0] for p in ptss]
    yy_ = [p[1] for p in ptss]
    zz_ = [p[2] for p in ptss]
    figg = plt.line_3d(x = xx_, y = yy_, z = zz_, title = "wave")
    figg.show()
if __name__ == "__main__" :
    main()
