import inspect
from functools import reduce
from collections import Sequence

import numpy as np
from scipy.odr import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def _validate(fun, eq_classes=({'x', 'y', 'err_x', 'err_y'})):
    """ Decorator that ensures that all arguments in the same equivalence class have the same length. """

    def wrapper(*args, **kwargs):

        arg_names = inspect.getfullargspec(fun)[0]
        for k, v in zip(arg_names, args):
            kwargs[k] = v

        def reducer(arg_a, arg_b):
            a, b = kwargs[arg_a], kwargs[arg_b]
            if a is not None and b is not None and len(a) != len(b):
                raise ValueError("{} must be of same length as {}.".format(arg_a, arg_b))
            return arg_b if b is not None else arg_a

        for cls in eq_classes:
            cls = filter(lambda arg: arg in kwargs, cls)
            try:
                reduce(reducer, cls)
            except TypeError:
                pass

        return fun(**kwargs)

    return wrapper

def _general_fit(x, y, funcs, err_y=None):
    """ Fits the data set (x and y) to a linear combination of the input functions, returning the parameters. """

    n = len(x)

    # Handle uncertainties
    if err_y is None:
        err_y = n * [1]
    else:
        err_y = [1 / err_i**2 for err_i in err_y]

    # Weight matrix
    # noinspection PyUnusedLocal
    W = [n * [0] for i in range(0, n)]
    for i in range(0, n):
        W[i][i] = err_y[i]
    W = np.matrix(W)

    # Obtain D from the x-values and the functions
    D = [[fun(x_i) for fun in funcs] for x_i in x]
    D = np.matrix(D)
    # Gets the pseudoinverse
    D = (D.T * W * D).I * D.T
    # Gets the parameters and returns the list
    res = D * W * np.matrix(y).T
    return res.flatten().tolist()[0]

@_validate
def ls_regression(x, y, funcs, err_y=None, p0=None):
    """
    Returns the results of a scipy curve fit for the input data set and functions.

    :param x: The sequence of x values for the data set.
    :param y: The sequence of y values for the data set.
    :param funcs: The single function or sequence of functions to determine parameters for. If a sequence of functions
        is input, the fit will be to a linear combination, with the coefficients the parameters. Otherwise, the function
        is assumed to be of the form f(x, *params).
    :param err_y: The errors in the y values.
    :param p0: An estimation of the parameters. Must be input if funcs is not a sequence.
    """

    if isinstance(funcs, Sequence):
        if p0 is None:
            p0 = _general_fit(x, y, funcs, err_y)
        funcs = comb_gen(*funcs)
    elif p0 is None:
        raise ValueError("p0 must be specified if funcs is not a sequence.")

    res = curve_fit(funcs, x, y, p0, err_y)
    return res[0], np.sqrt(np.diag(res[1]))

@_validate
def od_regression(x, y, funcs, err_x, err_y, p0=None):
    """
    Returns the results of an orthogonal distance regression performed for the input function.

    :param x: The sequence of x values for the data set.
    :param y: The sequence of y values for the data set.
    :param funcs: The single function or sequence of functions to determine parameters for. If a sequence of functions
        is input, the fit will be to a linear combination, with the coefficients the parameters.
    :param err_x: The errors in the x values.
    :param err_y: The errors in the y values.
    :param p0: An estimation of the parameters. Must be input if funcs is not a sequence.
    """

    # Setup
    data = RealData(x, y, sx=err_x, sy=err_y)
    if isinstance(funcs, Sequence):
        if p0 is None:
            p0 = _general_fit(x, y, funcs, err_y)
        funcs = swap_args(comb_gen(*funcs))
    elif p0 is None:
        raise ValueError("p0 must be specified if funcs is not a sequence.")
    model = Model(funcs)

    # Regression
    odr = ODR(data, model, beta0=p0)
    output = odr.run()
    return output.beta, output.sd_beta

@_validate
def lorentzian_fit(x, y, err_x=None, err_y=None):
    """
    Fits a Lorentzian distribution to the input data.

    :param x: The sequence of x values for the data set.
    :param y: The sequence of y values for the data set.
    :param err_x: The errors in the x values.
    :param err_y: The errors in the y values.
    :return: A two dimensional array, containing first the values of the parameters and then the corresponding errors.
        The parameters are amplitude, width (gamma) and location (x_0) respectively. Note that amplitude is not the
        peak value of the distribution, which is amplitude / (pi * gamma). Also note that the output width parameter is
        the half width at half maximum, not the full width at half maximum.
    """

    # Initializes components needed for estimation
    n = len(x)
    epsilon_n = 8 / np.sqrt(n) + 460 / n**2
    weights = [np.cos(np.pi * ((i - 0.5) / n - 0.5))**(2 + epsilon_n) * np.cos(2 * np.pi * ((i - 0.5) / n - 0.5))
               for i in range(1, n + 1)]
    c = sum(weights)
    x_s = sorted(range(0, n), key=lambda i: x[i])

    max_i = max(x_s, key=lambda i: y[x_s[i]])
    y_max = y[x_s[max_i]]
    half_max = y_max / 2

    # Location parameter estimate
    x_0 = 0
    for i in range(0, n):
        x_0 += weights[i] / c * x[x_s[i]]

    # Width parameter estimate
    q_1 = (n - 1) * 0.25
    q_3 = (n - 1) * 0.75
    q_1 = int(q_1 + 1)
    q_3 = int(q_3)

    get_y, get_x = lambda i: y[x_s[i]], lambda i: x[x_s[i]]
    # == in this instance is effectively XNOR
    direction = lambda loc: 1 if (get_y(loc) < half_max) == (loc < max_i) else -1

    def find_x(loc):
        """ Finds the approximate x-value that yields half-max. Slow, but it works on small datasets. """

        direct = init = get_y(loc) < half_max

        while (get_y(loc) < half_max) == init and 0 < loc < n - 1:
            direct = direction(loc)
            loc += direct

        m = (get_y(loc) - get_y(loc - direct)) / (get_x(loc) - get_x(loc - direct))
        b = get_y(loc) - m * get_x(loc)
        return (half_max - b) / m

    x_1 = find_x(q_1)
    x_3 = find_x(q_3)
    gamma = (x_3 - x_1) / 2

    # Amplitude estimate
    A = y_max * np.pi * ((x[x_s[max_i]] - x_0)**2 + gamma**2) / gamma

    p0 = A, gamma, x_0

    # Return the correct fit based on the input errors
    if err_x is None:
        return ls_regression(x, y, lorentzian, err_y, p0)
    return od_regression(x, y, swap_args(lorentzian), err_x, err_y, p0)

@_validate
def gaussian_fit(x, y, err_x=None, err_y=None):
    """
    Fits a Gaussian distribution to the input data.

    :param x: The sequence of x values for the data set.
    :param y: The sequence of y values for the data set.
    :param err_x: The errors in the x values.
    :param err_y: The errors in the y values.
    :return: A two dimensional array, containing first the values of the parameters and then the corresponding errors.
        The parameters are amplitude, variance and mean (mu) respectively.
    """

    mu = np.mean(x)
    var = np.var(x)
    A = max(y) * np.sqrt(2 * np.pi * var)

    p0 = A, var, mu

    # Return the correct fit based on the input errors
    if err_x is None:
        return ls_regression(x, y, gaussian, err_y, p0)
    return od_regression(x, y, swap_args(gaussian), err_x, err_y, p0)

@_validate
def poly_fit(x, y, err_x=None, err_y=None, n=1, filter_threshold=None):
    """
    Fits an nth order polynomial to the input data.

    :param x: The sequence of x values for the data set.
    :param y: The sequence of y values for the data set.
    :param err_x: The errors in the x values.
    :param err_y: The errors in the y values.
    :param n: The order of the polynomial.
    :return: A two dimensional array, containing first the values of the parameters and then the corresponding errors.
        The parameters are the coefficients of the terms in the polynomial, in descending order from 1 to x^n.
    """

    funcs = [power(i) for i in range(0, n + 1)]
    if err_x is None:
        if err_y is not None:
            err_y = [1 / e for e in err_y]
        res = np.polyfit(x, y, n, w=err_y, cov=True)
        return res[0][::-1], np.sqrt(np.diag(res[1]))[::-1]
    else:
        return od_regression(x, y, funcs, err_x, err_y)

@_validate
def lin_fit_252(x, y, err_y=None):
    """
    Returns a linear fit for the input data set as a tuple of tuples: the parameters and then their errors.
    Performed PHY 252 style.
    """

    if err_y is None:
        err_y = len(x) * [1]

    A = sum(x_i / sig_i**2 for x_i, sig_i in zip(x, err_y))
    B = sum(1 / sig_i**2 for sig_i in err_y)
    C = sum(y_i / sig_i**2 for y_i, sig_i in zip(y, err_y))
    D = sum((x_i / sig_i)**2 for x_i, sig_i in zip(x, err_y))
    E = sum(y_i * x_i / sig_i**2 for x_i, y_i, sig_i in zip(x, y, err_y))

    delta = B*D - A**2
    a = (B*E - A*C) / delta
    b = (C*D - A*E) / delta
    sig_a = np.sqrt(B / delta)
    sig_b = np.sqrt(D / delta)

    return (b, a), (sig_b, sig_a)

# Function constructors. Marginally more concise alternatives to lambda functions.

def power(n):
    """ Returns a function that raises its argument to the nth power. """

    return lambda x: x**n

def coeff(func, n):
    """
    Returns a single-variable function that evaluates to func(nx), where x is the I.V. May be used for quickly
    contructing linearly independent combinations of functions.
    """

    return lambda x: func(n * x)

# Distributions in general form

def lorentzian(x, *params):
    """
    Models a Lorentzian distribution. To convert to a function that only takes the independent variable,
    use fixed_params.

    :param x: Independent variable.
    :param params: The parameters of the distribution, in the order amplitude, width (gamma), and location (x_0).
    :return: The value of the distribution at x.
    """

    A, gamma, x_0 = params
    return A / np.pi * gamma / ((x - x_0)**2 + gamma**2)

def gaussian(x, *params):
    """
    Models a Gaussian distribution. To convert to a function that only takes the independent variable,
    use fixed_params.

    :param x: Independent variable.
    :param params: The parameters of the distribution, in the order amplitude, variance (sigma), and mean (mu).
    :return: The value of the distribution at x.
    """

    A, var, mu = params
    return A / np.sqrt(2 * np.pi * var) * np.exp(-(x - mu)**2 / (2 * var))

def poly(x, *params):
    """
    A function representing an nth order polynomial, where n is the length of params. To convert to a function that
    only takes the independent variable, use fixed_params.

    :param x: Independent variable.
    :param params: The coefficients of the terms in the polynomial, arranged in ascending order.
    :return: The value of the polynomial at x.
    """

    return sum([p * power(n)(x)for p, n in zip(params, range(0, len(params)))])

# General and particular linear combinations

def comb_gen(*funcs):
    """
    Returns a function that is a linear combination of the input functions, and takes the coefficients as the second
    argument (*params).
    """

    def comb_func(x, *params):
        return sum([par * fun(x) for par, fun in zip(params, funcs)])
    return comb_func

def comb(funcs, params):
    """
    Returns a function that is a linear combination of the input functions with the parameters as coefficients.
    May be used to retrieve the model after fitting to a dataset.
    """

    def comb_func(x):
        return sum([par * fun(x) for par, fun in zip(params, funcs)])
    return comb_func

def fixed_params(func, *params):
    """
    Returns a function defined as the input function with the given parameters. The function arguments must always be
    of the form (x, *params).

    :param func: The general function or distribution to set the parameters for.
    :param params: The parameters themselves, in the order specified inside the function.
    :return: The newly defined function, now in particular form.
    """

    def new_func(x):
        return func(x, *params)

    return new_func

def swap_args(func):
    """
    Decorator to swap the order of the parameters and the x values in a function's argument sequence. Necessary due to
    the inconsistency in the desired function arguments between curve_fit and ODR.

    :param func: The function to swap arguments for.
    :return: A newly defined function that takes the inputs in the opposite order.
    """

    def wrapper(params, x):
        return func(x, *params)

    return wrapper

# Analysis tools

def chi_squared(x, y, err_y, fit):
    """ Returns the chi-squared value for the input fit and data set. """

    return sum([((y_i - fit(x_i)) / err_i)**2 for x_i, y_i, err_i in zip(x, y, err_y)])

def r_squared(x, y, fit):
    """ Returns the unadjusted coefficient of determination for the input fit and data set. """

    y_bar = np.mean(y)

    # Total sum of squares
    total = sum([(y_i - y_bar)**2 for y_i in y])

    # Residual sum of squares
    res = sum([(y_i - fit(x_i))**2 for x_i, y_i in zip(x, y)])

    return 1 - res / total

# Input and output

def write_to_CSV(file_name='data', delim=',', columnar=True, **kwargs):
    """
    Writes the values in kwargs to a CSV file, preceded by the keys. The file extension is automatically appended.
    Use the csv module for a more versatile means of reading and writing csv formatted data. Writes data in columns
    with the keys as headers by default.
    """

    with open(file_name + '.csv', 'w') as file:

        def write_row(seq):
            file.write(delim.join(seq))
            file.write('\n')

        if columnar:
            n_rows = max(map(len, kwargs.values()))
            write_row(kwargs.keys())

            for i in range(0, n_rows):
                row = map(lambda v: str(v[i]) if len(v) > i else '', kwargs.values())
                write_row(row)
        else:
            for key, val in kwargs.items():
                write_row([key] + list(map(str, val)))

    file.close()

def read_from_CSV(file_name, delim=',', columnar=True):
    """
    Reads CSV data from the file with the given name (extensionless), returning a dictionary with the columns keyed by
    their headers. Use the csv module for a more versatile means of reading and writing csv formatted data. Assumes
    """

    data = dict()

    def convert(x):
        try:
            a = float(x)
        except ValueError:
            return x
        try:
            b = int(x)
        except ValueError:
            return a

        return b if a == b else a

    with open(file_name + ".csv") as file:

        if columnar:
            keys = file.readline().replace('\n', '').split(delim)
            for key in keys:
                data[key] = []

            for line in file:
                line = line.split(delim)
                for key, val in zip(keys, line):
                    val = val.replace('\n', '')
                    if val:
                        data[key].append(convert(val))
        else:
            for line in file:
                line = list(map(lambda x: convert(x.replace('\n', '')), line.split(delim)))
                data[line[0]] = line[1:] if len(line) > 1 else []

    file.close()
    return data

def draw_plot(x, y=None, err_x=None, err_y=None, fit=None, title=None, labels=None, dlegend='Data', flegend='Fit',
              data_style='ko', fit_style='b-', step=None, figure=None, subplot=None, top_adj=0.875, **kwargs):
    """
    Returns a plot object with the input settings. The 'style' parameters follow the abbreviations listed here:
    https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html, as well as supporting CN notation for color (try
    replacing 'b' with 'C4').
    Any other settings should be added by keyword. The list of kwargs can be found on the
    page linked to above. Color options may be found at https://matplotlib.org/examples/color/named_colors.html. They
    can be used with the color, markerfacecolor and markeredgecolor keywords, as well as a few others.

    :param x: The sequence of x values to plot.
    :param y: The sequence of y values to plot.
    :param err_x: The sequence of errors in the x values.
    :param err_y: The sequence of errors in the y values.
    :param fit: The function representing a fit to the data, to be plotted on the same axes.
    :param title: The title of the plot.
    :param labels: The axes labels, passed in as a sequence.
    .. note:: Matplotlib supports Latex syntax for titles and axes labels - see
            https://matplotlib.org/2.0.2/users/usetex.html
    :param dlegend: Legend label for the data.
    :param flegend: Legend label for the fit.
    :param data_style: Style for the data points. Black dots by default.
    :param fit_style: Style for the fit curve. Blue line by default.
    :param step: The step size for the points on the curve. Can be ignored for linear fits, but impacts the apparent
        smoothness of the curve for other functions.
    :param figure: The index of the figure to on which to draw the plot, starting at 0. By default the plot will be
        drawn on the last figure used. Multiple plots may be drawn in across multiple figures by modulating the index.
        The individual figures may be retrieved by index by calling plt.figure(figure).
    :param subplot: A 3-digit natural number or 3-item sequence that indicates the subplot of the figure to draw on.
        The first digit or item is the number of rows in the subplot grid, the second the number of columns, and the
        third the index of the location in the grid (starting at 1). Call plt.figure(figure).suptitle(<title>) or
        plt.gcf().suptitle(<title>) to set a title for the entire figure when displaying multiple subplots.
    :param top_adj: Due to a bug in matplotlib, a title added to the figure with suptitle will overlap with the subplot
        titles without manual adjustment. The default setting should work with a one line title and the standard font,
        but this will need to be set explicitly for longer/taller titles (by reducing the value, appears to be a drop
        of about 0.05 per line in the standard font?).
    :return: The figure on which the plots were drawn.
    """

    # Figure and subplot adjustments
    if figure is not None:
        plt.figure(figure)

    if subplot is not None:
        if isinstance(subplot, Sequence):
            plt.subplot(*subplot)
        else:
            plt.subplot(subplot)
        plt.tight_layout()
        plt.subplots_adjust(top=top_adj)

    # Error bar plot of the data
    if y is not None:
        plt.errorbar(x, y, err_y, err_x, data_style, label=dlegend, **kwargs)

    # Plots the fit
    if fit is not None:
        min_x, max_x = min(x), max(x)
        if step is None:
            step = (max_x - min_x) / 100

        num_steps = round((max_x - min_x) / step)
        f_x = [min_x + i * step for i in range(0, num_steps)]
        plt.plot(f_x, list(map(fit, f_x)), fit_style, label=flegend, **kwargs)

    # Sets the title
    if title is not None:
        plt.title(title)

    # Axis labels
    if labels is not None:
        xlabel, ylabel = labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    # Show the legend
    plt.legend()

    return plt.gcf()
