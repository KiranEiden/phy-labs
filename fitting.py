import inspect
from functools import reduce

import numpy as np
from scipy.odr import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def _validate(fun, eq_classes=({'x', 'y', 'err_x', 'err_y'}, {'funcs', 'beta', 'p0'})):
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
    if n < 2:
        raise ValueError("Invalid data set - the lists must have lengths greater than 1.")

    # Also error handling, but with the uncertainties in the data set
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
    """ Returns the results of a scipy curve fit for the input data set and functions. """

    if p0 is None:
        p0 = _general_fit(x, y, funcs, err_y)
        funcs = comb_gen(funcs)
    res = curve_fit(funcs, x, y, p0, sigma=err_y)
    return res[0], np.sqrt(np.diag(res[1]))

@_validate
def od_regression(x, y, funcs, err_x, err_y, beta=None):
    """ Returns the results of an orthogonal distance regression performed for the input function. """

    # Setup
    data = RealData(x, y, sx=err_x, sy=err_y)
    if beta is None:
        beta = np.array(_general_fit(x, y, funcs, err_y), dtype=np.double)
        funcs = comb_gen(funcs)
    model = Model(funcs)

    # Regression
    odr = ODR(data, model, beta0=beta)
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
        The parameters are amplitude, width (gamma) and location (x_0) respectively.
    """

    # Initializes components needed for estimation
    n = len(x)
    epsilon_n = 8 / np.sqrt(n) + 460 / n**2
    weights = [np.cos(np.pi * ((i - 0.5) / n - 0.5))**(2 + epsilon_n) * np.cos(2 * np.pi * ((i - 0.5) / n - 0.5))
               for i in range(1, n + 1)]
    c = sum(weights)
    x_s = sorted(range(0, n), key=lambda i: x[i])

    # Location parameter estimate
    x_0 = 0
    for i in range(0, n):
        x_0 += weights[i] / c * x[x_s[i]]

    # Width parameter estimate
    q_1 = (n - 1) * 0.25
    q_3 = (n - 1) * 0.75
    q_1 = int(q_1 + 1)
    q_3 = int(q_3)

    y_max = max(y)
    y_max_i = y.index(y_max)
    half_max = y_max / 2

    get_y, get_x = lambda i: y[x_s[i]], lambda i: x[x_s[i]]
    direction = lambda val, ref: int((ref - val) / abs(ref - val))

    def find_x(loc):
        """ Finds the approximate x-value that yields half-max. Slow, but it works on small datasets. """

        try:
            init_dir = direction(get_y(loc), half_max)
        except ZeroDivisionError:
            return get_x(loc)

        direct = init_dir
        while direct == init_dir:
            loc += direct
            try:
                direct = direction(get_y(loc), half_max)
            except ZeroDivisionError:
                return get_x(loc)

        try:
            m = (get_y(loc) - get_y(loc + direct)) / (get_x(loc) - get_x(loc + direct))
            b = get_y(loc) - m * get_x(loc)
            return (half_max - b) / m
        except IndexError:
           return get_x(n - 1) if loc >= n else get_x(0)

    x_1 = find_x(q_1)
    x_3 = find_x(q_3)
    gamma = (x_3 - x_1) / 2

    # Amplitude estimate
    A = y_max * np.pi * ((x[y_max_i] - x_0)**2 + gamma**2) / gamma

    p0 = A, gamma, x_0

    # Return the correct fit based on the input errors
    if err_x is None:
        return ls_regression(x, y, lorentzian, err_y, p0)
    else:
        return od_regression(x, y, lorentzian, err_x, err_y, p0)

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

    mu = sum(x) / len(x)
    var = sum([x_i**2 for x_i in x]) / len(x) - mu**2
    A = max(y) * np.sqrt(2 * np.pi * var)

    p0 = A, var, mu

    # Return the correct fit based on the input errors
    if err_x is None:
        return ls_regression(x, y, lorentzian, err_y, p0)
    else:
        return od_regression(x, y, lorentzian, err_x, err_y, p0)

@_validate
def poly_fit(x, y, err_x=None, err_y=None, n=1):
    """
    Fits an nth order polynomial to the input data.

    :param x: The sequence of x values for the data set.
    :param y: The sequence of y values for the data set.
    :param err_x: The errors in the x values.
    :param err_y: The errors in the y values.
    :param n: The order of the polynomial.
    :return: A two dimensional array, containing first the values of the parameters and then the corresponding errors.
        The parameters are the coefficients of the terms in the polynomial, in ascending order from 1 to x^n.
    """

    funcs = [power(i) for i in range(0, n + 1)]
    if err_x is None:
        return ls_regression(x, y, funcs, err_y)
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

# Function constructors. Alternatives to lambda expressions.

def power(n):
    """ Returns a power function that raises its argument to the given power. """

    def power_func(x):
        return x**n
    return power_func

def exp(n):
    """ Returns an exponential function with n as the coefficient in the exponent. """

    def exp_func(x):
        return np.exp(n * x)
    return exp_func

def log(base):
    """ Returns a log function with the given base. """

    def log_func(x):
        return np.log(x) / np.log(base)
    return log_func

# Distributions in general form

def lorentzian(x, *params):
    """
    Models a Lorentzian distribution. To convert to a function that only takes the independent variable,
    use fixed_params.

    :param x: Independent variable.
    :param params: The parameters of the distribution, in the order amplitude, width (gamma), and location (x_0).
    :return: The value of the distributiion at x.
    """

    A, gamma, x_0 = params
    return A / np.pi * gamma / ((x - x_0)**2 + gamma**2)

def gaussian(x, *params):
    """
    Models a Gaussian distribution. To convert to a function that only takes the independent variable,
    use fixed_params.

    :param x: Independent variable.
    :param params: The parameters of the distribution, in the order amplitude, variance (sigma), and mean (mu).
    :return: The value of the distributiion at x.
    """

    A, var, mu = params
    return A / np.sqrt(2 * np.pi * var) * np.exp(-(x - mu)**2 / (2 * var))

# General and particular linear combinations

def comb_gen(funcs):
    """
    Returns a function that is a linear combination of the input functions, and takes the coefficients as the second
    argument (*params).
    """

    try:
        if len(funcs) < 1:
            return None
    except AttributeError:
        return funcs

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

def fixed_params(fun, *params):
    """
    Returns a function defined as the input function with the given parameters. The function arguments must always be
    of the form (x, *params).

    :param fun: The general function or distribution to set the parameters for.
    :param params: The parameters themselves, in the order specified inside the function.
    :return: The newly constructed function, now in particular form.
    """

    def new_fun(x):
        return fun(x, *params)

    return new_fun

# Analysis tools

def chi_squared(x, y, err_y, fit):
    """ Returns the chi-squared value for the input fit and data set. """

    return sum([((y_i - fit(x_i)) / err_i)**2 for x_i, y_i, err_i in zip(x, y, err_y)])

def write_to_CSV(*data, headers=None, file_name="data"):
    """
    Writes the input data to a CSV file - the extension is automatically appended to the given file name. The number
    of rows is assumed to be equal to the length of the first element of data. Can also use csv module.
    """

    with open(file_name + ".csv", "w") as file:
        if headers is not None:
            for h in range(0, len(headers)):
                delim = "\n" if h == len(headers) - 1 else ","
                file.write(''.join((headers[h], delim)))

        for i in range(0, len(data[0])):
            for j in range(0, len(data)):
                delim = "\n" if j == len(data) - 1 else ","
                file.write(''.join((str(data[j][i]), delim)))

    file.close()

def read_from_CSV(file_name):
    """
    Reads CSV data from the file with the given name, returning a dictionary with the columns keyed by their
    headers. Can also use csv module.
    """

    data = dict()

    with open(file_name + ".csv") as file:
        line = file.readline()
        headers = line.replace("\n", "").split(",")
        for i in range(0, len(headers)):
            headers[i] = headers[i][1:len(headers[i]) - 1]
            data[headers[i]] = list()

        while True:
            line = file.readline()
            if line == "":
                break

            row = line.split(",")
            for header, cell in zip(headers, row):
                data[header].append(float(cell))

    return data


def get_plot(x, y, err_x=None, err_y=None, fit=None, title=None, labels=None, data_style='ko', fit_style='b-',
             step=None, **kwargs):
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
    :param data_style: Style for the data points. Black dots by default.
    :param fit_style: Style for the fit curve. Blue line by default.
    :param step: The step size for the points on the curve. Can be ignored for linear fits, but impacts the apparent
        smoothness of the curve for other functions.
    :return: The constructed plot object.
    """

    fig, ax = plt.subplots()

    # Error bar plot of the data
    ax.errorbar(x, y, err_y, err_x, data_style, label="Data", **kwargs)

    # Plots the fit
    if fit is not None:
        min_x, max_x = min(x), max(x)
        if step is None:
            step = (max_x - min_x) / 100

        num_steps = round((max_x - min_x) / step)
        f_x = [min_x + i * step for i in range(0, num_steps)]
        ax.plot(f_x, list(map(fit, f_x)), fit_style, label="Fit", **kwargs)

    # Sets the title
    if title is not None:
        ax.set_title(title)

    # Axis labels
    if labels is not None:
        xlabel, ylabel = labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # Show the legend
    ax.legend()

    return plt