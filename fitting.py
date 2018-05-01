import inspect
from functools import reduce
from collections import Sequence

import numpy as np
from scipy.odr import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class Fit:
    """
    Class for storing fit parameters. Its objects are callable, returning the results of the fitted function.
    """

    def __init__(self, func, params, sd=None):

        self.p = params
        self.func = func

        # Executed on calling the fit object.
        self.on_call = fixed_params(func, *params)

        """
        Added to maintain some level of generality - there are likely more efficient implementations, and ones less
        dependent on order.
        """
        self._observed = ('p', 'sd')
        self.items = None

        self.sd = sd if sd is not None else len(params) * [None]

    def __call__(self, *args, **kwargs):

        return self.on_call(*args, **kwargs)

    # String representations - what should __str__ return?

    def __repr__(self):

        return "{}, {}".format(self.func.__name__, self.items)

    # For iterating over items.

    def __getitem__(self, item):

        return self.items[item]

    def __iter__(self):

        return iter(self.items)

    # Acts almost as a listener, reassigning self.items only when one of the component sequences is changed.

    def __setattr__(self, key, value):

        super().__setattr__(key, value)

        if 'items' in self.__dict__ and key in self._observed:
            self.items = list(zip(*map(self.__getattribute__, self._observed)))

def _validate(fun, eq_classes=({'x', 'y', 'err_x', 'err_y'})):
    """
    Decorator that ensures that all arguments in the same equivalence class have the same length. Could add automatic
    conversion to sequences and some other useful checks for parameters that cause cryptic error messages.
    """

    def wrapper(*args, **kwargs):

        # Get argument names and add them to kwargs.
        arg_names = inspect.getfullargspec(fun)[0]
        for k, v in zip(arg_names, args):
            kwargs[k] = v

        # Function for pairwise comparison.
        def reducer(arg_a, arg_b):
            a, b = kwargs[arg_a], kwargs[arg_b]
            if a is not None and b is not None and len(a) != len(b):
                raise ValueError("{} must be of same length as {}.".format(arg_a, arg_b))
            return arg_b if b is not None else arg_a

        # Compare all of members of the equivalence classes that were passed in as arguments.
        for cls in eq_classes:
            cls = filter(lambda arg: arg in kwargs, cls)
            try:
                reduce(reducer, cls)
            except TypeError:
                pass

        return fun(**kwargs)

    return wrapper

def _general_fit(x, y, funcs, err_y=None):
    """ Fits the dataset (x and y) to a linear combination of the input functions, returning the parameters. """

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
    Returns the results of a scipy curve fit for the input dataset and functions.

    :param x: The sequence of x values for the dataset.
    :param y: The sequence of y values for the dataset.
    :param funcs: The single function or sequence of functions to determine parameters for. If a sequence of functions
        is input, the fit will be to a linear combination, with the coefficients the parameters. Otherwise, the function
        is assumed to be of the form f(x, *params), where *params may either be varargs or expanded into the individual
        parameters.
    :param err_y: The errors in the y values.
    :param p0: An estimation of the parameters. If None (the default), the values will either be computed (if funcs is
        a sequence) or set to 1 by scipy. In the latter case, the arguments to the function should be in expanded form.
    """

    if isinstance(funcs, Sequence):
        if p0 is None:
            p0 = _general_fit(x, y, funcs, err_y)
        funcs = comb_gen(*funcs)

    res = curve_fit(funcs, x, y, p0, err_y)
    # noinspection PyTypeChecker
    return Fit(funcs, res[0], np.sqrt(np.diag(res[1])))

@_validate
def od_regression(x, y, funcs, err_x, err_y, p0=None):
    """
    Returns the results of an orthogonal distance regression performed for the input function.

    :param x: The sequence of x values for the dataset.
    :param y: The sequence of y values for the dataset.
    :param funcs: The single function or sequence of functions to determine parameters for. If a sequence of functions
        is input, the fit will be to a linear combination, with the coefficients the parameters. Otherwise, the function
        is assumed to be of the form f(x, *params), where *params may either be varargs or expanded into the individual
        parameters.
    :param err_x: The errors in the x values.
    :param err_y: The errors in the y values.
    :param p0: An estimation of the parameters. Must be input if funcs is not a sequence.
    """

    # Setup
    data = RealData(x, y, sx=err_x, sy=err_y)
    if isinstance(funcs, Sequence):
        if p0 is None:
            p0 = _general_fit(x, y, funcs, err_y)
        funcs = comb_gen(*funcs)
    elif p0 is None:
        raise ValueError("p0 must be specified if funcs is not a sequence.")
    model = Model(swap_args(funcs))

    # Regression
    odr = ODR(data, model, beta0=p0)
    output = odr.run()
    return Fit(funcs, output.beta, output.sd_beta)

@_validate
def lorentzian_fit(x, y, err_x=None, err_y=None):
    """
    Fits a Lorentzian distribution to the input data.

    :param x: The sequence of x values for the dataset.
    :param y: The sequence of y values for the dataset.
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
    gamma = (x_3 - x_1) / 2 if x_1 != x_3 else get_x(q_3) - get_x(q_1)

    # Amplitude estimate
    A = y_max * np.pi * ((x[x_s[max_i]] - x_0)**2 + gamma**2) / gamma

    p0 = A, gamma, x_0

    # Return the correct fit based on the input errors
    if err_x is None:
        return ls_regression(x, y, lorentzian, err_y, p0)

    return od_regression(x, y, lorentzian, err_x, err_y, p0)

@_validate
def gaussian_fit(x, y, err_x=None, err_y=None):
    """
    Fits a Gaussian distribution to the input data.

    :param x: The sequence of x values for the dataset.
    :param y: The sequence of y values for the dataset.
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

    return od_regression(x, y, gaussian, err_x, err_y, p0)

@_validate
def poly_fit(x, y, err_x=None, err_y=None, n=1):
    """
    Fits an nth order polynomial to the input data.

    :param x: The sequence of x values for the dataset.
    :param y: The sequence of y values for the dataset.
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
        return Fit(poly, res[0][::-1], np.sqrt(np.diag(res[1]))[::-1])

    return od_regression(x, y, funcs, err_x, err_y)

@_validate
def lin_fit_252(x, y, err_y=None):
    """
    Returns a linear fit for the input dataset as a tuple of tuples: the parameters and then their errors.
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
    constructing linearly independent combinations of functions.
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

# General and particular linear combinations.

def comb_gen(*funcs):
    """
    Returns a function that is a linear combination of the input functions, and takes the coefficients as the second
    argument (*params).
    """

    def linear_combination(x, *params):
        return sum([par * fun(x) for par, fun in zip(params, funcs)])

    return linear_combination

def comb(funcs, *params):
    """
    Returns a function that is a linear combination of the input functions with the parameters as coefficients.
    May be used to retrieve the model after fitting to a dataset.
    """

    def linear_combination(x):
        return sum([par * fun(x) for par, fun in zip(params, funcs)])

    return linear_combination

# Wrappers for modifying the input to other functions.

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

def weighted_mean(data, sd):
    """
    Computes an error weighted mean of the input data.

    :param data: Sequence of values to average.
    :param sd: Uncertainties for the dataset.
    """

    # Inverse squares
    inv_sq = [1 / sigma**2 for sigma in sd]
    sum_sq = sum(inv_sq)
    mean = sum(x * w for x, w in zip(data, inv_sq)) / sum_sq
    sd_mean = np.sqrt(1 / sum_sq)

    return mean, sd_mean

def chi_squared(x, y, err_y, fit):
    """ Returns the chi-squared value for the input fit and dataset. """

    return sum([((y_i - fit(x_i)) / err_i)**2 for x_i, y_i, err_i in zip(x, y, err_y)])

def r_squared(x, y, fit):
    """ Returns the unadjusted coefficient of determination for the input fit and dataset. """

    y_bar = np.mean(y)

    # Total sum of squares
    total = sum([(y_i - y_bar)**2 for y_i in y])

    # Residual sum of squares
    res = sum([(y_i - fit(x_i))**2 for x_i, y_i in zip(x, y)])

    return 1 - res / total

# Input and output

def write_to_CSV(file='data.csv', delim=',', columnar=True, **kwargs):
    """
    Writes the values in kwargs to a CSV file, preceded by the keys. Use the csv module for a more versatile means of
    reading and writing csv formatted data.

    :param file: The open file object or path to the file to write to, with extension.
    :type file: TextIO, str
    :param delim: The delimiter to use, set to a comma by default.
    :param columnar: Whether to output the data in columns (True) or in rows (False).

    Keywords, avoid using as column headers:
    :keyword fmt: A format to apply to data before writing. May be supplied as a single string or a dictionary - the
        format is applied to all data in the former case, or applied by header in the latter. Give a sequence of the
        same length as the data sequence in place of the string to apply by element.
    :keyword def_fmt: A default format that may be supplied when formatting using dictionaries. Will apply to any header
        not in fmt.
    :keyword line_end: A string to write to the end of the line before the newline character.
    :keyword preheader: Characters to be written to the line prior the header in columnar formats. Will be delimited
        if passed in as a sequence. A newline character is automatically appended to strings.
    :keyword postheader: Characters to be written to the line after the header in columnar formats. Will be delimited
        if passed in as a sequence. A newline character is automatically appended to strings.
    :keyword append: Whether to open the file in append or write mode. The default is write (False). This option is
        ignored if a file object was passed in.
    :keyword close: Whether to close the file before returning. The default setting is True.

    :return The file object that was written to.
    """

    # Some kwarg handling
    def_fmt = kwargs.pop('def_fmt', '{}')
    fmt = kwargs.pop('fmt', def_fmt)
    line_end = kwargs.pop('line_end', '')
    pre_header = kwargs.pop('preheader', None)
    post_header = kwargs.pop('postheader', None)
    close = kwargs.pop('close', True)
    file_mode = 'a' if kwargs.pop('append', False) else 'w'

    # Function for applying format
    def convert(key, i, out):

        fmt_ki = fmt
        if isinstance(fmt_ki, dict):
            fmt_ki = fmt_ki.get(key, def_fmt)
        if not isinstance(fmt_ki, str) and isinstance(fmt_ki, Sequence):
            fmt_ki = fmt_ki[i]
        return fmt_ki.format(out)

    # Open new file if path was given
    if isinstance(file, str):
        file = open(file, file_mode)

    # Helper functions
    def write_row(seq):
        file.write(delim.join(seq))
        file.write(line_end + '\n')

    def handle_padding(pad):
        if pad is not None:
            if isinstance(pad, str):
                file.write(pad + '\n')
            else:
                write_row(pad)

    # Write to file
    if columnar:
        handle_padding(pre_header)

        n_rows = max(map(len, kwargs.values()))
        write_row(kwargs.keys())

        handle_padding(post_header)

        for i in range(0, n_rows):
            row = map(lambda item: convert(item[0], i, item[1][i]) if len(item[1]) > i else '', kwargs.items())
            write_row(row)
    else:
        for k, v in kwargs.items():
            write_row([k] + [convert(k, i, v[i]) for i in range(len(v))])
    if close:
        file.close()
    return file

def read_from_CSV(file, delim=',', columnar=True, **kwargs):
    """
    Reads delimited data from the file with the given name (with extension). Use the csv module for a more versatile
    means of reading and writing csv formatted data.

    :param file: The open file object or path to the file to read from, with extension.
    :type file: TextIO, str
    :param delim: The delimiter used in the file, set to a comma by default.
    :param columnar: Whether the data should be read by column (True) or by row (False).

    Keywords:
    :keyword line_end: Any additional characters inserted before the newline that require removal.
    Specific to columnar formats:
    :keyword preheader: Integer value. Will skip the first pre_header lines in the file if set.
    :keyword postheader: Integer value. Will skip that number of lines immediately after the header if set.

    :return A dictionary containing lists of values, keyed by the headers.
    """

    data = dict()

    # Helper functions
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

    line_end = kwargs.setdefault('line_end', '')
    process_line = lambda l: l.replace(line_end + '\n', '').strip().split(delim)

    # Open file if path was given
    if isinstance(file, str):
        file = open(file)

    # Read in data
    if columnar:
        for i in range(0, kwargs.setdefault('preheader', 0)):
            next(file)

        keys = process_line(file.readline())
        for key in keys:
            data[key] = []

        for i in range(0, kwargs.setdefault('postheader', 0)):
            next(file)

        for line in file:
            line = line.split(delim)
            for key, val in zip(keys, line):
                val = val.replace('\n', '')
                if val:
                    data[key].append(convert(val))
    else:
        for line in map(process_line, file):
            line = list(map(convert, line))
            data[line[0]] = line[1:] if len(line) > 1 else []

    file.close()
    return data

def write_to_Tex(file='data.tex', table_spec='c', columnar=True, **kwargs):
    """
    Writes the data in the keyword arguments to the specified file in a LaTeX tabular format.

    :param file: The path to the file to write to.
    :param table_spec: Specifies the alignments and vertical dividers for the tabular (passed as argument to
        \begin{tabular}). A single alignment character will be expanded across all columns, which will be separated
        by vertical lines unless an alternative is specified with the 'separator' keyword.
    :param columnar: Data will be written in rows if set to False, columns otherwise.

    Keyword Arguments. Anything not listed will be interpreted as data (keyed by the header).
    :keyword append: Whether to append to the file (True) or write over it (False). False by default.
    :keyword fmt: Formatting to appy to the data. See write_to_CSV.
    :keyword center: Whether to center the table. True by default.
    :keyword postheader: Characters to write after the header row in columnar formats. '\hline' by default.
    :keyword separator: Will replace ' | ' in a cross-column expansion of a single alignment character.
    :keyword as_table: Whether to nest the tabular in a table environment. False by default.
    :keyword caption: The caption of the table. Will be ignored if unnested.
    :keyword label: The label of the table, for reference within a document. Will be ignored if unnested.
    :keyword placement: Fills the space of 'pos' in '\begin{table}[pos]'.
    """

    # Retrieve information from kwargs. Add some exceptions for improper input?
    def_fmt = kwargs.pop('def_fmt', None)
    fmt = kwargs.pop('fmt', None)
    append = kwargs.pop('append', False)
    center = kwargs.pop('center', True)
    file_mode = 'a' if append else 'w'
    postheader = kwargs.pop('postheader', '\\hline')
    separator = kwargs.pop('separator', ' | ')
    as_table = kwargs.pop('as_table', False)
    caption = kwargs.pop('caption', None)
    label = kwargs.pop('label', None)
    placement = '[{}]'.format(kwargs.pop('placement')) if 'placement' in kwargs else ''

    # Expands single alignment character across all columns, if necessary
    n_cols = len(kwargs.keys())
    if len(table_spec) == 1:
        table_spec = separator.join(n_cols * [table_spec])

    # Begin writing
    with open(file, file_mode) as file:

        file.write('\n')

        # LaTeX table stuff
        if as_table:
            file.write('\\begin{table}' + placement + '\n')
            if caption is not None:
                file.write('\\caption{' + caption + '}\n')
            if label is not None:
                file.write('\\label{' + label + '}\n')
        if center:
            file.write('\\begin{center}\n')

        file.write('\\begin{tabular}{' + table_spec + '}\n')

        # Write actual data
        if fmt is not None:
            kwargs['fmt'] = fmt
        if def_fmt is not None:
            kwargs['def_fmt'] = def_fmt

        write_to_CSV(file, delim=' & ', columnar=columnar, line_end=r'\\', postheader=postheader,
                append=True, close=False, **kwargs)

        # End environments
        file.write('\\end{tabular}\n')
        if center:
            file.write('\\end{center}\n')
        if as_table:
            file.write('\\end{table}\n')

def draw_plot(x, y=None, err_x=None, err_y=None, fit=None, title=None, xlabel=None, ylabel=None, **kwargs):
    """
    Returns a plot object with the input settings. The 'style' parameters follow the abbreviations listed here:
    https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html, as well as supporting CN notation for color (try
    replacing 'b' with 'C4').
    Any other settings should be added by keyword. The list of kwargs can be found on the
    page linked to above. Color options may be found at https://matplotlib.org/examples/color/named_colors.html. They
    can be used with the color, markerfacecolor and markeredgecolor keywords, as well as a few others.

    Parameters:
    :param x: The sequence of x values to plot.
    :param y: The sequence of y values to plot.
    :param err_x: The sequence of errors in the x values.
    :param err_y: The sequence of errors in the y values.
    :param fit: A function to be plotted on the same axes. The domain has the same bounds as the x argument, and the
        number of steps is controlled by the num_steps keyword.
    :param title: The title of the plot.
    :param xlabel: The x-axis label.
    :param ylabel: The y-axis label.
    .. note:: Matplotlib supports Latex syntax for titles and axes labels - see
            https://matplotlib.org/2.0.2/users/usetex.html

    Keywords:
    :keyword dstyle: The style specifier for the data points, using matplotlib abbreviations. Defaults to black
        circles. Colors may be overwritten with matplotlib keyword arguments.
    :keyword fstyle: The style specifier for the input function, using matplotlib abbreviations. Defaults to a solid
        blue line. Colors may be overwritten with matplotlib keyword arguments.
    :keyword dlegend: The legend entry for the data series.
    :keyword flegend: The legend entry for the function.
    :keyword num_steps: The number of points to be computed from the fit argument. Impacts the smoothness of the curve.
        The default setting is 100.
    :keyword figure: The index of the figure to on which to draw the plot, starting at 0. By default the plot will be
        drawn on the last figure used. Multiple plots may be drawn in across multiple figures by modulating the index.
        The individual figures may be retrieved by index by calling plt.figure(figure).
    :keyword subplot: A 3-digit natural number or 3-item sequence that indicates the subplot of the figure to draw on.
        The first digit or item is the number of rows in the subplot grid, the second the number of columns, and the
        third the index of the location in the grid (starting at 1). Call plt.figure(figure).suptitle(<title>) or
        plt.gcf().suptitle(<title>) to set a title for the entire figure when displaying multiple subplots, or use the
        suptitle keyword.
    :keyword suptitle: Sets the title at the top of the figure if plots were drawn on subplots.
    :keyword top_adj: Due to a bug in matplotlib, a title added to the figure with suptitle will overlap with the subplot
        titles without manual adjustment. The default setting should work with a one line title and the standard font,
        but this will need to be set explicitly for longer/taller titles (by reducing the value, appears to be a drop
        of about 0.05 per line in the standard font?).
    :return: The figure on which the plot was drawn.
    """

    # Figure and subplot adjustments
    if 'figure' in kwargs:
        plt.figure(kwargs.pop('figure'))

    if 'subplot' in kwargs:
        subplot = kwargs.pop('subplot')
        if isinstance(subplot, Sequence):
            plt.subplot(*subplot)
        else:
            plt.subplot(subplot)
        plt.tight_layout()
        plt.subplots_adjust(top=kwargs.pop('top_adj', 0.875))

    if 'suptitle' in kwargs:
        plt.gcf().suptitle(kwargs.pop('suptitle'))

    # Data keywords
    dstyle = kwargs.pop('dstyle', 'ko')
    dlegend = kwargs.pop('dlegend', 'Data')

    # Fit keywords
    num_steps = kwargs.pop('num_steps', 100)
    fstyle = kwargs.pop('fstyle', 'b-')
    flegend = kwargs.pop('flegend', 'Fit')

    # Error bar plot of the data
    if y is not None:
        plt.errorbar(x, y, err_y, err_x, dstyle, label=dlegend, **kwargs)

    # Plots the fit
    if fit is not None:
        min_x, max_x = min(x), max(x)
        step = (max_x - min_x) / num_steps
        f_x = [min_x + i * step for i in range(0, num_steps)]
        plt.plot(f_x, list(map(fit, f_x)), fstyle, label=flegend, **kwargs)

    # Sets the title
    if title is not None:
        plt.title(title)

    # Axis labels
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    # Show the legend
    plt.legend()

    return plt.gcf()

def save_figure(file_name, *extensions, **kwargs):
    """
    Saves the current or specified figure to files with the given name and extensions and shows the plot if requested,
    clearing all figures.

    :param file_name: The name of the plot files.
    :param extensions: A sequence of extensions to save the file with.

    :keyword figure: The figure to save and display. Default to the current figure.
    :keyword mode: May be set to 'clf', 'show' or None to apply none of those options. The
        corresponding pyplot method will be called after saving the figure. The default is 'clf'.
    A list of additional kwargs may be found here: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html.
    The defaults for 'dpi' and 'bbox_inches' are reset to 'figure' and 'tight', but may still be overridden.
    """

    kwargs.setdefault('dpi', 'figure')
    kwargs.setdefault('bbox_inches', 'tight')

    figure = plt.figure(kwargs.pop('figure')) if 'figure' in kwargs else plt.gcf()
    mode = kwargs.pop('mode', 'clf')

    for ext in extensions:
        figure.savefig(file_name + '.' + ext, **kwargs)

    modes = {'clf', 'show'}
    if mode is None:
        return
    if mode not in modes:
        raise ValueError('Invalid mode specified.')
    getattr(plt, mode)()
