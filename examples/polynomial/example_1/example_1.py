# Specifies path to fitting, if this is to be run from the terminal.
# You may also export the location of the module to the $PYTHONPATH variable in a shell
# source file or directly from the terminal.
import sys
sys.path.append('../../../')
from fitting import *

# Reads data from a .csv file
data = read_from_CSV('data.csv')
x = data['x']
y = data['y']
err_y = data['err_y']

# Function for nice printing
def print_poly(fit):
    """ Prints the regression result in a tabular format. """

    print("Order {} Fit".format(len(fit.p) - 1))

    row_format = "{:>15}" * (len(fit.p) + 1)
    print(row_format.format('Power:', *range(0, len(fit.p))))
    row_format = "{:>15.6}" * (len(fit.p) + 1)
    print(row_format.format('Coeff:', *fit.p))
    print(row_format.format('Error:', *fit.sd))

# Linear fit
fit_1 = poly_fit(x, y, err_y=err_y)
print_poly(fit_1)

# Cubic fit
fit_3 = poly_fit(x, y, err_y=err_y, n=3)
print_poly(fit_3)

# 5th order fit
fit_5 = poly_fit(x, y, err_y=err_y, n=5)
print_poly(fit_5)

# Draws the plot with the data and the 3 fits
draw_plot(x, y, err_y=err_y, fit=fit_1, title="Sample Polynomial Fits", flegend='Linear')
draw_plot(x, fit=fit_3, flegend='Cubic', color='orangered')
draw_plot(x, fit=fit_5, flegend='5th Order', color='darkorchid')

# Save the figure in two separate formats and display the plot
save_figure('plot', 'eps', 'png', display=True)