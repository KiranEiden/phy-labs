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
def print_poly(res):
    """ Prints the regression result in a tabular format. """

    print("Order {} Fit".format(len(res[0]) - 1))

    row_format = "{:>15}" * (len(res[0]) + 1)
    print(row_format.format('Power:', *range(0, len(res[0]))))
    row_format = "{:>15.6}" * (len(res[0]) + 1)
    print(row_format.format('Coeff:', *res[0]))
    print(row_format.format('Error:', *res[1]))

# Linear fit
res_1 = poly_fit(x, y, err_y=err_y)
fit_1 = fixed_params(poly, *res_1[0])
print_poly(res_1)

# Cubic fit
res_3 = poly_fit(x, y, err_y=err_y, n=3)
fit_3 = fixed_params(poly, *res_3[0])
print_poly(res_3)

# 5th order fit
res_5 = poly_fit(x, y, err_y=err_y, n=5)
fit_5 = fixed_params(poly, *res_5[0])
print_poly(res_5)

# Draws the plot with the data and the 3 fits
draw_plot(x, y, err_y=err_y, fit=fit_1, title="Sample Polynomial Fits", flegend='Linear')
draw_plot(x, fit=fit_3, flegend='Cubic', color='orangered')
draw_plot(x, fit=fit_5, flegend='5th Order', color='darkorchid')

# Save the figure in two separate formats and display the plot
save_figure('plot', 'eps', 'png', display=True)