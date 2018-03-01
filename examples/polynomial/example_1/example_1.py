# Specifies path to fitting, if this is to be run from the terminal.
# You may also export the location of the module to the $PYTHONPATH variable in a shell
# source file or directly from the terminal.
import sys
sys.path.append('../../../')
from fitting import *

# Reads data from a .csv file
data = read_from_CSV('data')
x = data['x']
y = data['y']
err_y = data['err_y']

res_1 = poly_fit(x, y, err_y=err_y)
fit_1 = fixed_params(poly, *res_1[0])
print("Linear Fit: {}".format(res_1))

res_3 = poly_fit(x, y, err_y=err_y, n=3)
fit_3 = fixed_params(poly, *res_3[0])
print("Cubic Fit: {}".format(res_3))

res_5 = poly_fit(x, y, err_y=err_y, n=5)
fit_5 = fixed_params(poly, *res_5[0])
print("5th-Order Fit: {}".format(res_5))

draw_plot(x, y, err_y=err_y)
plt.show()