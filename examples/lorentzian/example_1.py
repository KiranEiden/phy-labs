"""
Example fit to a Lorentzian data set. This may all be done interactively using a Jupyter notebook - see
https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook for more information.
"""

# Specifies path to fitting, if this is to be run from the command line.
import sys
sys.path.append('../../')
from fitting import *

# Reads the data from a CSV file
data = read_from_CSV('data')
x, y, err_y = data['x'], data['y'], data['err_y']
# The data set may be programmatically declared as x = [x_1, x_2, ..., x_n], or using a list comprehension.
p, err_p = lorentzian_fit(x, y, err_y=err_y)

# Print the parameters
print("Amplitude: {} +- {}".format(p[0], err_p[0]))
print("Gamma: {} +- {}".format(p[1], err_p[1]))
print("Location: {} +- {}".format(p[2], err_p[2]))

# Convert the result to a fit function and plot the data
fit = fixed_params(lorentzian, *p)
labels = "x (unit)", "y (unit)"
plot = get_plot(x, y, err_y=err_y, fit=fit, title=r"$\Pi$co de Gallo", labels=labels, color='hotpink')

# Save the plot programmatically.
# fig.savefig('plot.png', dpi=fig.dpi)

# Display the plot, with an interactive toolbar (for panning, saving, etc.) at the bottom.
plot.show()