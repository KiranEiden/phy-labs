"""
Example fit to a Lorentzian data set. This may all be done interactively using a Jupyter notebook - see
https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook for more information.
"""

# Specifies path to fitting, if this is to be run from the terminal.
# You may also export the location of the module to the $PYTHONPATH variable in a shell
# source file or directly from the terminal.
import sys
sys.path.append('../../../')
from fitting import *

# Reads the data from a CSV file
data = read_from_CSV('data.csv')
x, y, err_y = data['x'], data['y'], data['err_y']

"""
Alternative ways to declare lists:
x = [4, 14, 27.1e12, 19.2, 4.4, 3, np.sqrt(8)] # List of integers and floats
x = list(range(0, 100)) # Numbers from 0 to 99
y = [(18 + i) / 10 for i in range(0, 61)] # Numbers from 18.0 to 24.0
err_y = len(y) * [0.05] # A list of the same length as y, but containing only 0.05
"""

fit = lorentzian_fit(x, y, err_y=err_y)

# Print the parameters
print("Amplitude: {} +- {}".format(fit.p[0], fit.sd[0]))
print("Peak Value: {}".format(fit.p[0] / (np.pi * fit.p[1])))
print("Gamma: {} +- {}".format(fit.p[1], fit.sd[1]))
print("Location: {} +- {}".format(fit.p[2], fit.sd[2]))

# Convert the result to a fit function and plot the data
labels = "x (unit)", "y (unit)"
draw_plot(x, y, err_y=err_y, fit=fit, title=r"$\Pi$co de Gallo", labels=labels, color='orangered')

# Save the plot programmatically.
# plt.gcf().savefig('plot.png', dpi='figure')

# Display the plot, with an interactive toolbar (for panning, saving, etc.) at the bottom.
plt.show()