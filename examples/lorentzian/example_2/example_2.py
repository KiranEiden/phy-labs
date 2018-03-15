# Example with small, manually input data sets. Demonstrates multiple plots on a single figure.

# Specifies path to fitting, if this is to be run from the terminal.
# You may also export the location of the module to the $PYTHONPATH variable in a shell
# source file or directly from the terminal.
import sys
sys.path.append('../../../')
from fitting import *

# Function for repeated plotting of Lorentzian distributions
def do_fit(x, y, err_x, err_y, title, pos, file=None, color='orangered'):

    print(title)

    # Fits to the data set
    fit = lorentzian_fit(x, y, err_x, err_y)
    # Amplitude parameter
    amp, err_amp = fit.p[0], fit.sd[1]
    # Half width at half maximum
    gam, err_gam = fit.p[1], fit.sd[1]
    # Location parameter
    loc, err_loc = fit.p[2], fit.sd[2]
    # Computes peak value from width and amplitude
    peak, err_peak = amp / (np.pi * gam), np.sqrt(
        (err_amp / (np.pi * gam))**2 + (amp * err_gam / (np.pi * gam**2))**2)

    # Prints parameters
    print("Resonant Frequency: {} +- {}".format(loc, err_loc))
    print("Gamma: {} +- {}".format(gam, err_gam))
    print("Amplitude: {} +- {}".format(amp, err_amp))
    print("Peak: {} +- {}".format(peak, err_peak))
    print()

    # Plot on figure 0
    labels = r"$\omega$ (Hz)", "A (cm)"

    # Note that the first 7 arguments may also be specified by keyword (e.g. fit=fit), and must be if an argument is
    # skipped or they are entered in a different order. The keyword names all match the variable names.
    draw_plot(x, y, err_x, err_y, fit, title, labels, figure=0, subplot=(2, 2, pos), color=color, markersize=4.0)

    if file is not None:
        # Plot on separate figure
        draw_plot(x, y, err_x, err_y, fit, title, labels, figure=pos, color=color, markersize=4.5)
        save_figure(file, 'png', figure=pos)

# Sets the title of the first figure
plt.figure(0)
plt.suptitle("Amplitude vs. Driving Frequency")

# Errors - all the same in this example
err_freq = 9 * [0.03]
err_A = 9 * [0.007]

# Plot 1
freq_1 = [(2 + i) / 10 for i in range(0, len(err_freq))]
A_1 = [0.053, 0.083, 0.144, 0.299, 0.593, 0.383, 0.225, 0.101, 0.062]

do_fit(freq_1, A_1, err_freq, err_A, "Plot 1", 1, file='plot_1')

# Plot 2
freq_2 = [(1.6 + i) / 10 for i in range(0, len(err_freq))]
A_2 = [0.057, 0.072, 0.111, 0.232, 0.333, 0.256, 0.133, 0.088, 0.066]

do_fit(freq_2, A_2, err_freq, err_A, "Plot 2", 2, file='plot_2', color='peru')

# Plot 3
freq_3 = [(2.3 + i) / 10 for i in range(0, len(err_freq))]
A_3 = [0.049, 0.067, 0.083, 0.228, 0.444, 0.272, 0.112, 0.077, 0.055]

do_fit(freq_3, A_3, err_freq, err_A, "Plot 3", 3, file='plot_3', color='lightskyblue')

# Plot 4
freq_4 = [(1.3 + i) / 10 for i in range(0, len(err_freq))]
A_4 = [0.043, 0.064, 0.075, 0.193, 0.346, 0.270, 0.129, 0.069, 0.054]

do_fit(freq_4, A_4, err_freq, err_A, "Plot 4", 4, file='plot_4', color='navajowhite')

# Save figure with multiple plots, and display all
save_figure('all_plots', 'png', figure=0, display=True)