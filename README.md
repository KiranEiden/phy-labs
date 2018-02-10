# phy-labs
A Python module designed to provide a convenient interface for fitting to and plotting datasets using scipy and 
matplotlib. The intended use at present is on undergraduate physics labs of the Stony Brook University variety - 
effectiveness on anything else is unconfirmed and therefore not guaranteed.

Check the 'examples' directory for sample files using the functions in the module.

#### On Python
For those new to Python: the latest release may be found at https://www.python.org/downloads/release/python-364/.
Python comes preinstalled on OS X and most Linux distributions.

Some additional packages are required for this module, such as scipy, numpy and matplotlib. They may be installed using
Any recent Python release will include the package manager pip, which may be used to install these by typing 'pip 
install' followed by the package name in a terminal window. 

**Note**: If both Python 2 and Python 3 are installed, 
Python 3 may need to be referenced using 'python3' and 'pip3' instead of 'python' and 'pip'. **Also Note**: Attempting
to use pip to install packages may result in a permission error. If this occurs, try doing a user install by adding a
'-U' or '--user' after 'pip install' (ex. pip install -U matplotlib). Another option is to install as a superuser, but 
that is generally not recommended as it can mess with system-level packages.

Python files may be run from the terminal by typing 'python' followed by the file name (with extension). The command
with no file name starts the Python shell. Check out Jupyter notebooks for a more interactive means of manipulating
data.