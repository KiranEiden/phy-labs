from setuptools import setup, find_packages

setup(name='phylabs',
      version='2.0',
      description='Python plotting/analysis interface usable with undergraduate physics labs',
      url='https://github.com/KiranEiden/phy-labs',
      author='Kiran Eiden',
      author_email='kiran.eiden@stonybrook.edu',
      packages=find_packages(),
      classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
      ],
      install_requires=['numpy', 'scipy', 'matplotlib', 'sympy'],
      python_requires='>=3.6',
      zip_safe=False)
