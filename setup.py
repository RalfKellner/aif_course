from setuptools import setup

setup(name='aif_course',
      version='0.1',
      description='Class and function definitions for the AIF course',
      url='https://github.com/RalfKellner/aif_course',
      author='Ralf Kellner',
      author_email='ralf.kellner@uni-passau.de',
      license='MIT',
      packages=['aif_course'],
      install_requires=[
        'yfinance',
        'pandas',
        'pandas_ta',
        'numpy',
        'matplotlib',
        'gym',
        'sklearn'    
      ],
      zip_safe=False)