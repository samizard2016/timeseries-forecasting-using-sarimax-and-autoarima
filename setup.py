# -*- coding: utf-8 -*-
"""
Created on Feb 11 2022

@author: Paul, Samir
building exe - python setup1.py build
building installer for windows - python setup1.py bdist_msi
"""
from cx_Freeze import setup, Executable
import os
import sys
from cx_Freeze import hooks
import scipy
import sklearn

def _init_numpy_mkl():
    # Numpy+MKL on Windows only
    import os
    import ctypes
    if os.name != 'nt':
        return
    # disable Intel Fortran default console event handler
    env = 'FOR_DISABLE_CONSOLE_CTRL_HANDLER'
    if env not in os.environ:
        os.environ[env] = '1'
    # preload MKL DLLs from numpy.core
    try:
        _core = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'core')
        for _dll in ('mkl_rt', 'libiomp5md', 'mkl_core', 'mkl_intel_thread',
                     'libmmd', 'libifcoremd', 'libimalloc',
                     'libopenblas-57db09cfe174768fb409a6bb5a530d4c.dll'):
            ctypes.cdll.LoadLibrary(os.path.join(_core,_dll))
    except Exception:
        pass

_init_numpy_mkl()

PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))
os.environ['TCL_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR,'tcl','tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tk8.6')

#mkl_int = "C://Users//pauls//Anaconda3//Library//bin//mkl_intel_thread.dll"
#'atexit', 'numpy.core._methods', 'numpy.lib.format'
# scipy_path = os.path.dirname(scipy.__file__)
# sklearn_path = os.path.dirname(sklearn.__file__)
files = {"include_files": ["C://Python310//DLLs//tcl86t.dll",
         "C://python310//python3.dll",
         "C://Python310//vcruntime140.dll",
         "C://Python310//DLLs//tk86t.dll",      
         "Lib",
         "anuman.png","anuman.ico","dashboard.png","dataset.png","run.png","setup.png","start afefx.png",
         "tsfx.png","Training_Punjab.xlsx","new sample dataset.xlsx",
         "forecast_and_elasticity.afefx","settings.json","NPI growth data.xlsx"],
         "includes": ['pandas','numpy','atexit', 'numpy.core._methods', 'numpy.lib.format','multiprocessing'],
          'excludes': ['boto.compat.sys',
                 'boto.compat._sre',
                 'boto.compat._json',
                 'boto.compat._locale',
                 'boto.compat._struct',
                 'boto.compat.array'],
          "packages":['sklearn','scipy'],
        #   "zip_include_packages":['encodings'],
          "include_msvcr": True,
          "add_to_path":True
         }
# files['include_files'].append("lib/")

base = None

if sys.platform == 'win32':
    base = "Win32GUI"

executables = [Executable("anuman.py", 
                shortcut_name="Anuman - time-series forecaster",
                shortcut_dir="DesktopFolder",
                icon="anuman.ico",
base=base)]

options = {'build_exe': files}
setup(
    name = "Anuman",
    author = "Samir Paul",
    options = options,
    version = "0.1.4",
    description = 'Anuman 1.4 - time-series forecasting & elasticity framework',
    executables = executables

)
