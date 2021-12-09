# ------------------------------------------------------------------------------ #
#                                                                                #
# Title:                                                       Check Dependecies #
# Authors:                     Stephan Weißbach, Stanislav Sys Clement Schneider #                                                                              
# Purpose:                                     Test to see if deps are installed #                                                                              
# Last Update:                                                        02.06.2021 #
#                                                                                #
# ------------------------------------------------------------------------------ #

from termcolor import colored
import importlib
import subprocess
import sys


assert sys.version_info >= (3, 6)

# get installed packages from pip freeze
reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
#split Strings by "==", so we don't look at versioning
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

# list of necessary deps
deps = ["torch", "detectron2", "torchvision", "opencv-python", "sklearn",\
    "pandas", "numpy", "funcy", "matplotlib", "Pillow", "Shapely", "termcolor"]

# check if deps are installed and loadable
deps_installed = True
for dependency in deps:
    if (dependency in installed_packages):
        print(colored(f'{dependency+":" : <14}{"is installed" : >20}', 'green'))
        try:
            if(dependency =="opencv-python"):
                importlib.import_module("cv2")

            elif(dependency =="Pillow"):
                importlib.import_module("PIL")
            
            elif(dependency =="Shapely"):
                importlib.import_module("shapely")
            else:
                importlib.import_module(dependency)

        except ModuleNotFoundError:
            print(colored(f'ERROR - Can\'t import module: {dependency}!', "red") )
            pass

    else:
        print( colored(f'{dependency+":" : <14} {"not installed" : >20}', 'red'))
        deps_installed = False

if deps_installed:
    print(f"\nDependency requirements are met!\n")