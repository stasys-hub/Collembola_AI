# ------------------------------------------------------------------------------ #
#                                                                                #
# Title:                                                       Check Dependecies #
# Authors:                                      Stephan Weißbach & Stanislav Sys #                                                                              
# Purpose:                                     Test to see if deps are installed #                                                                              
# Last Update:                                                        02.06.2021 #
#                                                                                #
# ------------------------------------------------------------------------------ #

 #imports
import importlib
import sys



def check_dependencies(list_of_dependencies: list) -> None:

    for dependency in list_of_dependencies:
        try:
            # load dependency
            importlib.import_module(dependency)
            print(f"Succesfully loaded:\t {dependency}")

        except ModuleNotFoundError:
            print(f'ERROR: Can\'t import module: {dependency}!' )
            pass

if __name__=="__main__":

    # check python version
    assert sys.version_info >= (3, 6)

    # list of dpendecies
    deps = ["torch", "detectron2", "torchvision", "cv2","pandas", "sklearn", "PIL", "json", "shapely", "numpy"]

    check_dependencies(deps)
