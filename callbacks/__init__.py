
# export all the callbacks
from callbacks.csv_log_callback import CSVLogCallback
from callbacks.save_on_timestep_callback import SaveOnTimestepCallback

__all__ = [
    "CSVLogCallback",
    "SaveOnTimestepCallback"
]
# __all__ is a list of public objects of that module, as interpreted by import *
# This is a convention in Python to define what is exported when the module is imported
# from this module. It helps to avoid polluting the namespace with private objects.
# It is not strictly necessary, but it is a good practice to include it in modules
# that are intended to be used as libraries or packages.
# This allows the module to be imported and used in other scripts or modules
# without exposing internal implementation details.
# This is especially useful when the module contains a lot of functions or classes,
# and you want to control what is accessible to the user.
# By defining __all__, you can ensure that only the specified objects are imported
# when the module is imported with from module import *.
# This can help to avoid name clashes and make the code more readable.
# It also serves as a form of documentation, indicating which objects are intended
# to be part of the public API of the module.
# In this case, we are exporting the CSVLogCallback and SaveOnTimestepCallback classes
# from the callbacks module, which are intended to be used as callbacks for
# training reinforcement learning agents. By including them in __all__, we are
# indicating that these classes are part of the public API of the module and
# should be accessible when the module is imported.