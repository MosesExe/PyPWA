"""
PyPWA.lib.data:
"""

__author__ = "Mark Jones"
__credits__ = ["Mark Jones"]
__license__ = "MIT"
__version__ = "[CURRENT_VERSION]"
__maintainer__ = "Mark Jones"
__email__ = "maj@jlab.org"
__status__ = "[CURRENT_STATUS]"

import PyPWA.data.filehandling, os

class Interface(object):

    supported_file_types = { ".txt" : "Kv" }

    cache = True

    def __init__(self, config = None):
        if config != None:
            self.cache = config["Use Cache"]


    def extension_test(self, file_location ):
        try:
            handlers = self.supported_file_types[os.path.splitext(file_location)[1]]
        except KeyError:
            raise TypeError("{0} is not a supported file extension".format(os.path.splitext(file_location)[1]))

        return handlers

    def parse(self, file_location, data_type = None):
        if type(data_type) == type(None):
            print True
            handlers = self.extension_test(file_location)
        else:
            print False
            handlers = data_type

        print handlers

        if handlers == "Kv":
            handle = PyPWA.data.filehandling.Kv()
            return handle.parse(file_location)

    def write(self, file_location, data, data_type = None):
        if type(data_type) == type(None):
            handlers = self.extension_test(file_location)
        else:
            handlers = data_type

        if handlers == "Kv":
            handle = PyPWA.data.filehandling.Kv()
            handle.data = data
            handle.write(file_location)



