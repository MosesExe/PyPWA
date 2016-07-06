#    PyPWA, a scientific analysis toolkit.
#    Copyright (C) 2016  JLab
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Handles EVIL to / from memory.

The objects in this file are dedicated to reading the EVIL files from disk
and into memory. This file type is being depreciated for many reasons, and
will live here until it shrivels away, is completely forgotten, and dies.

EVIL, Expanded Variable Identification Lists, earned their name from their
inefficient nature when it comes to reading in, writing out, or simply
existing, its a name given to these EVIL formats out of a mixture of spite
and love by current and former developers alike.

This format exists currently only as backwards compatibility, and may not
be bug free or entirely optimized, and may never be. If you are a user
trying to figure out what you should export your data to, or a developer
trying to learn the nature of data within PyPWA, you should move your
attention to CSV/TSV in the SV object and forget that this ever existed.
"""

import io
import logging

import numpy

from PyPWA.libs.data import definitions
from PyPWA import VERSION, LICENSE, STATUS

__author__ = ["Mark Jones"]
__credits__ = ["Mark Jones"]
__maintainer__ = ["Mark Jones"]
__email__ = "maj@jlab.org"
__status__ = STATUS
__license__ = LICENSE
__version__ = VERSION


class KvInterface(definitions.TemplateMemory):

    def parse(self, file_location):
        raise NotImplementedError()

    def write(self, file_location, data):
        raise NotImplementedError()

    @staticmethod
    def file_length(file_location):
        """
        Determines the number of lines in the file.

        Args:
            file_location (str): The file to check for line _count.

        Returns:
            int: The number of lines.
        """
        with io.open(file_location) as stream:
            for file_length, line in enumerate(stream):
                pass
        return file_length + 1


class DictOfArrays(KvInterface):
    """
    Handles old Kv format
    """

    def _create_empty_array(self, file_location):

        file_length = self.file_length(file_location)

        with open(file_location) as stream:
            first_line = stream.readline()

        names = []

        for x in range(len(first_line.split(","))):
            names.append(first_line.split(",")[x].split("=")[0])

        types = []
        for name in names:
            types.append((name, "f8"))

        return numpy.zeros(file_length, types)

    def parse(self, file_location):
        """
        Loads Kv data into memory

        Args:
            file_location (str): Path of file

        Returns:
            numpy.ndarray: A structured array of the data.
        """
        data = self._create_empty_array(file_location)

        # This is ugly, don't look.
        with io.open(file_location) as stream:
            for index, line in enumerate(stream):
                for particle_count in range(len(line.split(","))):

                    particle_name = \
                        line.split(",")[particle_count].split("=")[0]

                    particle_value = \
                        numpy.float64(
                            line.strip("\n").split(",")[
                                particle_count].split("=")[1]
                        )

                    data[particle_name][index] = particle_value

        return data

    def write(self, file_location, data):
        """
        Writes Classic Kvs to file

        Args:
            file_location (str): path to file
            data (numpy.ndarray): dict of numpy arrays
        """

        kinematic_variables = list(data.dtype.names)

        with open(file_location, "w") as stream:
            for event in range(len(data[kinematic_variables[0]])):
                line = ""
                for kinematic_variable in range(len(kinematic_variables)):
                    if kinematic_variable > 0:
                        line += ","
                    line += "{0}={1}".format(
                        kinematic_variables[kinematic_variable],
                        repr(data[
                            kinematic_variables[kinematic_variable]
                        ][event])
                    )
                line += "\n"
                stream.write(line)


class ListOfFloats(KvInterface):
    """
    Handles QFactor list parsing
    """

    def parse(self, file_location):
        """
        Parses a list of factors

        Args:
            file_location (str): The path to file

        Returns:
            numpy.ndarray: Array of factors
        """

        file_length = self.file_length(file_location)

        parsed = numpy.zeros(file_length, "f8")

        with io.open(file_location) as stream:
            for count, line in enumerate(stream):
                parsed[count] = line.strip("\n")
        return parsed

    def write(self, file_location, data):
        """
        Writes Arrays to disk as floats

        Args:
            file_location (str): Path to file
            data (numpy.ndarray): Data to be written to disk
        """
        with open(file_location, "w") as stream:
            for event in data:
                stream.write(repr(event) + "\n")


class ListOfBooleans(KvInterface):
    """
    Classic boolean per line data type
    """

    def parse(self, file_location):
        """
        Parses list of booleans into numpy array.

        Args:
            file_location (str): Path to file

        Returns:
            numpy.ndarray: Bool array of weights
        """

        file_length = self.file_length(file_location)

        weights = numpy.zeros(shape=file_length, dtype=bool)

        with io.open(file_location) as stream:
            for index, weight in enumerate(stream):
                weights[index] = int(weight)

        return weights

    def write(self, file_location, data):
        """
        Writes booleans to text file with each weight on a new line

        Args:
            file_location (str): Path to file
            data (numpy.ndarray): Array of booleans
        """
        with open(file_location, "w") as stream:
            for weight in data:
                stream.write(repr(int(weight)) + "\n")


class SomewhatIntelligentSelector(KvInterface):

    def __init__(self):
        """
        Attempts to select the write object to load and write Expanded
        Variable Identification Lists to and from the disk. It does this
        by examining the EVIL data and using its types to select the EVIL
        object.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.addHandler(logging.NullHandler())

    def parse(self, file_location):
        """
        Reads in EVIL format from disk, searches the first line of data
        for clues as to the data type. If there are = or , in the first
        line it assumes its a list of dict, if . then float, and if none
        of the above pure bool.

        If it doesn't work, perhaps use CSV?
        Args:
            file_location (str): The location of the file that needs to be
                read in from the disk.

        Returns:
            numpy.ndarray:  The data that was parsed from the disk.
        """
        validator = EVILValidator(file_location)
        validator.test()
        if validator.evil_type == "DictOfArrays":
            parser = DictOfArrays()
        elif validator.evil_type == "ListOfFloats":
            parser = ListOfFloats()
        elif validator.evil_type == "ListOfBools":
            parser = ListOfBooleans()
        else:
            raise RuntimeError("How did you even break this?")

        return parser.parse(file_location)

    def write(self, file_location, data):
        """
        Writes EVIL data types to disk, detects the data in the same way
        that parse works, however does it by running the type check
        against the object that was received.

        Args:
            file_location (str): Where to write the data.
            data numpy.ndarray: The data that needs to be written to disk.
        """
        if isinstance(data, tuple):
            self._logger.debug("Found type tuple, assuming GenericEvent.")
            writer = DictOfArrays()
        elif isinstance(data[0], numpy.float64):
            self._logger.debug("Found type float64, assuming float list.")
            writer = ListOfFloats()
        elif isinstance(data[0], numpy.bool_):
            self._logger.debug("Found type bool, assuming bool list.")
            writer = ListOfBooleans()
        else:
            string = "Data type not expected! Found data type {0}".format(
                type(data)
            )
            try:
                string += " or type {0}".format(type(data[0]))
            except IndexError:
                pass
            string += "!!"

            self._logger.error(string)
            raise RuntimeError(string)
        writer.write(file_location, data)


class EVILReader(definitions.TemplateReader):

    def __init__(self, file_location):
        """
        Reads in the EVIL Type one event at a time.

        Args:
            file_location (str): The location of the EVIL file.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.addHandler(logging.NullHandler())

        super(EVILReader, self).__init__(file_location)
        self._previous_event = None
        self._file = False  # type: io.TextIOBase
        self._parameters = False  # type: [str]
        self._file_data_type = False  # type: str

        self._start_input()

    def _start_input(self):
        """
        This file completely resets the the file handler if it exists and
        creates it otherwise.
        """
        if self._file:
            self._file.close()

        self._file = io.open(self._the_file, "rt")

        if not isinstance(self._file_data_type, str):
            self._set_data_type()
        if not isinstance(self._parameters, list):
            self._build_params()

    def _build_params(self):
        """
        Searches for the parameters in the file then sets them to
        self._parameters.
        """
        first_line = self._file.readline()

        if self._file_data_type == "DictOfArrays":
            self._parameters = []
            for parameter in first_line.split(","):
                self._parameters.append(parameter.split("=")[0])

        elif self._file_data_type == "ListOfBools":
            self._parameters = ["rejection_list"]

        elif self._file_data_type == "ListOfFloats":
            self._parameters = ["QFactor"]

        self._file.seek(0)

    def _set_data_type(self):
        """
        Sets self._file_data_type using the validator object. Mostly
        Accurate.
        """
        validator = EVILValidator(self._the_file)
        validator.test()
        self._file_data_type = validator.evil_type

    def reset(self):
        """
        Wrapper for _start_input
        """
        self._start_input()

    @property
    def next_event(self):
        """
        Reads in a single line and parses the line into a GenericEvent.

        Returns:
            numpy.ndarray: The values of the array.
        """
        if self._file_data_type == "DictOfArrays":
            values = self._read_dict()
        elif self._file_data_type == "ListOfBools":
            values = self._read_bool()
        else:
            values = self._read_float()

        self._previous_event = values
        return self._previous_event

    @property
    def previous_event(self):
        return self.previous_event

    def __read(self):
        """
        Reads a single line from the file and removes the spaces and
        newline.

        Raises:
            StopIteration: Raised when there is no data left in the file.

        Returns:
            str: The read in line without spaces and newlines.
        """
        string = self._file.readline().strip("\n").strip(" ")
        if string == "":
            raise StopIteration
        return string

    def _read_bool(self):
        """
        Reads a single line and returns the bool value from that line.

        Returns:
            numpy.ndarray: True or False depending on the value of the
                line that was read.
        """
        x = numpy.zeros(1, bool)
        x[0] = bool(self.__read())
        return x

    def _read_float(self):
        """
        Reads a single line and returns the float value from the line.

        Returns:
            numpy.ndarray: The value read in from the file.
        """
        x = numpy.zeros(1, "f8")
        x[0] = numpy.float64(self.__read())
        return x

    def _read_dict(self):
        """
        Reads a single line and returns the list of the values rendered
        from the file.

        Returns:
            numpy.ndarray: The values read in from the file.
        """
        the_line = self.__read()
        self._logger.debug("Found: " + the_line)

        names = []
        for variable in the_line.split(","):
            names.append(variable.split("=")[0])

        types = []
        for name in names:
            types.append((name, "f8"))

        final = numpy.zeros(1, types)

        for variable in the_line.split(","):
            value = numpy.float64(variable.split("=")[1])
            name = variable.split("=")[0]
            final[name][0] = value

        return final

    def close(self):
        self._file.close()


class EVILWriter(definitions.TemplateWriter):

    def __init__(self, file_location):
        """
        Single event writer for EVIL data types. Writes a single event at
        a time and leaves the file handle open until its explicitly closed
        by the developer or user.

        Args:
            file_location (str): Where to write the data.
        """
        super(EVILWriter, self).__init__(file_location)
        self._file = io.open(file_location, "wt")

    def write(self, data):
        """
        Writes a single event to file at a time.

        Args:
            data (numpy.ndarray): The array that contains the data to be
                writen to the file.
        """
        key_count = len(list(data.dtype.names)) - 1
        string = ""
        for index, key in enumerate(list(data.dtype.names)):
            if not index == 0 and not index == key_count:
                string += ","
                string += repr(key) + "=" + repr(data[key])
        string += "\n"

        self._file.write(string)

    def close(self):
        """
        Closes the file safely.
        """
        self._file.close()


class EVILValidator(definitions.TemplateValidator):

    def __init__(self, file_location, full=False):
        """
        This attempts to validate the files to see if it can be read in by
        this plugin.

        Args:
            file_location (str): The location of the file.
            full (Optional[bool]): Whether or not to do a full test of the
                file.
        """
        super(EVILValidator, self).__init__(file_location, full)
        self._the_file = io.open(file_location)

    def _check_data_type(self):
        """
        Performs a really simple test to see if its a support format.

        Raises:
            PyPWA.libs.data.exceptions.IncompatibleData:
                Raised when the test fails to find a supported format.
        """
        test_data = self._the_file.read(100).split("\n")[0]
        if "=" in test_data and "," in test_data:
            self._evil_type = "DictOfArrays"
        elif "." in test_data and len(test_data) > 1:
            self._evil_type = "ListOfFloats"
        elif len(test_data) == 1:
            self._evil_type = "ListOfBools"
        else:
            raise definitions.IncompatibleData("Failed to find a data")

    def test(self):
        """
        Runs the various tests included tests.
        """
        self._check_data_type()

    @property
    def evil_type(self):
        """
        Property that returns the data type that was detected.

        Returns:
            str: The type of data the validator detected during its tests.
        """
        try:
            return self._evil_type
        except AttributeError:
            try:
                self._check_data_type()
            except definitions.IncompatibleData:
                raise IOError(
                    "Data is not of EVIL Type, "
                    "double check and try again."
                )
            return self._evil_type


metadata_data = {
    "name": "EVIL",
    "extension": "txt",
    "validator": EVILValidator,
    "reader": EVILReader,
    "writer": EVILWriter,
    "memory": SomewhatIntelligentSelector
}