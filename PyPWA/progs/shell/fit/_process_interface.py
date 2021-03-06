#  coding=utf-8
#
#  PyPWA, a scientific analysis toolkit.
#  Copyright (C) 2016 JLab
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Processing and Output
---------------------
This is where the processing interface is defined along with the automated
output from the the parallel thread.

- _ThreadInterface - The object between the main thread and the output thread.

- _OutputThread - The actual output thread, its started with each call to the
  likelihood. Has a 1hz output rate.

- FittingInterface - The interface between the Likelihood Kernels and the
  optimizer module.
"""

from __future__ import print_function

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

import logging
import threading
import time

import numpy

from PyPWA import AUTHOR, VERSION
from PyPWA.libs.interfaces import kernel

__credits__ = ["Mark Jones"]
__author__ = AUTHOR
__version__ = VERSION


class _ThreadInterface(object):

    __LOGGER = logging.getLogger(__name__ + "_ThreadInterface")

    def __init__(self):
        self.__root_logger = logging.getLogger()
        self.__send_queue = Queue()
        self.__receive_queue = Queue()
        self.__times = []
        self.__initial_time = time.time()

        if not self.__root_logger.isEnabledFor(logging.INFO):
            self.__enabled = True
        else:
            self.__enabled = False
            self.__LOGGER.info(
                "Processor Output is disabled while info logging is enabled"
            )

    def start(self, last_value):
        if self.__enabled:
            thread = _OutputThread(
                self.__receive_queue, self.__send_queue,
                last_value, self.__get_last_time(), self.__average_time(),
                self.__initial_time
            )

            thread.start()

    def __get_last_time(self):
        if len(self.__times) == 0:
            return 0
        else:
            return self.__times[-1]

    def __average_time(self):
        if len(self.__times) == 0:
            return 0
        else:
            return sum(self.__times) / len(self.__times)

    def stop(self):
        if self.__enabled:
            self.__send_queue.put("die")
            self.__times.append(float(self.__receive_queue.get()))


class _OutputThread(threading.Thread):

    def __init__(
            self,
            send_queue,  # type: Queue
            receive_queue, # type: Queue
            last_value,  # type: numpy.float64
            last_time,  # type: float
            average_time,  # type: float
            start_time
    ):
        # type: (...) -> None
        super(_OutputThread, self).__init__()
        self.__send_queue = send_queue
        self.__receive_queue = receive_queue
        self.__last_value = last_value
        self.__last_time = last_time
        self.__average_time = average_time
        self.__start_time = start_time
        self.__output_pulse = "-"
        self.__index = None
        self.__initial_time = time.time()
        self.daemon = True

    def run(self):
        while True:
            if not self.__receive_queue.empty():
                self.__send_runtime()
                break
            self.__output()

    def __output(self):
        print("\r" + self.__create_output(), end="\r")

    def __create_output(self):
        if isinstance(self.__last_value, type(None)):
            return self.__simple_output()
        else:
            return self.__full_output()

    def __simple_output(self):
        self.__pulse()
        runtime = self.__get_current_runtime()
        return "Elapsed time: {0: .2f} {1}".format(
            runtime, self.__output_pulse
        )

    def __full_output(self):
        self.__pulse()
        runtime = self.__get_current_runtime()
        return "Last Value: {0: .3f}, Average Time: {1: .2f}, " \
               "Elapsed Time: {2: .2f}, " \
               "Total Runtime {3: .2f} {4}".format(
                 self.__last_value, self.__average_time,
                 runtime, self.__get_total_runtime(), self.__output_pulse
               )

    def __get_current_runtime(self):
        return time.time() - self.__initial_time

    def __get_total_runtime(self):
        return time.time() - self.__start_time

    def __pulse(self):
        if self.__output_pulse is "-":
            self.__output_pulse = "/"
        elif self.__output_pulse is "/":
            self.__output_pulse = "\\"
        elif self.__output_pulse is "\\":
            self.__output_pulse = "-"

    def __send_runtime(self):
        self.__receive_queue.get()
        self.__send_queue.put(self.__get_current_runtime())


class FittingInterface(kernel.KernelInterface):

    IS_DUPLEX = True
    __LOGGER = logging.getLogger(__name__ + ".FittingInterfaceKernel")

    def __init__(self, minimizer_function):
        self.__parameter_parser = minimizer_function
        self.__thread_interface = _ThreadInterface()
        self.__last_value = None  # type: numpy.float64

    def run(self, communication, *args):
        self.__send_arguments(communication, args)
        self.__thread_interface.start(self.__last_value)
        self.__get_final_value(communication)
        self.__thread_interface.stop()
        self.__log_final_value()
        return self.__last_value

    def __send_arguments(self, communication, args):
        parsed_arguments = self.__parameter_parser.convert(args)

        for pipe in communication:
            pipe.send(parsed_arguments)

    def __get_final_value(self, communication):
        values = numpy.zeros(shape=len(communication))
        for index, pipe in enumerate(communication):
            values[index] = pipe.recv()
        self.__last_value = numpy.sum(values)

    def __log_final_value(self):
        self.__LOGGER.info("Final Value is: %f15" % self.__last_value)
