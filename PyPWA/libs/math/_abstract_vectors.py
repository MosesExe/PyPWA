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
The backbone of all vectors
---------------------------
This is the base abstract class for the vectors inside PyPWA, this is what
allows for them to split, be iterated over, compared, accessed like a
dictionary, and have properties. It even contains some mathematics and error
checking that *should* be universal for all vectors.
"""

from typing import List, Union

import numpy

from PyPWA import AUTHOR, VERSION

__credits__ = ["Mark Jones", "Stephanie Bramlett"]
__author__ = AUTHOR
__version__ = VERSION


class AbstractVector(object):

    def __init__(self, array, vector_type):
        # type: (numpy.ndarray, type(self)) -> None
        self._vector = array
        self.__vector_type = vector_type
        self.__count = 0

    def __repr__(self):
        # type: () -> str
        raise NotImplementedError

    def __str__(self):
        # type: () -> str
        return str(self._vector)

    def __add__(self, vector):
        # type: (type(self)) -> type(self)
        new_vector = numpy.zeros(len(self), self._vector.dtype)
        for name in self._vector.dtype.names:
            new_vector[name] = self[name] + vector[name]
        return self.__vector_type(new_vector)

    def __sub__(self, vector):
        # type: (type(self)) -> type(self)
        new_vector = numpy.zeros(len(self), self._vector.dtype)
        for name in self._vector.dtype.names:
            new_vector[name] = self[name] - vector[name]
        return self.__vector_type(new_vector)

    def __mul__(self, vector):
        # type: (Union[float, type(self)]) -> type(self)
        if isinstance(vector, self.__vector_type):
            return self._vector_multiplication(vector)
        else:
            return self.__multiply_by_scalar(vector)

    def __multiply_by_scalar(self, scalar):
        # type: (float) -> numpy.ndarray
        new_vector = numpy.zeros(len(self), self._vector.dtype)
        for name in self._vector.dtype.names:
            new_vector[name] = self[name] * scalar
        return self.__vector_type(new_vector)

    def _vector_multiplication(self, vector):
        # type: (type(self)) -> type(self)
        raise NotImplementedError

    def __len__(self):
        # type: () -> int
        return len(self._vector)

    def __iter__(self):
        return self

    def __next__(self):
        # type: () -> type(self)
        return self.next()

    def next(self):
        self.__count += 1
        try:
            return self.__vector_type(self._vector[self.__count - 1])
        except IndexError:
            raise StopIteration

    def __getitem__(self, item):
        # type: (str) -> numpy.ndarray
        if item.lower() in self._vector.dtype.names:
            return self._vector[item.lower()].copy()
        else:
            raise ValueError(
                "Vector has components %s!" % self._vector.dtype.names
            )

    def __setitem__(self, key, value):
        # type: (str, Union[float, numpy.ndarray]) -> None
        if key.lower() in self._vector.dtype.names:
            self.__set_new_values(key, value)
        else:
            raise ValueError(
                "Vector has components %s!" % self._vector.dtype.names
            )

    def __set_new_values(self, key, value):
        # type: (str, Union[float, numpy.ndarray]) -> None
        if isinstance(value, numpy.ndarray):
            self.__set_numpy_array(key, value)
        else:
            self.__set_single_value(key, value)

    def __set_numpy_array(self, key, value):
        # type: (str, numpy.ndarray) -> None
        if len(value) == len(self):
            self._vector[key] = value
        else:
            raise ValueError(
                "New array must have same number of events as old array!"
            )

    def __set_single_value(self, key, value):
        # type: (str, float) -> None
        self._vector['key'] = numpy.full(len(self), value)

    def __eq__(self, vector):
        # type: (type(self)) -> numpy.ndarray
        # vector == vector
        return self._vector == vector.get_array()

    def __ne__(self, vector):
        # type: (type(self)) -> numpy.ndarray
        # vector != vector
        return self._vector != vector.get_array()

    def __bool__(self):
        # type: (type(self)) -> numpy.ndarray
        # bool(vector)
        return self._vector.astype(bool)

    def get_array(self):
        # type: () -> numpy.ndarray
        return self._vector

    def split(self, number):
        # type: (int) -> List[type(self)]
        split_vectors = []
        for new_array in numpy.array_split(self._vector, number):
            split_vectors.append(self.__vector_type(new_array))
        return split_vectors

    def get_dot(self, vector):
        # type: (type(self)) -> type(self)
        if isinstance(vector, self.__vector_type):
            return self._dot_product(vector)
        else:
            raise ValueError("Can only dot product vectors of the same type!")

    def _dot_product(self, vector):
        # type: (type(self)) -> type(self)
        raise NotImplementedError

    @property
    def length(self):
        # type: () -> numpy.ndarray
        return numpy.sqrt(self.x**2 + self.y**2 + self.z**2)

    @property
    def length_squared(self):
        # type: () -> numpy.ndarray
        raise NotImplementedError

    @property
    def theta(self):
        # type: () -> numpy.ndarray
        return numpy.arccos(self.cos_theta)

    @property
    def phi(self):
        # type: () -> numpy.ndarray
        return numpy.arctan2(self.y, self.x)

    @property
    def sin_theta(self):
        # type: () -> numpy.ndarray
        return (self.x**2 + self.y**2) / self.length

    @property
    def cos_theta(self):
        # type: () -> numpy.ndarray
        return self.z / self.length

    @property
    def x(self):
        # type: () -> numpy.ndarray
        return self['x']

    @x.setter
    def x(self, value):
        # type: (Union[float, numpy.ndarray]) -> None
        self['x'] = value

    @property
    def y(self):
        # type: () -> numpy.ndarray
        return self['y']

    @y.setter
    def y(self, value):
        # type: (Union[float, numpy.ndarray]) -> None
        self['y'] = value

    @property
    def z(self):
        # type: () -> numpy.ndarray
        return self['z']

    @z.setter
    def z(self, value):
        # type: (Union[float, numpy.ndarray]) -> None
        self['z'] = value
