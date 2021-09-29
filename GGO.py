import sympy as sp
import numpy as np


class ConfigurationSpace:
    """
    class representing a configuration space on M points:

    { x^0, x^1, ..., x^{M-1} in R3 where x^i != x^j }
    """

    def __init__(self, M):

        self.M = M

        self.cartesian_coordinate_symbol = {}

        for i in range(M):
            self.cartesian_coordinate_symbol[i] = [
                sp.Symbol('x_0^' + str(i)),
                sp.Symbol('x_1^' + str(i)),
                sp.Symbol('x_2^' + str(i))]


        self._distance_expression = {}
        self._angle_expression = {}


    def distance_expression(self, i, j):
        if i >= self.M or j >= self.M:
            raise Exception("indices out of range")

        if i == j:
            raise Exception("indices are equal")

        if j < i:
            arg = (j, i)
        else:
            arg = (i, j)

        if arg not in self._distance_expression:

            displacement_vector = (
                sp.Matrix([self.cartesian_coordinate_symbol[arg[1]]]) -
                sp.Matrix([self.cartesian_coordinate_symbol[arg[0]]])
            )

            self._distance_expression[arg] = displacement_vector.dot(
                displacement_vector)

        return self._distance_expression[arg]

