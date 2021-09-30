from itertools import combinations, permutations
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

        # we represent a distance invariant function as a tuple (i,j)
        # where i < j. we represent an angle invariant function as s
        # tuble (base,i,j) where i < j

        self.distance_expression = {}
        for (i,j) in combinations(range(M), 2):
            displacement_vector = (
                sp.Matrix([self.cartesian_coordinate_symbol[i]]) -
                sp.Matrix([self.cartesian_coordinate_symbol[j]])
            )

            self.distance_expression[(i,j)] = displacement_vector.dot(
                displacement_vector)



        self.angle_expression = {}
        for (base, i, j) in permutations(range(M), 3):
            if i > j:
                continue

            displacement_vector_1 = (
                sp.Matrix([self.cartesian_coordinate_symbol[i]]) -
                sp.Matrix([self.cartesian_coordinate_symbol[base]])
            )

            displacement_vector_2 = (
                sp.Matrix([self.cartesian_coordinate_symbol[j]]) -
                sp.Matrix([self.cartesian_coordinate_symbol[base]])
            )


            self.angle_expression[(base,i,j)] = displacement_vector_1.dot(
                displacement_vector_2)


        self.distance_differential = {}
        self.angle_differential = {}


        for distance_function in self.distance_expression:
            differential = []
            for i in self.cartesian_coordinate_symbol:
                for symbol in self.cartesian_coordinate_symbol[i]:
                    differential.append(sp.diff(
                        self.distance_expression[distance_function],
                        symbol))

            self.distance_differential[distance_function] = differential



        for angle_function in self.angle_expression:
            differential = []
            for i in self.cartesian_coordinate_symbol:
                for symbol in self.cartesian_coordinate_symbol[i]:
                    differential.append(sp.diff(
                        self.angle_expression[angle_function],
                        symbol))

            self.angle_differential[angle_function] = differential

