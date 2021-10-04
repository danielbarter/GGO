from multiprocessing import Pool
import sympy as sp
import numpy as np


# exterior derivative
d = sp.Function('d')

def chunks(lst, n):
    """iterator returning successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def matrix_map(matrix, func, num_threads=8):
    (m,n) = matrix.shape
    entry_list = []
    for i in range(m):
        for j in range(n):
            entry_list.append(matrix[i,j])


    with Pool(min(m * n, num_threads)) as p:
        entry_list_simplified = p.map(func, entry_list)

    return sp.Matrix(list(chunks(entry_list_simplified, n)))


def simplify(e):
    return e.simplify()


def maurer_cartan_constructor():
    rotation_matrix_coordinate_list = [
        [sp.Symbol('g_0^0'), sp.Symbol('g_1^0'), sp.Symbol('g_2^0')],
        [sp.Symbol('g_0^1'), sp.Symbol('g_1^1'), sp.Symbol('g_2^1')],
        [sp.Symbol('g_0^2'), sp.Symbol('g_1^2'), sp.Symbol('g_2^2')]]

    rotation_matrix_coordinate = sp.Matrix(rotation_matrix_coordinate_list)

    rotation_matrix_coordinate_differential = sp.Matrix(
        [[d(symbol) for symbol in row]
         for row in rotation_matrix_coordinate_list]
    )

    translation_matrix_coordinate_list = [
        sp.Symbol('z^0'), sp.Symbol('z^1'), sp.Symbol('z^2')]

    translation_matrix_coordinate = sp.transpose(sp.Matrix(
        [translation_matrix_coordinate_list]))

    translation_matrix_coordinate_differential = sp.transpose(sp.Matrix([
        [d(translation_matrix_coordinate[i])
         for i in range(3)]]))


    euclidean_matrix_coordinate = (
        rotation_matrix_coordinate.row_join(
        translation_matrix_coordinate).col_join(
            sp.Matrix([[0,0,0,1]])))

    euclidean_matrix_coordinate_differential = (
        rotation_matrix_coordinate_differential.row_join(
        translation_matrix_coordinate_differential).col_join(
        sp.Matrix([[0,0,0,0]])))

    euclidean_matrix_coordinate_inverse = (
        sp.transpose(rotation_matrix_coordinate).row_join(
            - sp.transpose(rotation_matrix_coordinate) *
            translation_matrix_coordinate).col_join(
                sp.Matrix([[0,0,0,1]])))

    maurer_cartan_form = (
        euclidean_matrix_coordinate_inverse *
        euclidean_matrix_coordinate_differential )

    return (
        euclidean_matrix_coordinate,
        maurer_cartan_form)


(euclidean_matrix_coordinate, maurer_cartan_form) = maurer_cartan_constructor()


class ConfigurationSpace:
    """
    class representing a configuration space on M points:

    C = { x^0, x^1, ..., x^{M-1} in R3 where x^i != x^j }
    """

    def __init__(self, M):

        self.M = M
        self.cartesian_coordinate = {}

        for i in range(M):
            self.cartesian_coordinate[i] = [
                sp.Symbol('x_0^' + str(i)),
                sp.Symbol('x_1^' + str(i)),
                sp.Symbol('x_2^' + str(i))]


        # cache dicts
        self._distance_expression = {}
        self._angle_expression = {}
        self._moving_frame_expression = {}
        self._distance_differential = {}
        self._angle_differential = {}
        self._moving_frame_differential = {}


    def validate_distance_function(self, distance_function):
        (i,j) = distance_function

        if (i >= j or
            i >= self.M or
            j >= self.M
            ):
            raise Exception("not a valid distance function")


    def validate_angle_function(self, angle_function):
        (base, i, j) = angle_function

        if (len(set(angle_function)) < 3 or
            i >= j or
            base >= self.M or
            i >= self.M or
            j >= self.M
            ):
            raise Exception("not a valid angle function")



    def distance_expression(self, distance_function):
        """
        we represent a distance invariant function as a tuple (i,j)
        where i < j, i < M and j < M
        """

        self.validate_distance_function(distance_function)

        if distance_function not in self._distance_expression:

            (i,j) = distance_function

            displacement_vector = (
                sp.Matrix([self.cartesian_coordinate[i]]) -
                sp.Matrix([self.cartesian_coordinate[j]])
            )

            self._distance_expression[distance_function] = displacement_vector.dot(
                displacement_vector)

        return self._distance_expression[distance_function]


    def angle_expression(self, angle_function):
        """
        we represent an angle invariant function as a
        tuple (base,i,j) with no repeats where i < j,
        base < M, i < M and j < M
        """

        self.validate_angle_function(angle_function)

        if angle_function not in self._angle_expression:

            (base, i, j) = angle_function

            displacement_vector_1 = (
                sp.Matrix([self.cartesian_coordinate[i]]) -
                sp.Matrix([self.cartesian_coordinate[base]])
            )

            displacement_vector_2 = (
                sp.Matrix([self.cartesian_coordinate[j]]) -
                sp.Matrix([self.cartesian_coordinate[base]])
            )


            self._angle_expression[angle_function] = displacement_vector_1.dot(
                displacement_vector_2)

        return self._angle_expression[angle_function]


    def moving_frame_expression(self, angle_function):
        """
        moving frame is parameterized by an angle function
        """
        self.validate_angle_function(angle_function)

        if angle_function not in self._moving_frame_expression:
            (base, i, j) = angle_function

            base_vector = sp.transpose(sp.Matrix([self.cartesian_coordinate[base]]))

            displacement_vector_1 = (
                sp.Matrix([self.cartesian_coordinate[i]]) -
                sp.Matrix([self.cartesian_coordinate[base]])
            )

            displacement_vector_2 = (
                sp.Matrix([self.cartesian_coordinate[j]]) -
                sp.Matrix([self.cartesian_coordinate[base]])
            )


            column_1_unnormalized = displacement_vector_1
            column_2_unnormalized = displacement_vector_1.cross(displacement_vector_2)

            column_1 = column_1_unnormalized / sp.sqrt(
                column_1_unnormalized.dot(column_1_unnormalized))

            column_2 = column_2_unnormalized / sp.sqrt(
                column_2_unnormalized.dot(column_2_unnormalized))


            column_3 = column_1.cross(column_2)

            rotation_matrix = sp.transpose(
                sp.Matrix(
                    [column_1,
                     column_2,
                     column_3]))

            full_matrix = rotation_matrix.row_join(base_vector).col_join(
                sp.Matrix([[0,0,0,1]])
            )

            self._moving_frame_expression[angle_function] = matrix_map(
                full_matrix,
                simplify
            )

        return self._moving_frame_expression[angle_function]


    def exterior_derivative(self, expression):
        """
        compute exterior derivative in cartesian coordinates
       """
        accumulator = 0
        for i in self.cartesian_coordinate:
            for symbol, differential in zip(
                    self.cartesian_coordinate[i],
                    d(self.cartesian_coordinate[i])):

                accumulator += sp.diff(
                    expression,
                    symbol) * differential

        return accumulator


    def distance_differential(self, distance_function):

        if distance_function not in self._distance_differential:

            distance_expression = self.distance_expression(distance_function)

            self._distance_differential[distance_function] = self.exterior_derivative(
                distance_expression)

        return self._distance_differential[distance_function]


    def angle_differential(self, angle_function):

        if angle_function not in self._angle_differential:

            angle_expression = self.angle_expression(angle_function)

            self._angle_differential[angle_function] = self.exterior_derivative(
                angle_expression)

        return self._angle_differential[angle_function]


    def moving_frame_differential(self, angle_function):
        if angle_function not in self._moving_frame_differential:

            moving_frame = self.moving_frame_expression(angle_function)

            # this is a reference
            moving_frame_differential = maurer_cartan_form


            for i in range(3):
                for j in range(4):
                    symbol = euclidean_matrix_coordinate[i,j]

                    # the sympy subs method creates a copy of the expression
                    # before substituting. This is nice for us because we
                    # don't need to reconstruct the maurer-cartan form every time
                    # we call this method
                    moving_frame_differential = moving_frame_differential.subs(
                        d(symbol), self.exterior_derivative(moving_frame[i,j])
                    )


            for i in range(3):
                for j in range(4):
                    symbol = euclidean_matrix_coordinate[i,j]
                    moving_frame_differential = moving_frame_differential.subs(
                        symbol, moving_frame[i,j])

            self._moving_frame_differential[
                angle_function] = matrix_map(
                    moving_frame_differential,
                    simplify
                )



        return self._moving_frame_differential[angle_function]


