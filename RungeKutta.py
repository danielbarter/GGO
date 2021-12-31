import sympy as sp
import math

f = sp.Function('f')
y = sp.Function('y')
t = sp.Symbol('t')
T = sp.Symbol('T')

def symbolic_derivative(function, degree):
    return sp.Function(
        function.name + "^(" + str(degree) + ")")

def higher_differential(function, argument, displacement, degree, shift=0):
    accumulator = 0
    for i in range(1, degree + 1):
        accumulator += (
            symbolic_derivative(function, i + shift)(argument) * (displacement ** i) /
            math.factorial(i))

    return accumulator


def ode_taylor_expansion(
        function,
        solution,
        time,
        time_displacement,
        degree):

    argument = sp.Symbol('x')
    displacement = sp.Symbol('X')

    function_taylor = higher_differential(
        function,
        argument,
        displacement,
        degree)

    solution_taylor = higher_differential(
        solution,
        time,
        time_displacement,
        degree)

    rhs = function_taylor.subs([
        (displacement, solution_taylor),
        (argument, solution(time))
    ])

    lhs = higher_differential(
        solution,
        time,
        time_displacement,
        degree,
        shift=1)

    rhs_coefficients =  sp.Poly(rhs,time_displacement).coeffs()
    rhs_coefficients.reverse()

    lhs_coefficients = sp.Poly(lhs, time_displacement).coeffs()
    lhs_coefficients.reverse()

    substitutions = [(symbolic_derivative(y,1)(time), function(solution(time)))]
    for i in range(degree):
        factor = math.factorial(i + 1)
        l = lhs_coefficients[i] * factor
        r = rhs_coefficients[i].subs(substitutions) * factor
        substitutions.append((l,r))


    return solution_taylor.subs(substitutions)
