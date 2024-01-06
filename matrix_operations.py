"""
This module contains functions for matrix operations.
"""
from typing import Callable


def multiply_matrix_matrix(
    matrix1: list[list[float]], matrix2: list[list[float]]
) -> list[list[float]]:
    return [
        [
            sum(matrix1[i][k] * matrix2[k][j] for k in range(len(matrix2)))
            for j in range(len(matrix2[0]))
        ]
        for i in range(len(matrix1))
    ]


def add_matrix_matrix(
    matrix1: list[list[float]], matrix2: list[list[float]]
) -> list[list[float]]:
    return [
        [matrix1[i][j] + matrix2[i][j] for j in range(len(matrix1[0]))]
        for i in range(len(matrix1))
    ]


def apply_function_matrix(
    matrix: list[list[float]], function: Callable[[float], float]
) -> list[list[float]]:
    return [
        [function(matrix[i][j]) for j in range(len(matrix[0]))]
        for i in range(len(matrix))
    ]
