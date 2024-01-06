from matrix_operations import (
    add_matrix_matrix,
    apply_function_matrix,
    multiply_matrix_matrix,
)


def test_multiply_matrix_matrix():
    assert multiply_matrix_matrix([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [
        [19, 22],
        [43, 50],
    ]
    assert multiply_matrix_matrix(
        [[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]
    ) == [[58, 64], [139, 154]]
    assert multiply_matrix_matrix([[1, 2, 3], [4, 5, 6]], [[7], [8], [9]]) == [
        [50],
        [122],
    ]


def test_add_matrix_matrix():
    assert add_matrix_matrix([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[6, 8], [10, 12]]
    assert add_matrix_matrix([[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]) == [
        [8, 10, 12],
        [14, 16, 18],
    ]
    assert add_matrix_matrix([[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]) == [
        [8, 10, 12],
        [14, 16, 18],
    ]


def test_apply_function_matrix():
    assert apply_function_matrix([[1, 2], [3, 4]], lambda x: x * 2) == [[2, 4], [6, 8]]
    assert apply_function_matrix([[1, 2, 3], [4, 5, 6]], lambda x: x * 3) == [
        [3, 6, 9],
        [12, 15, 18],
    ]
    assert apply_function_matrix([[1, 2, 3], [4, 5, 6]], lambda x: x * 4) == [
        [4, 8, 12],
        [16, 20, 24],
    ]
