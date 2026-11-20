import pytest

from agent.tools.base import ToolError
from agent.tools.calculator import calculate


def test_basic_arithmetic_and_precedence():
    assert calculate("2 + 3 * 4") == "2 + 3 * 4 = 14"
    assert calculate("(2 + 3) * 4") == "(2 + 3) * 4 = 20"


def test_division_gives_a_clean_number():
    assert calculate("10 / 4") == "10 / 4 = 2.5"
    assert calculate("10 / 5") == "10 / 5 = 2"


def test_caret_means_power():
    assert calculate("2 ^ 10") == "2 ^ 10 = 1024"


def test_percentage_example():
    assert calculate("2450 * 18 / 100") == "2450 * 18 / 100 = 441"


def test_functions_and_constants():
    assert calculate("sqrt(144) + 2**5") == "sqrt(144) + 2**5 = 44"
    assert calculate("round(pi, 2)") == "round(pi, 2) = 3.14"
    assert calculate("abs(-7)") == "abs(-7) = 7"


def test_negative_numbers():
    assert calculate("-5 + 2") == "-5 + 2 = -3"


def test_long_decimals_are_shortened():
    assert calculate("1 / 3") == "1 / 3 = 0.3333333333"


def test_division_by_zero():
    with pytest.raises(ToolError, match="Division by zero"):
        calculate("1 / 0")


def test_square_root_of_a_negative_number():
    with pytest.raises(ToolError, match="not valid"):
        calculate("sqrt(-1)")


def test_wrong_function_arguments():
    with pytest.raises(ToolError, match="not valid"):
        calculate("round(2.5, 1.5)")


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "open('secret.txt')",
        "().__class__",
        "'a' * 3",
        "x + 1",
        "sqrt(4, key=1)",
        "[1, 2]",
        "True + 1",
        "lambda: 1",
    ],
)
def test_dangerous_or_unsupported_input_is_rejected(expression):
    with pytest.raises(ToolError, match="allowed"):
        calculate(expression)


def test_huge_power_is_refused():
    with pytest.raises(ToolError, match="too large"):
        calculate("9 ** 9 ** 9")


def test_huge_power_built_in_steps_is_refused():
    with pytest.raises(ToolError, match="too large"):
        calculate("(10 ** 1000) ** 1000")


def test_float_overflow_is_reported():
    with pytest.raises(ToolError):
        calculate("1e308 * 10")


def test_empty_and_long_expressions():
    with pytest.raises(ToolError, match="empty"):
        calculate("   ")
    with pytest.raises(ToolError, match="too long"):
        calculate("1+" * 150 + "1")


def test_syntax_error():
    with pytest.raises(ToolError, match="not a valid"):
        calculate("2 +")