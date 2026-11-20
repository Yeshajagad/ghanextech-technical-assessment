"""A safe calculator.

The expression is parsed into a tree and only a short list of node types is
allowed. Nothing is ever passed to eval(), so the model (or a user) cannot
run code through it.
"""

import ast
import math
import operator

from agent.tools.base import Tool, ToolError

MAX_LENGTH = 200  # characters in the expression
MAX_POWER_BITS = 10_000  # stops things like 9**9**9 from eating memory

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "abs": abs,
    "round": round,
}
CONSTANTS = {"pi": math.pi, "e": math.e}

NOT_ALLOWED = (
    "Only numbers, + - * / // % ** ^, parentheses, pi, e and the functions "
    "sqrt, sin, cos, tan, log, log10, abs, round are allowed."
)


def _evaluate(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
    elif isinstance(node, ast.Name) and node.id in CONSTANTS:
        return CONSTANTS[node.id]
    elif isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
        return UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    elif isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and isinstance(left, int) and isinstance(right, int):
            if abs(right) * max(1, abs(left).bit_length()) > MAX_POWER_BITS:
                raise ToolError("That power is too large to calculate.")
        return OPERATORS[type(node.op)](left, right)
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in FUNCTIONS
        and not node.keywords
    ):
        return FUNCTIONS[node.func.id](*[_evaluate(arg) for arg in node.args])
    raise ToolError(NOT_ALLOWED)


def _format(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ToolError("The result is too large or undefined.")
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.10g}"
    return str(value)


def calculate(expression):
    expression = expression.strip()
    if not expression:
        raise ToolError("The expression is empty.")
    if len(expression) > MAX_LENGTH:
        raise ToolError(f"The expression is too long (the limit is {MAX_LENGTH} characters).")

    try:
        tree = ast.parse(expression.replace("^", "**"), mode="eval")
    except SyntaxError:
        raise ToolError("That is not a valid math expression.")

    try:
        result = _evaluate(tree.body)
    except ZeroDivisionError:
        raise ToolError("Division by zero is not possible.")
    except (ValueError, TypeError, OverflowError):
        raise ToolError("The numbers are not valid for that operation.")
    return f"{expression} = {_format(result)}"


TOOL = Tool(
    name="calculator",
    description=(
        "Evaluate a math expression and return the exact result. Use it for ANY "
        "arithmetic, percentages, powers or conversions instead of calculating yourself. "
        "Supports + - * / // % ** parentheses, pi, e and sqrt, sin, cos, tan, log, log10, abs, round."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The expression to evaluate, for example '(25 * 4) / 8' or 'sqrt(144) + 2**5'.",
            }
        },
        "required": ["expression"],
    },
    func=calculate,
)