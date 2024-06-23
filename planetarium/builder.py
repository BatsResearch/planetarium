import typing

from collections.abc import Iterable

from planetarium.graph import ProblemGraph

from pddl.core import And, Problem
from pddl.logic.predicates import Predicate
from pddl.logic.terms import Constant
from pddl.parser.problem import LenientProblemParser


def _constant_to_dict(constant: Constant) -> dict[str, typing.Any]:
    """
    Convert a PDDL Constant object to a dictionary representation.

    Parameters:
        constant (Constant): The PDDL Constant object.

    Returns:
        dict: A dictionary containing the constant name and typing information.
    """
    return {
        "name": constant.name,
        "typing": set(constant.type_tags),
    }


def _predicate_to_dict(predicate: Predicate) -> dict[str, typing.Any]:
    """
    Convert a PDDL Predicate object to a dictionary representation.

    Parameters:
        predicate (Predicate): The PDDL Predicate object.

    Returns:
        dict: A dictionary containing the predicate name and its parameter names.
    """
    return {
        "typing": predicate.name,
        "parameters": [constant.name for constant in predicate.terms],
    }


def _build_constants(constants: Iterable[Constant]) -> list[dict[str, typing.Any]]:
    """
    Build a list of dictionaries representing PDDL constants.

    Parameters:
        constants (Iterable[Constant]): An iterable of PDDL Constant objects.

    Returns:
        list: A list of dictionaries containing constant information.
    """
    return [_constant_to_dict(constant) for constant in constants]


def _build_predicates(
    predicates: Iterable[Predicate],
) -> list[dict[str, typing.Any]]:
    """
    Build a list of dictionaries representing PDDL predicates.

    Parameters:
        predicates (Iterable[Predicate]): An iterable of PDDL Predicate objects.

    Returns:
        list: A list of dictionaries containing predicate information.
    """
    return [_predicate_to_dict(predicate) for predicate in predicates]


def build(problem: str) -> ProblemGraph:
    """
    Build scene graphs from a PDDL problem description.

    Parameters:
        problem (str): A string containing the PDDL problem description.

    Returns:
        tuple: Two SceneGraph instances representing the initial state and goal state.
    """
    problem: Problem = LenientProblemParser()(problem)

    if isinstance(problem.goal, Predicate):
        goal = [problem.goal]
    elif isinstance(problem.goal, And):
        goal = problem.goal.operands
    else:
        raise ValueError(f"Unsupported goal type: {type(problem.goal)}")

    return ProblemGraph(
        _build_constants(problem.objects),
        _build_predicates(problem.init),
        _build_predicates(goal),
        domain=problem.domain_name,
    )
