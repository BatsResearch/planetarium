import os

from pddl.parser.problem import LenientProblemParser
from pddl.formatter import problem_to_string

from planetarium import *


VALIDATE = os.getenv("VALIDATE", "Validate")


def evaluate(
    source_pddl_str: str,
    target_pddl_str: str,
    domain_str: str,
    is_placeholder: bool = False,
) -> tuple[bool, bool, bool]:
    """Evaluate two PDDL problem descriptions for equivalence.

    Args:
        source_pddl_str (str):
        target_pddl_str (str): The second problem PDDL string.
        domain_str (str): The domain PDDL string.
        is_placeholder (bool, optional): Whether or not to treat the ground truth
            as a "placeholder" description. Defaults to False.

    Returns:
        tuple: A tuple containing the following boolean elements:
            - parseable: Whether or not the PDDL string is parseable.
            - solveable: Whether or not the PDDL string is solveable.
            - equivalent: Whether or not the PDDL strings are equivalent.
    """
    parseable = False
    solveable = False
    equivalent = False

    source_graph = builder.build(source_pddl_str)

    try:
        target_graph = builder.build(target_pddl_str)
        parseable = True
        clean_pddl_str = problem_to_string(LenientProblemParser(target_pddl_str))

        solveable = downward.validate(
            builder.build(domain_str),
            clean_pddl_str,
            oracle.plan_to_string(oracle.plan(target_graph)),
            VALIDATE,
        )

        if source_graph == target_graph:
            equivalent = True
        elif source_graph.decompose()[0] != target_graph.decompose()[0]:
            equivalent = False
        else:
            equivalent = metric.equals(
                oracle.fully_specify(source_graph, return_reduced=True),
                oracle.fully_specify(target_graph, return_reduced=True),
                is_placeholder=is_placeholder,
            )
    except Exception:
        pass

    return parseable, solveable, equivalent
