import os

from pddl.parser.problem import LenientProblemParser
from pddl.formatter import problem_to_string

from planetarium import *


VALIDATE = os.getenv("VALIDATE", "Validate")
DOMAINS = {
    "blocksworld": """;; source: https://github.com/AI-Planning/pddl-generators/blob/main/blocksworld/domain.pddl
    ;; same as used in IPC 2023
    ;;
    (define (domain blocksworld)

    (:requirements :strips)

    (:predicates (clear ?x)
                (on-table ?x)
                (arm-empty)
                (holding ?x)
                (on ?x ?y))

    (:action pickup
    :parameters (?ob)
    :precondition (and (clear ?ob) (on-table ?ob) (arm-empty))
    :effect (and (holding ?ob) (not (clear ?ob)) (not (on-table ?ob))
                (not (arm-empty))))

    (:action putdown
    :parameters  (?ob)
    :precondition (holding ?ob)
    :effect (and (clear ?ob) (arm-empty) (on-table ?ob)
                (not (holding ?ob))))

    (:action stack
    :parameters  (?ob ?underob)
    :precondition (and (clear ?underob) (holding ?ob))
    :effect (and (arm-empty) (clear ?ob) (on ?ob ?underob)
                (not (clear ?underob)) (not (holding ?ob))))

    (:action unstack
    :parameters  (?ob ?underob)
    :precondition (and (on ?ob ?underob) (clear ?ob) (arm-empty))
    :effect (and (holding ?ob) (clear ?underob)
                (not (on ?ob ?underob)) (not (clear ?ob)) (not (arm-empty)))))
    """,
    "gripper": """;; source: https://github.com/AI-Planning/pddl-generators/blob/main/gripper/domain.pddl
    (define (domain gripper)
       (:requirements :strips)
       (:predicates (room ?r)
            (ball ?b)
            (gripper ?g)
            (at-robby ?r)
            (at ?b ?r)
            (free ?g)
            (carry ?o ?g))

       (:action move
           :parameters  (?from ?to)
           :precondition (and  (room ?from) (room ?to) (at-robby ?from))
           :effect (and  (at-robby ?to)
                 (not (at-robby ?from))))

       (:action pick
           :parameters (?obj ?room ?gripper)
           :precondition  (and  (ball ?obj) (room ?room) (gripper ?gripper)
                    (at ?obj ?room) (at-robby ?room) (free ?gripper))
           :effect (and (carry ?obj ?gripper)
                (not (at ?obj ?room))
                (not (free ?gripper))))

       (:action drop
           :parameters  (?obj  ?room ?gripper)
           :precondition  (and  (ball ?obj) (room ?room) (gripper ?gripper)
                    (carry ?obj ?gripper) (at-robby ?room))
           :effect (and (at ?obj ?room)
                (free ?gripper)
                (not (carry ?obj ?gripper)))))
    """,
}


def evaluate(
    source_pddl_str: str,
    target_pddl_str: str,
    domain_str: str | None = None,
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
    except Exception:
        return parseable, solveable, equivalent

    clean_pddl_str = problem_to_string(LenientProblemParser()(target_pddl_str))
    domain_str = domain_str or DOMAINS.get(target_graph.domain)

    try:
        solveable = downward.validate(
            domain_str,
            clean_pddl_str,
            oracle.plan_to_string(oracle.plan(target_graph)),
            VALIDATE,
        )
    except:
        return parseable, solveable, equivalent

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

    return parseable, solveable, equivalent
