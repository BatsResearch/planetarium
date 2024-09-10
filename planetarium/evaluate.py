import importlib.resources as resources
import os

from pddl.parser.problem import LenientProblemParser
from pddl.formatter import problem_to_string

from planetarium import builder, oracle, metric, downward
from . import domains


VALIDATE = os.getenv("VALIDATE", "Validate")
DOWNWARD = os.getenv("DOWNWARD", "downward")
DOMAINS = dict()

# load domains
for domain in resources.files(domains).iterdir():
    with domain.open() as f:
        DOMAINS[os.path.basename(domain).split(".")[0]] = f.read()


def evaluate(
    source_pddl_str: str,
    target_pddl_str: str,
    domain_str: str | None = None,
    is_placeholder: bool = False,
    check_solveable: bool = True,
    val: str = VALIDATE,
    fast_downward: str = DOWNWARD,
    **downward_args,
) -> tuple[bool, bool, bool]:
    """Evaluate two PDDL problem descriptions for equivalence.

    Args:
        source_pddl_str (str): The ground truth problem PDDL string.
        target_pddl_str (str): The second problem PDDL string.
        domain_str (str): The domain PDDL string.
        is_placeholder (bool, optional): Whether or not to treat the ground truth
            as a "placeholder" description. Defaults to False.
        check_solveable (bool, optional): Whether or not to check if the problem
            is solveable. Defaults to True. If False, the function will return
            False for the solveable element.

    Returns:
        tuple: A tuple containing the following boolean elements:
            - parseable: Whether or not the target PDDL string is parseable.
            - solveable: Whether or not the target PDDL string is solveable.
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

    if check_solveable and isinstance(domain_str, str):
        try:
            plan_str = oracle.plan_to_string(oracle.plan(target_graph))
        except (oracle.DomainNotSupportedError, NotImplementedError):
            try:
                plan_str, _ = downward.plan(
                    domain_str,
                    clean_pddl_str,
                    downward=fast_downward,
                    **downward_args,
                )
            except:
                return parseable, solveable, equivalent
        except:
            return parseable, solveable, equivalent

        try:
            if not (
                solveable := downward.validate(
                    domain_str,
                    clean_pddl_str,
                    plan_str,
                    val=val,
                )
            ):
                return parseable, solveable, equivalent
        except:
            return parseable, solveable, equivalent

    if source_graph == target_graph:
        equivalent = True
    elif not metric.equals(source_graph.init(), target_graph.init()):
        equivalent = False
    else:
        try:
            equivalent = metric.equals(
                oracle.fully_specify(source_graph, return_reduced=True),
                oracle.fully_specify(target_graph, return_reduced=True),
                is_placeholder=is_placeholder,
            )
        except:
            pass

    return parseable, solveable, equivalent
