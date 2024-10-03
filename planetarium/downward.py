# FastDownward python wrapper

import glob
import os
import re
import subprocess
import tempfile


def _get_best_plan(plan_filepath: str) -> tuple[str | None, float]:
    """Get the best plan from a FastDownward plan file.

    Args:
        plan_filepath (str): The path to the plan file.

    Returns:
        The best plan and its cost.
    """

    best_cost = float("inf")
    best_plan = None

    for plan_fp in glob.glob(f"{plan_filepath}*"):
        with open(plan_fp, "r") as f:
            *pddl_plan, cost_str = f.readlines()
            match = re.search(r"cost = ([-\d\.]+)", cost_str)
            if match:
                cost = float(match.group(1))

                if cost < best_cost:
                    best_cost = cost
                    best_plan = "\n".join([*pddl_plan, ";"])
    return best_plan, best_cost


def plan(
    domain: str,
    problem: str,
    downward: str = "downward",
    alias: str = "lama",
    **kwargs,
) -> tuple[str | None, float]:
    """Find plan using FastDownward.

    Args:
        domain (str): A string containing a PDDL domain definition.
        problem (str): A string containing a PDDL task/problem definition.
        downward (str, optional): Path to FastDownward. Defaults to "downward".
        alias (str, optional): The FastDownward alias to. Defaults to "lama".

    Returns:
        Returns the PDDL plan string, or `None` if the planner failed, and the
        plan cost.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        domain_filepath = os.path.join(tmpdir, "domain.pddl")
        task_filepath = os.path.join(tmpdir, "task.pddl")

        plan_filepath = os.path.join(tmpdir, "plan.pddl")
        sas_filepath = os.path.join(tmpdir, "output.sas")

        # build temporary domain and task files
        with open(domain_filepath, "w") as f:
            f.write(domain)
        with open(task_filepath, "w") as f:
            f.write(problem)

        # update arguments
        kwargs["plan-file"] = plan_filepath
        kwargs["sas-file"] = sas_filepath
        kwargs["alias"] = alias

        # build FastDownward arguments
        downward_args = []
        for k, v in kwargs.items():
            downward_args.append(f"--{k}")
            if isinstance(v, list) or isinstance(v, tuple):
                downward_args.extend(str(v_i) for v_i in v)
            else:
                downward_args.append(str(v))

        # call FastDownward
        subprocess.run(
            [downward, *downward_args, domain_filepath, task_filepath],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )

        # get results
        best_plan, best_cost = _get_best_plan(plan_filepath)

    return best_plan, best_cost


def validate(domain: str, problem: str, plan: str, val: str = "validate"):
    """Validate a plan using VAL.

    Args:
        domain (str): A string containing a PDDL domain definition.
        problem (str): A string containing a PDDL task/problem definition.
        plan (str): A string containing a PDDL plan.
        val (str, optional): Path to VAL. Defaults to "validate".
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        domain_filepath = os.path.join(tmpdir, "domain.pddl")
        task_filepath = os.path.join(tmpdir, "task.pddl")
        plan_filepath = os.path.join(tmpdir, "plan.pddl")

        # build temporary domain, task, and plan files
        with open(domain_filepath, "w") as f:
            f.write(domain)
        with open(task_filepath, "w") as f:
            f.write(problem)
        with open(plan_filepath, "w") as f:
            f.write(plan)

        # call VAL
        res = subprocess.run(
            [val, domain_filepath, task_filepath, plan_filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    return "Plan valid" in res.stdout.decode("utf-8")
