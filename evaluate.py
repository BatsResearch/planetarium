from typing import Any, Callable, Mapping

import dotenv

dotenv.load_dotenv()  # load openai api key

import multiprocessing as mp
import os
import signal
import sqlite3
import yaml

from lark.exceptions import LarkError
from pddl.core import Problem
from pddl.formatter import problem_to_string
from pddl.parser.problem import LenientProblemParser
import tqdm
import torch

from planetarium import builder, downward, graph, metric, oracle
import llm_planner as llmp

HF_USER_TOKEN = os.getenv("HF_USER_TOKEN")
VALIDATE = os.getenv("VALIDATE", "Validate")
DOWNWARD = os.getenv("DOWNWARD", "downward")

def signal_handler(signum, frame):
    raise TimeoutError("Timed out")


signal.signal(signal.SIGALRM, signal_handler)


def timeout_and_retry(
    func: Callable,
    *args,
    timeout: int = 30,
    retries: int = 5,
    **kwargs,
):
    """Run a function with a timeout and retries.

    Args:
        func (Callable): The function to run.
        timeout (int, optional): Seconds per attempt. Defaults to 30.
        retries (int, optional): Number of retries. Defaults to 5.

    Raises:
        TimeoutError: If the function times out.

    Returns:
        Any: The function's return value.
    """
    for _ in range(retries):
        try:
            signal.alarm(timeout)
            return func(*args, **kwargs)
        except TimeoutError:
            continue
        finally:
            signal.alarm(0)
    raise TimeoutError(f"Timed out after {retries} retries")


def plan(
    planner: llmp.Planner,
    problem: llmp.PlanningProblem | list[llmp.PlanningProblem],
    example_problems: list[llmp.PlanningProblem],
    domain_prompt: str,
    problem_prompt: str,
    max_new_tokens: int = 10_000,
) -> str:
    """Get plan for a given problem.

    Args:
        planner (llmp.Planner): The planner object to use.
        problem (llmp.PlanningProblem): The problem description to plan for.
        example_problem (llmp.PlanningProblem): An example problem description.
        problem_prompt (str): The prompt to use for the problem.

    Returns:
        str: The message completion.
    """
    context = []
    for example_problem in example_problems:
        context.extend(
            example_problem.apply_template(
                domain_prompt,
                problem_prompt,
            )
        )

    if isinstance(problem, llmp.PlanningProblem):
        messages = [
            problem.apply_template(
                domain_prompt,
                problem_prompt,
                include_answer=False,
            )
        ]
    else:
        messages = [
            p.apply_template(
                domain_prompt,
                problem_prompt,
                include_answer=False,
            )
            for p in problem
        ]

    device = None
    messages = [context + m for m in messages]
    if isinstance(planner, llmp.HFPlanner):
        device = planner.model.device

    return planner.plan_chat(
        messages,
        max_new_tokens=max_new_tokens,
        device=device,
    )


def load_planner(config: Mapping[str, dict[str, str]]) -> llmp.Planner:
    """Load a model based on the configuration.

    Args:
        config (Mapping[str, str]): The configuration for the model.

    Raises:
        ValueError: If the model type is not 'openai' or 'hf'.

    Returns:
        llmp.Planner: The loaded model.
    """
    if config["model"]["type"] == "openai":
        llm = llmp.OpenAIPlanner(
            config["model"]["model_name"],
            **config["model"].get("kwargs", {}),
        )
    elif config["model"]["type"] == "hf":
        llm = llmp.VLLMPlanner(
            config["model"]["model_name"],
            lora=config["model"].get("lora"),
            tokenizer=config["model"]["tokenizer_name"],
            trust_remote_code=True,
            dtype=torch.bfloat16,
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=config.get("max_model_len"),
        )
    else:
        raise ValueError(
            f"Unknown model type: {config['model']['type']}. Must be 'openai' or 'hf'."
        )

    return llm


def fast_equivalence(
    problem_pddl: str,
    llm_problem_pddl: str,
) -> tuple[bool, tuple[bool, bool, bool], dict[str, graph.ProblemGraph]]:
    """Evaluate a PDDL problem quickly (if possible).

    Args:
        problem_pddl (str): The ground truth PDDL.
        llm_problem_pddl (str): The PDDL output from the LLM.

    Returns:
        tuple[bool, dict[str, bool], dict[str, graph.ProblemGraph]]: A tuple
            with a boolean indicating if the problem was resolved, a tuple
            containing whether the PDDL is parseable, valid, and equivalent,
            and a dictionary containing the problem graphs.
    """
    # initialize variables
    parseable = False
    valid = False
    equivalent = False

    problem_graph = None
    llm_problem_graph = None

    resolved = False

    def result():
        return (
            resolved,
            (
                parseable,
                valid,
                equivalent,
            ),
            {
                "problem_graph": problem_graph,
                "llm_problem_graph": llm_problem_graph,
            },
        )

    try:
        # try to parse the LLM output
        llm_problem_graph = builder.build(llm_problem_pddl)
        parseable = True

        # reduce and further validate the LLM output
        oracle.reduce(llm_problem_graph.init())
        oracle.reduce(llm_problem_graph.goal())
        valid = True

        problem_graph = builder.build(problem_pddl)
        init, _ = problem_graph.decompose()

        if len(llm_problem_graph.constants) != len(problem_graph.constants):
            resolved = True
            return result()

        llm_init, _ = llm_problem_graph.decompose()

        if not timeout_and_retry(
            metric.equals,
            init,
            llm_init,
            is_placeholder=False,
            timeout=30,
            retries=5,
        ):
            # If the initial states are not equal, then the problems cannot be equivalent
            resolved = True
            return result()

    except LarkError:
        resolved = True
    except AttributeError:
        resolved = True
    except ValueError:
        resolved = True
    except TimeoutError:
        pass

    return result()


def full_equivalence(
    source: graph.ProblemGraph,
    target: graph.ProblemGraph,
    is_placeholder: bool = False,
) -> bool:
    """Checks if two scene graphs are equivalent.

    Args:
        source (graph.ProblemGraph): The source scene graph.
        target (graph.ProblemGraph): The target scene graph.

    Returns:
        bool: True if the scene graphs are equivalent, False otherwise.
    """
    return metric.equals(
        oracle.fully_specify(source, return_reduced=True),
        oracle.fully_specify(target, return_reduced=True),
        is_placeholder=is_placeholder,
    )


def clean(pddl_str: str) -> str:
    """Clean a PDDL string.

    Args:
        pddl_str (str): The PDDL string to clean.

    Returns:
        str: The cleaned PDDL string.
    """
    problem: Problem = LenientProblemParser()(pddl_str)
    return problem_to_string(problem)


def validate(
    pddl_str: str,
    domain_str: str,
    fast_downward: str = DOWNWARD,
    **downward_args,
) -> bool:
    """Validate a PDDL problem as "solvable".

    Args:
        pddl_str (str): The PDDL problem.
        domain_str (str): The PDDL domain.

    Returns:
        bool: Whether the PDDL is parseable and valid.
    """
    valid = False
    pddl_str = clean(pddl_str)
    try:
        problem_graph = builder.build(pddl_str)
        plan = oracle.plan_to_string(oracle.plan(problem_graph))
        valid = downward.validate(domain_str, pddl_str, plan, VALIDATE)
    except (LarkError, AttributeError, ValueError):
        pass
    except (oracle.DomainNotSupportedError, NotImplementedError):
        try:
            plan_str, _ = downward.plan(
                domain_str,
                pddl_str,
                downward=fast_downward,
                **downward_args,
            )
            valid = downward.validate(domain_str, pddl_str, plan_str, VALIDATE)
        except:
            pass

    return valid


def equivalence(
    problem_pddl: str,
    llm_problem_pddl: str,
    domains: dict[str, str],
    is_placeholder: bool = False,
) -> tuple[bool, bool, bool]:
    """Evaluate a PDDL problem and save the results.

    Args:
        problem_pddl (str): The ground truth PDDL.
        llm_problem_pddl (str): The PDDL output from the LLM.
        domains (dict[str, str]): The domains to use.
        is_placeholder (bool, optional): Whether the LLM output is a
            placeholder. Defaults to False.

    Returns:
        tuple[bool, bool, bool]: A tuple containing whether the PDDL is
            parseable, valid, and equivalent.
    """

    # fast equivalence check
    resolved, (parseable, valid, equivalent), graphs = fast_equivalence(
        problem_pddl, llm_problem_pddl
    )
    if resolved:
        return parseable, valid, equivalent

    return (
        parseable,
        validate(
            llm_problem_pddl,
            domains[graphs["llm_problem_graph"].domain],
            alias="lama-first",
        ),
        full_equivalence(
            graphs["problem_graph"],
            graphs["llm_problem_graph"],
            is_placeholder=is_placeholder,
        ),
    )


def load_problem_ids(config: dict, splits: list[str]) -> list[int]:
    """Load the problem ids for the splits.

    Args:
        config (dict): The configuration for the splits.
        splits (list[str]): The list of splits to load.

    Returns:
        list[int]: The list of problem ids.
    """
    with open(config["dataset"]["splits_path"], "r") as f:
        split_ids_cfg = yaml.safe_load(f)

    problem_ids = []

    for split in splits:
        split_keys: list[str] = config["dataset"]["splits"][split]
        for split_key in split_keys:
            split_ids = split_ids_cfg
            for key in split_key:
                split_ids = split_ids[key]

            problem_ids.extend(split_ids)

    return problem_ids


def load_ungenerated_problems(
    config: Mapping[str, str | Any],
    config_str: str,
    problem_ids: list[int],
) -> dict[int, llmp.PlanningProblem]:
    """Load a list of problems from the database.

    Args:
        config (Mapping[str, str | Any]): The configuration for the database.
        config_str (str): The configuration string.
        problem_ids (list[int]): The list of problem ids to load.

    Returns:
        dict[int, llmp.PlanningProblem]: The loaded problems.
    """
    problems = {}
    with sqlite3.connect(config["dataset"]["database_path"]) as conn:
        cursor = conn.cursor()
        # get domains
        cursor.execute("SELECT name, domain_pddl FROM domains")
        domains = {name: domain for name, domain in cursor.fetchall()}
        # get problems from problems table if it doesn't exist in llm_outputs table
        cursor.execute(
            f"SELECT problem_id FROM llm_outputs WHERE problem_id IN ({','.join('?' * len(problem_ids))}) AND config = ? AND model_name = ?",
            problem_ids + [config_str, config["evaluate"]["model"]["model_name"]],
        )
        ids = cursor.fetchall()
        problem_ids = set(problem_ids) - set(ids[0] for ids in ids)
        cursor.execute(
            f"SELECT id, domain, problem_pddl, natural_language FROM problems WHERE id IN ({','.join('?' * len(problem_ids))})",
            list(problem_ids),
        )

        for (
            problem_id,
            domain,
            problem_pddl,
            natural_language,
        ) in cursor.fetchall():
            problems[problem_id] = llmp.PlanningProblem(
                natural_language,
                domains[domain],
                problem_pddl,
            )

        conn.commit()
        cursor.close()

    return problems


def _generate_openai(args):
    problem_id, problem, example_problems, domain_prompt, problem_prompt, config = args
    planner = load_planner(config)
    return problem_id, plan(
        planner,
        problem,
        example_problems,
        domain_prompt,
        problem_prompt,
        max_new_tokens=None,
    )


def generate_openai(
    problems: dict[int, llmp.PlanningProblem],
    config: dict[str, dict[str, str | Any]],
    config_str: str,
):
    """Generate the PDDL output for a list of problems.

    Args:
        planner (llmp.Planner): The planner to use.
        problems (dict[int, llmp.PlanningProblem]): The problems to generate PDDL for.
        config (dict[str, dict[str, str | Any]): The configuration for the evaluation.
        config_str (str): The configuration string.
    """
    domain_prompt = config["dataset"]["prompts"]["domain"]
    problem_prompt = config["dataset"]["prompts"]["problem"]
    model_name = config["evaluate"]["model"]["model_name"]
    with sqlite3.connect(config["dataset"]["database_path"]) as conn:
        cursor = conn.cursor()
        with tqdm.tqdm(total=len(problems), desc="Generating PDDL") as pbar:
            with mp.Pool(8) as pool:
                args = (
                    (
                        problem_id,
                        problem,
                        [],
                        domain_prompt,
                        problem_prompt,
                        config["evaluate"],
                    )
                    for problem_id, problem in problems.items()
                )
                for problem_id, llm_problem_pddl in pool.imap_unordered(
                    _generate_openai, args
                ):
                    cursor.execute(
                        "INSERT INTO llm_outputs (problem_id, config, model_name, output) VALUES (?, ?, ?, ?)",
                        (
                            problem_id,
                            config_str,
                            model_name,
                            llm_problem_pddl[0],
                        ),
                    )
                    pbar.update()
                    conn.commit()
        cursor.close()


def generate_hf(
    problems: dict[int, llmp.PlanningProblem],
    config: dict[str, dict[str, str | Any]],
    config_str: str,
):
    """Generate the PDDL output for a list of problems.

    Args:
        problems (dict[int, llmp.PlanningProblem]): The problems to generate PDDL for.
        config (dict[str, dict[str, str  |  Any]]): The configuration for the evaluation.
        config_str (str): The configuration string.
    """
    domain_prompt = config["dataset"]["prompts"]["domain"]
    problem_prompt = config["dataset"]["prompts"]["problem"]
    model_name = config["evaluate"]["model"]["model_name"]
    batch_size = config["evaluate"].get("batch_size", 1)

    planner = load_planner(config["evaluate"])

    problems_iter = iter(problems.items())
    with sqlite3.connect(config["dataset"]["database_path"], timeout=30) as conn:
        cursor = conn.cursor()

        with tqdm.tqdm(
            total=len(problems), desc="Generating PDDL", smoothing=0.1
        ) as pbar:
            # sample problems of batch size
            while problems_iter:
                batch = []
                for problem_id, problem in problems_iter:
                    # Check if problem already exists
                    cursor.execute(
                        "SELECT output FROM llm_outputs WHERE problem_id = ? AND config = ? AND model_name = ?",
                        (problem_id, config_str, model_name),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        batch.append((problem_id, problem))
                        # add placeholder row
                        cursor.execute(
                            "INSERT INTO llm_outputs (output, problem_id, config, model_name) VALUES (?, ?, ?, ?)",
                            ("GENERATING", problem_id, config_str, model_name),
                        )
                        conn.commit()
                    if len(batch) == batch_size:
                        break

                if not batch:
                    break
                # Generate batch
                outputs = plan(
                    planner,
                    [p for _, p in batch],
                    [],
                    domain_prompt,
                    problem_prompt,
                )

                # save batch
                cursor.executemany(
                    "INSERT OR REPLACE INTO llm_outputs (output, problem_id, config, model_name) VALUES (?, ?, ?, ?)",
                    [
                        (output, problem_id, config_str, model_name)
                        for (problem_id, _), output in zip(batch, outputs)
                    ],
                )
                pbar.update(len(batch))
                conn.commit()
        cursor.close()


def _evaluate(args):
    domains, dataset_path, problem_id, config_str, model_name = args
    with sqlite3.connect(dataset_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT problem_pddl, is_placeholder FROM problems WHERE id = ?",
            (problem_id,),
        )
        problem_pddl, is_placeholder = cursor.fetchone()
        cursor.execute(
            "SELECT output, parseable, valid, equivalent FROM llm_outputs WHERE problem_id = ? AND config = ? AND model_name = ?",
            (problem_id, config_str, model_name),
        )
        llm_problem_pddl, parseable, valid, equivalent = cursor.fetchone()
        if equivalent is not None:
            return problem_id, config_str, model_name, (parseable, valid, equivalent)
        try:
            signal.alarm(900)
            parseable, valid, equivalent = equivalence(
                problem_pddl,
                llm_problem_pddl,
                domains,
                bool(is_placeholder),
            )
            signal.alarm(0)
        except TimeoutError as e:
            print("TIMEOUT", problem_id, llm_problem_pddl)
            return problem_id, config_str, model_name, (None, None, None)
        except Exception as e:
            equivalent = None
            print("ERROR", e, problem_id, llm_problem_pddl)
        cursor.close()
    return problem_id, config_str, model_name, (parseable, valid, equivalent)


def evaluate(problem_ids: list[int], config: dict):
    """Evaluate the output of the LLM.

    Args:
        problem_ids (list[int]): The list of problem ids to evaluate.
        config (dict): The configuration for the evaluation.
        config_str (str): The configuration string.
    """
    with sqlite3.connect(config["dataset"]["database_path"]) as conn:
        cursor = conn.cursor()
        # get domains
        cursor.execute("SELECT name, domain_pddl FROM domains")
        domains = {name: domain for name, domain in cursor.fetchall()}
        cursor.execute(
            f"""SELECT problem_id, config, model_name FROM llm_outputs WHERE
            problem_id IN ({','.join('?' * len(problem_ids))})
            AND equivalent IS NULL""",
            problem_ids,
        )
        problem_ids = cursor.fetchall()
        cursor.close()

    with mp.Pool(processes=max(1, min(mp.cpu_count(), len(problem_ids)))) as pool:
        args = (
            (
                domains,
                config["dataset"]["database_path"],
                problem_id,
                config_str,
                model_name,
            )
            for problem_id, config_str, model_name in problem_ids
        )
        for (
            problem_id,
            config_str,
            model_name,
            (parseable, valid, equivalent),
        ) in tqdm.tqdm(
            pool.imap_unordered(_evaluate, args),
            total=len(problem_ids),
            desc="Evaluating",
        ):
            with sqlite3.connect(config["dataset"]["database_path"]) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE llm_outputs SET equivalent = ?, parseable = ?, valid = ? WHERE problem_id = ? AND config = ? AND model_name = ?",
                    (
                        equivalent,
                        parseable,
                        valid,
                        problem_id,
                        config_str,
                        model_name,
                    ),
                )
                cursor.close()


def main(config_path: str):
    """Main function for evaluating an entire dataset.

    Args:
        config_path (str): A path to the dictionary containing the
            configuration for the evaluation.
    """
    with open(config_path, "r") as f:
        config: dict = yaml.safe_load(f)

    config_str = yaml.dump(config["evaluate"]["model"])

    problem_ids = load_problem_ids(config, config["evaluate"]["splits"])

    # Get LLM output first
    problems = load_ungenerated_problems(config, config_str, problem_ids)

    if len(problems) > 0:
        print("Generating: Run script with same arguments again to evaluate.")
        # It is very hard if not impossible at the moment to kill the vLLM
        # Ray, so re-running the script is the best option at the
        # moment.
        if config["evaluate"]["model"]["type"] == "openai":
            generate_openai(problems, config, config_str)
        elif config["evaluate"]["model"]["type"] == "hf":
            generate_hf(problems, config, config_str)
    else:
        evaluate(problem_ids, config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate planetarium.")

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="The configuration file to use.",
    )

    args = parser.parse_args()

    main(args.config)
