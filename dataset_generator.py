import abc
from collections import Counter, defaultdict
import itertools
import random
import re
import sqlite3
import yaml

import jinja2 as jinja
from pddl.core import Problem
from pddl import formatter as pddl_formatter
from pddl.core import And, Constant, Domain, Predicate
from pddl.parser import domain as domain_parser

import tqdm

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

SPLITS = None


class DatasetGenerator(abc.ABC):

    def __init__(self, domain: str, predicate_template: str = ""):
        self.domain: Domain = domain_parser.DomainParser()(DOMAINS[domain])

        # Remove leading whitespace from each line
        self.predicate_template = jinja.Template(
            re.sub(r"^\s+", "", predicate_template, flags=re.MULTILINE)
        )

    def explicit_description(
        self,
        predicates: list[Predicate],
        is_init: bool = True,
        randomize: bool = True,
        **kwargs,
    ) -> str:
        """Generate an explicit description of the state.

        Args:
            predicates (list[Predicate]): List of predicates.
            is_init (bool, optional): Whether the description is for the initial
                state. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            str: State description.
        """
        if randomize:
            random.shuffle(predicates)
        return self.predicate_template.render(
            predicates=predicates,
            is_init=is_init,
            **kwargs,
        ).strip()

    @abc.abstractmethod
    def abstract_description(
        self,
        task: str,
        is_init: bool = False,
        **kwargs,
    ) -> str:
        """Generate an abstract description of the state.

        Args:
            task (str): The task to describe.
            is_init (bool, optional): Whether the description is for the initial
                state. Defaults to False.

        Raises:
            ValueError: If the task is invalid/unsupported, or if the number of
                blocks is invalid.

        Returns:
            str: State description.
        """

    @abc.abstractmethod
    def get_task(
        self,
        init: str,
        goal: str,
        *args,
    ) -> tuple[Problem, dict[str, dict[str, str]], dict[str]]:
        """Generate a task.

        Args:
            init (str): Initial state setting type.
            goal (str): Goal state setting type.

        Returns:
            tuple[Problem, dict[str, dict[str, str]]]: PDDL problem,
                descriptions, and data.
        """
        raise NotImplementedError


class BlocksworldDatasetGenerator(DatasetGenerator):

    def __init__(self):
        super().__init__(
            "blocksworld",
            """
            {%- set tense = "is" if is_init else "should be" -%}
            {%- set arm_tense = "are" if is_init else "should be" -%}
            {%- if is_init -%}
                You have {{ num_blocks }} blocks.
            {%- else -%}
                Your goal is to have the following:
            {%- endif -%}
            {%- for predicate in predicates -%}
                {%- if predicate.name == "clear" %}
                    {{ predicate.terms[0].name }} {{ tense }} {{ predicate.name }}.
                {%- elif predicate.name == "on-table" %}
                    {{ predicate.terms[0].name }} {{ tense }} on the table.
                {%- elif predicate.name == "arm-empty" %}
                    Your arm {{ tense }} empty.
                {%- elif predicate.name == "holding" %}
                    You {{ arm_tense }} holding {{ predicate.terms[0].name }}.
                {%- elif predicate.name == "on" %}
                    {{ predicate.terms[0].name }} {{ tense }} on {{ predicate.terms[1].name }}.
                {%- endif -%}
            {%- endfor -%}
            """,
        )

    def stack(self, blocks: list[Constant], *_, **kwargs) -> list[Predicate]:
        predicates = [Predicate("arm-empty")]
        for i in range(1, len(blocks)):
            predicates.append(
                Predicate(
                    "on",
                    blocks[i],
                    blocks[i - 1],
                )
            )

        predicates.append(Predicate("clear", blocks[-1]))
        predicates.append(Predicate("on-table", blocks[0]))

        return predicates

    def on_table(self, blocks: list[Constant], *_, **kwargs) -> list[Predicate]:
        predicates = [Predicate("arm-empty")]
        for block in blocks:
            predicates.append(Predicate("clear", block))
            predicates.append(Predicate("on-table", block))

        return predicates

    def holding_one(self, blocks: list[Constant], *_, **kwargs) -> list[Predicate]:
        predicates = [Predicate("holding", blocks[0])]
        for block in blocks[1:]:
            predicates.append(Predicate("clear", block))
            predicates.append(Predicate("on-table", block))

        return predicates

    def staircase(
        self,
        blocks: list[Constant],
        *_,
        **kwargs,
    ) -> list[Predicate]:
        predicates = [Predicate("arm-empty")]
        idx = 0
        steps = self._staircase_num_steps(len(blocks))
        total = sum(range(steps + 1))
        assert len(blocks) == total
        for i in range(steps):
            predicates.append(Predicate("clear", blocks[idx]))
            for _ in range(i):
                idx += 1
                predicates.append(Predicate("on", blocks[idx - 1], blocks[idx]))
            predicates.append(Predicate("on-table", blocks[idx]))
            idx += 1

        return predicates

    def _equal_towers(self, num_blocks: int | list[int]) -> list[int]:
        def _get_height(num_blocks) -> list[int]:
            heights = [5, 4, 3, 2, 1]
            for height in heights:
                if num_blocks % height == 0:
                    num_blocks = [height] * (num_blocks // height)
                    return num_blocks

        if isinstance(num_blocks, int):
            num_blocks = _get_height(num_blocks)
        elif isinstance(num_blocks, list) and len(num_blocks) == 1:
            num_blocks = _get_height(num_blocks[0])
        elif isinstance(num_blocks, list) and any(
            n != num_blocks[0] for n in num_blocks
        ):
            num_blocks = _get_height(sum(num_blocks))
        else:
            raise ValueError("Invalid number of blocks for equal towers")

        return num_blocks

    def equal_towers(
        self,
        blocks: list[Constant],
        num_blocks: int | list[int],
        *_,
        **kwargs,
    ) -> list[Predicate]:
        num_blocks = self._equal_towers(num_blocks)
        assert sum(num_blocks) > 0, "Invalid number of blocks for equal towers"
        predicates = [Predicate("arm-empty")]
        blocks_iter = iter(blocks)

        for _ in range(len(num_blocks)):
            block = next(blocks_iter)
            predicates.append(Predicate("on-table", block))
            for _ in range(num_blocks[0] - 1):
                next_block = next(blocks_iter)
                predicates.append(Predicate("on", next_block, block))
                block = next_block
            predicates.append(Predicate("clear", block))

        return predicates

    def swap(
        self,
        blocks: list[Constant],
        num_blocks: list[int],
        *_,
        goal: bool = False,
        **kwargs,
    ) -> list[Predicate]:
        if len(num_blocks) != 2 or num_blocks[0] < 2 or num_blocks[1] < 2:
            raise ValueError("Swap requires two towers with at least 2 blocks each")

        new_blocks = blocks
        if goal:
            new_blocks[0], new_blocks[1] = new_blocks[1], new_blocks[0]

        predicates = [Predicate("arm-empty")]
        predicates.append(Predicate("on-table", blocks[0]))
        predicates.append(Predicate("on-table", blocks[1]))

        blocks_iter = iter(blocks[2:])
        for i, num in enumerate(num_blocks):
            block = blocks[i]
            for _ in range(num - 1):
                next_block = next(blocks_iter)
                predicates.append(Predicate("on", next_block, block))
                block = next_block
            predicates.append(Predicate("clear", block))

        return predicates

    def invert(
        self,
        blocks: list[Constant],
        num_blocks: int | list[int],
        *_,
        goal: bool = False,
    ) -> list[Predicate]:
        if isinstance(num_blocks, int):
            num_blocks = [num_blocks]
        if len(blocks) != sum(num_blocks):
            raise ValueError("Number of blocks does not match the sum of block counts")

        if goal:
            blocks = blocks[::-1]
            num_blocks = num_blocks[::-1]

        predicates = [Predicate("arm-empty")]
        idx = 0
        for tower_height in num_blocks:
            predicates.append(Predicate("clear", blocks[idx]))
            for _ in range(tower_height - 1):
                idx += 1
                predicates.append(Predicate("on", blocks[idx - 1], blocks[idx]))
            predicates.append(Predicate("on-table", blocks[idx]))
            idx += 1

        return predicates

    def tower(
        self,
        blocks: list[Constant],
        num_blocks: int | list[int],
        *_,
        **kwargs,
    ) -> list[Predicate]:
        if isinstance(num_blocks, int):
            num_blocks = [num_blocks]
        if len(blocks) != sum(num_blocks):
            raise ValueError("Number of blocks does not match the sum of block counts")

        predicates = [Predicate("arm-empty")]
        idx = 0
        for tower_height in num_blocks:
            predicates.append(Predicate("clear", blocks[idx]))
            for _ in range(tower_height - 1):
                idx += 1
                predicates.append(Predicate("on", blocks[idx - 1], blocks[idx]))
            predicates.append(Predicate("on-table", blocks[idx]))
            idx += 1

        return predicates

    def _staircase_num_steps(self, num_blocks: int) -> int:
        if isinstance(num_blocks, list):
            num_blocks = sum(num_blocks)
        num_steps = (2 * num_blocks + 0.25) ** 0.5 - 0.5
        if not num_steps.is_integer():
            raise ValueError(f"Invalid number of blocks for staircase: {num_blocks}")
        return int(num_steps)

    def abstract_description(
        self,
        task: str,
        num_blocks: int | list[int],
        is_init: bool = False,
    ) -> str:
        """Generate an abstract description of the state.

        Args:
            task (str): The task to describe.
            num_blocks (int | list[int]): Number of blocks in the scene.
            is_init (bool, optional): Whether the description is for the initial
                state. Defaults to False.

        Raises:
            ValueError: If the task is invalid/unsupported, or if the number of
                blocks is invalid.

        Returns:
            str: State description.
        """
        if isinstance(num_blocks, list):
            blocks = num_blocks
            num_blocks = sum(blocks)
        else:
            blocks = [num_blocks]
        match task, is_init:
            case ("on_table", True):
                return f"You have {num_blocks} blocks, each laying directly on the table, and your arm is empty."
            case ("on_table", False):
                return "Your goal is to unstack the blocks into individual blocks on the table."

            case ("stack", False):
                return "Your goal is to stack the blocks into a single stack."
            case ("stack", True):
                return f"You have {num_blocks} blocks, b1 through b{num_blocks}, stacked on top of each other, and your arm is empty."

            case ("holding_one", True):
                return f"You have {num_blocks} blocks. You are holding one block, and the rest are on the table."
            case ("holding_one", False):
                return f"You have {num_blocks} blocks, b1 through b{num_blocks}. Your goal is to unstack the blocks into individual blocks on the table, and to hold one of the blocks."

            case ("staircase", True):
                num_steps = self._staircase_num_steps(num_blocks)
                return f"You have {num_blocks} blocks, b1 through b{num_blocks}, stacked into {num_steps} stacks of increasing heights, starting with a stack of height 1."
            case ("staircase", False):
                num_steps = self._staircase_num_steps(num_blocks)
                return f"Your goal is to stack the blocks into {num_steps} stacks of increasing heights, starting with a stack of height 1."

            case ("equal_towers", True):
                num_blocks = self._equal_towers(blocks)
                return f"You have {sum(num_blocks)} blocks, b1 through b{sum(num_blocks)}, stacked into {len(blocks)} towers of equal heights, and your arm is empty."
            case ("equal_towers", False):
                num_blocks = self._equal_towers(blocks)
                return f"Your goal is to stack the blocks into {len(blocks)} towers of equal heights."

            case ("swap", True):
                return f"You have {num_blocks} blocks, b1 through b{num_blocks} in two towers with {blocks[0]} blocks in one and {blocks[1]} blocks in the other, and your arm is empty."
            case ("swap", False):
                return f"Your goal is to swap all blocks except the bottom blocks from one tower to the other."

            case ("invert", True):
                return f"You have {num_blocks} blocks, stacked into {len(blocks)} towers of heights {', '.join(str(h) for h in blocks)}, and your arm is empty."
            case ("invert", False):
                return f"Your goal is to invert each individual stack of blocks, such that the block that in each tower that was originally on the bottom will be on the top."

            case ("tower", True):
                return f"You have {num_blocks} blocks, stacked into {len(blocks)} towers of heights {', '.join(str(h) for h in blocks)}, and your arm is empty."
            case ("tower", False):
                return f"Your goal is to stack the blocks into a towers of heights {', '.join(str(h) for h in blocks)}."
            case _:
                raise ValueError(f"Invalid task: {task}")

    def get_task(
        self,
        init: str,
        goal: str,
        *args,
        randomize: bool = True,
    ) -> tuple[Problem, dict[str, dict[str, str]], dict[str, str]]:
        """Generate a blocksworld task.

        Args:
            init (str): Initial state setting type.
            goal (str): Goal state setting type.
            *args: Additional arguments for the task.
            randomize (bool, optional): Whether to randomize the order of the
                blocks. Defaults to True.

        Returns:
            tuple[Problem, dict[str, dict[str, str]]]: PDDL problem,
                descriptions, and data.
        """

        (num_blocks,) = args
        if isinstance(num_blocks, int):
            total = num_blocks
            num_blocks_str = str(num_blocks)
        else:
            total = sum(num_blocks)
            num_blocks_str = "_".join(str(arg) for arg in num_blocks)

        constants = [Constant(f"b{i + 1}") for i in range(total)]

        init_predicates = getattr(self, init)(constants, num_blocks)
        goal_predicates = getattr(self, goal)(
            constants,
            num_blocks,
            goal=True,
        )

        problem = Problem(
            name=f"{init}_to_{goal}_{num_blocks_str}",
            domain=self.domain,
            objects=constants,
            init=init_predicates,
            goal=And(*goal_predicates),
        )

        descriptions = {
            "init": {
                "abstract": self.abstract_description(
                    init,
                    is_init=True,
                    num_blocks=num_blocks,
                ),
                "explicit": self.explicit_description(
                    init_predicates,
                    is_init=True,
                    randomize=randomize,
                    num_blocks=(
                        num_blocks if isinstance(num_blocks, int) else sum(num_blocks)
                    ),
                ),
            },
            "goal": {
                "abstract": self.abstract_description(
                    goal,
                    is_init=False,
                    num_blocks=num_blocks,
                ),
                "explicit": self.explicit_description(
                    goal_predicates,
                    is_init=False,
                    randomize=randomize,
                    num_blocks=(
                        num_blocks if isinstance(num_blocks, int) else sum(num_blocks)
                    ),
                ),
            },
        }

        data = {
            "num_objects": len(constants),
            "init_num_propositions": len(init_predicates),
            "goal_num_propositions": len(goal_predicates),
        }

        return problem, descriptions, data


class GripperDatasetGenerator(DatasetGenerator):

    def __init__(self):
        super().__init__(
            "gripper",
            """
            {%- set tense = "is" if is_init else "should be" -%}
            {%- if is_init -%}
                You have {{ n_rooms }} rooms.
                You have {{ n_balls }} balls.
                You have 2 grippers.
            {%- else -%}
                Your goal is to have the following:
            {%- endif -%}
            {%- for predicate in predicates -%}
                {%- if predicate.name == "free" %}
                    {{ predicate.terms[0].name }} {{ tense }} {{ predicate.name }}.
                {%- elif predicate.name == "at-robby" %}
                    {{ predicate.terms[0].name }} {{ tense }} on the table.
                {%- elif predicate.name == "holding" %}
                    You {{ arm_tense }} holding {{ predicate.terms[0].name }}.
                {%- elif predicate.name == "at" %}
                    {{ predicate.terms[0].name }} {{ tense }} at {{ predicate.terms[1].name }}.
                {%- elif predicate.name == "carry" %}
                    Gripper {{ predicate.terms[1].name }} {{ tense }} carrying {{ predicate.terms[0].name }}.
                {%- endif -%}
            {%- endfor -%}""",
        )

    def _apply_typing(self, objects: list[Constant], typing: str) -> list[Predicate]:
        """Apply typing to a list of objects.

        Args:
            objects (list[Constant]): List of objects.
            typing (str): The type of the objects.

        Returns:
            list[Predicate]: List of typing predicates.
        """
        return [Predicate(typing, obj) for obj in objects]

    def one_room(
        self,
        rooms: list[Constant],
        balls: list[Constant],
        grippers: list[Constant],
        *_,
        **kwargs,
    ) -> list[Predicate]:
        """Task where all balls are in a single room.

        Args:
            rooms (list[Constant]): List of all available rooms.
            balls (list[Constant]): List of all balls.
            grippers (list[Constant]): List of all grippers.

        Returns:
            list[Predicate]: List of predicates describing the state.
        """
        predicates = [Predicate("free", gripper) for gripper in grippers]
        predicates += [Predicate("at", ball, rooms[0]) for ball in balls]

        return predicates

    def n_room_distributed(
        self,
        rooms: list[Constant],
        balls: list[Constant],
        grippers: list[Constant],
        balls_count: list[int],
        *_,
        **kwargs,
    ) -> list[Predicate]:
        """Task where balls are distributed across rooms.

        Args:
            rooms (list[Constant]): List of all available rooms.
            balls (list[Constant]): List of all balls.
            grippers (list[Constant]): List of all grippers.
            balls_count (list[int]): List of ball counts per room.

        Returns:
            list[Predicate]: List of predicates describing the state.
        """
        predicates = [Predicate("free", gripper) for gripper in grippers]
        iter_balls = iter(balls)

        for room, num_balls in zip(rooms, balls_count):
            for _ in range(num_balls):
                predicates.append(Predicate("at", next(iter_balls), room))

        return predicates

    def focus_max(
        self,
        rooms: list[Constant],
        balls: list[Constant],
        grippers: list[Constant],
        balls_count: list[int],
        *_,
        goal: bool = False,
        **kwargs,
    ) -> list[Predicate]:
        """Bring all balls to the room with the most balls.

        Args:
            rooms (list[Constant]): List of all available rooms.
            balls (list[Constant]): List of all balls.
            grippers (list[Constant]): List of all grippers.
            balls_count (list[int]): List of ball counts per room.

        Returns:
            list[Predicate]: List of predicates describing the state.

        Raises:
            ValueError: If the max number of balls in a room is not unique.
            ValueError: If there are less than 2 rooms or 2 balls.
            ValueError: If this is not a goal.
        """
        if len(balls_count) < 2 or sum(balls_count) < 2:
            raise ValueError("Focus max requires at least 2 rooms and 2 balls")
        if not goal:
            raise ValueError("Focus max requires a goal state")
        # find maximum and its frequency
        max_balls = max(balls_count)
        counts = Counter(balls_count)
        if counts[max_balls] > 1:
            raise ValueError("Focus max requires unique max number of balls in a room")

        predicates = [Predicate("free", gripper) for gripper in grippers]
        max_room = rooms[balls_count.index(max_balls)]

        for ball in balls:
            predicates.append(Predicate("at", ball, max_room))

        return predicates

    def focus_min(
        self,
        rooms: list[Constant],
        balls: list[Constant],
        grippers: list[Constant],
        balls_count: list[int],
        *_,
        goal: bool = False,
        **kwargs,
    ) -> list[Predicate]:
        """Bring all balls to the room with the minimum number of balls.

        Args:
            rooms (list[Constant]): List of all available rooms.
            balls (list[Constant]): List of all balls.
            grippers (list[Constant]): List of all grippers.
            balls_count (list[int]): List of ball counts per room.

        Returns:
            list[Predicate]: List of predicates describing the state.

        Raises:
            ValueError: If the min number of balls in a room is not unique.
            ValueError: If there are less than 2 rooms or 2 balls.
            ValueError: If this is not a goal.
        """
        if len(balls_count) < 2 or sum(balls_count) < 2:
            raise ValueError("Focus min requires at least 2 rooms and 2 balls")
        if not goal:
            raise ValueError("Focus min requires a goal state")
        # find maximum and its frequency
        min_balls = min(balls_count)
        counts = Counter(balls_count)
        if counts[min_balls] > 1:
            raise ValueError("Focus min requires unique min number of balls in a room")

        predicates = [Predicate("free", gripper) for gripper in grippers]
        min_room = rooms[balls_count.index(min_balls)]

        for ball in balls:
            predicates.append(Predicate("at", ball, min_room))

        return predicates

    def evenly_distributed(
        self,
        rooms: list[Constant],
        balls: list[Constant],
        grippers: list[Constant],
        *_,
        **kwargs,
    ) -> list[Predicate]:
        """Task where balls are evenly distributed across rooms.

        Args:
            rooms (list[Constant]): List of all available rooms.
            balls (list[Constant]): List of all balls.
            grippers (list[Constant]): List of all grippers.

        Returns:
            list[Predicate]: List of predicates describing the state.

        Raises:
            ValueError: If the number of balls is not divisible by the number of
                rooms.
        """
        if len(balls) % len(rooms) != 0:
            raise ValueError("Number of balls must be divisible by the number of rooms")

        predicates = [Predicate("free", gripper) for gripper in grippers]
        for i, room in enumerate(rooms):
            for ball in balls[i :: len(rooms)]:
                predicates.append(Predicate("at", ball, room))

        return predicates

    def swap(
        self,
        rooms: list[Constant],
        balls: list[Constant],
        grippers: list[Constant],
        *_,
        goal: bool = False,
    ) -> list[Predicate]:
        """Task where balls are swapped between rooms.

        Args:
            rooms (list[Constant]): List of all available rooms.
            balls (list[Constant]): List of all balls.
            grippers (list[Constant]): List of all grippers.
            goal (bool, optional): Whether to return goal state. Defaults to False.

        Returns:
            list[Predicate]: List of predicates describing the state.
        """
        if len(rooms) != 2:
            raise ValueError("Swap requires exactly 2 rooms")
        if len(balls) % 2 != 0:
            raise ValueError("Swap requires an even number of balls")

        if goal:
            balls = balls[::-1]

        predicates = [Predicate("free", gripper) for gripper in grippers]
        for i, room in enumerate(rooms):
            for ball in balls[i::2]:
                predicates.append(Predicate("at", ball, room))

        return predicates

    def abstract_description(
        self,
        task: str,
        n_rooms: int,
        balls: list[int],
        n_grippers: int = 2,
        is_init: bool = False,
    ) -> str:
        """Generate an abstract description of the state.

        Args:
            task (str): The task to describe.
            n_rooms (int): Number of rooms.
            balls (list[int]): List of ball counts.
            n_grippers (int, optional): Number of grippers. Defaults to 2.
            is_init (bool, optional): Whether the description is for the initial
                state. Defaults to False.

        Raises:
            ValueError: If the task is invalid/unsupported, or if the number of
                blocks is invalid.

        Returns:
            str: State description.
        """
        n_balls = sum(balls)
        objects = (
            f"You have {n_rooms} rooms, {n_balls} balls, and {n_grippers} grippers"
        )

        def n_room_distributed() -> str:
            ball_counter = Counter(balls)
            balls_per_room = ""
            for i, (count, room) in enumerate(ball_counter.items()):
                room_s = "s" if room > 1 else ""
                count_s = "s" if count > 1 else ""
                if i == len(ball_counter) - 1:
                    if len(ball_counter) > 1:
                        balls_per_room += "and "
                    balls_per_room += f"{room} room{room_s} with {count} ball{count_s}."
                else:
                    balls_per_room += (
                        f"{room} room{room_s} with {count} ball{count_s}, "
                    )
            return balls_per_room

        match task, is_init:
            case ("one_room", True):
                return f"{objects}. All the balls are in the first room, and the grippers are free. The robby is in the first room."
            case ("one_room", False):
                return "Your goal is to gather all balls into one room."

            case ("evenly_distributed", True):
                return f"{objects}. The balls are equally distributed across the rooms, and the grippers are free. The robby is in the first room."
            case ("evenly_distributed", False):
                return "Your goal is to distribute the balls equally across the rooms."

            case ("n_room_distributed", True):
                return f"{objects}. You have {n_room_distributed()}. The grippers are free. The robby is in the first room."
            case ("n_room_distributed", False):
                return f"Your goal is to have {n_room_distributed()}"

            case ("swap", True):
                return f"{objects}. The balls are split evenly between both rooms, and the grippers are free. The robby is in the first room."
            case ("swap", False):
                return "Your goal is to swap the location of the balls between the two rooms."

            case ("focus_max", False):
                return "Your goal is to bring all the balls into the room which already has the most balls."

            case ("focus_min", False):
                return "Your goal is to bring all the balls into the room which already has the least balls."

            case _:
                raise ValueError(f"Invalid task: {task}")

    def get_task(
        self,
        init: str,
        goal: str,
        gripper_and_balls_count: list[int],
        randomize: bool = True,
    ) -> tuple[Problem, dict[str, dict[str, str]]]:
        """Generate a gripper task.

        Args:
            init (str): Initial state setting type.
            goal (str): Goal state setting type.
            num_rooms (int): Number of rooms.
            gripper_and_balls_count (list[int]): List of ball
                counts. First element is the number of grippers.
            randomize (bool, optional): Whether to randomize the order of the
                balls. Defaults to True.

        Returns:
            tuple[Problem, dict[str, dict[str, str]]]: PDDL problem and descriptions.
        """
        num_grippers, *balls_count = gripper_and_balls_count
        num_rooms = len(balls_count)
        num_balls = sum(balls_count)

        rooms = [Constant(f"room{i + 1}") for i in range(num_rooms)]
        balls = [Constant(f"ball{i + 1}") for i in range(num_balls)]
        grippers = [Constant(f"gripper{i + 1}") for i in range(num_grippers)]

        constants = rooms + balls + grippers

        typing_predicates = [
            *self._apply_typing(rooms, "room"),
            *self._apply_typing(balls, "ball"),
            *self._apply_typing(grippers, "gripper"),
        ]

        init_predicates = getattr(self, init)(
            rooms,
            balls,
            grippers,
            balls_count,
        )
        goal_predicates = getattr(self, goal)(
            rooms,
            balls,
            grippers,
            balls_count,
            goal=True,
        )

        problem = Problem(
            name=f"{init}_to_{goal}_{num_rooms}_{num_balls}",
            domain=self.domain,
            objects=constants,
            init=[
                Predicate("at-robby", rooms[0]),
                *typing_predicates,
                *init_predicates,
            ],
            goal=And(*goal_predicates),
        )

        descriptions = {
            "init": {
                "abstract": self.abstract_description(
                    init,
                    is_init=True,
                    n_rooms=num_rooms,
                    balls=balls_count,
                ),
                "explicit": self.explicit_description(
                    init_predicates,
                    is_init=True,
                    n_rooms=num_rooms,
                    n_balls=num_balls,
                    randomize=randomize,
                ),
            },
            "goal": {
                "abstract": self.abstract_description(
                    goal,
                    is_init=False,
                    n_rooms=num_rooms,
                    balls=balls_count,
                ),
                "explicit": self.explicit_description(
                    goal_predicates,
                    is_init=False,
                    n_rooms=num_rooms,
                    n_balls=num_balls,
                    randomize=randomize,
                ),
            },
        }

        data = {
            "num_objects": len(constants),
            "init_num_propositions": len(init_predicates) + len(typing_predicates),
            "goal_num_propositions": len(goal_predicates),
        }

        return problem, descriptions, data


def create_tables(conn: sqlite3.Connection):
    """Create tables in the database.

    Args:
        conn (sqlite3.Connection): SQLite database connection.
    """
    cursor = conn.cursor()
    # Drop if exists
    cursor.execute("DROP TABLE IF EXISTS domains")
    cursor.execute("DROP TABLE IF EXISTS problems")
    cursor.execute("DROP TABLE IF EXISTS llm_outputs")
    cursor.execute("DROP TABLE IF EXISTS splits")
    # Create domains table
    cursor.execute(
        """
        CREATE TABLE domains (
            name TEXT PRIMARY KEY,
            domain_pddl TEXT NOT NULL
        );
        """
    )
    # Create problems table
    cursor.execute(
        """
        CREATE TABLE problems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            domain TEXT NOT NULL,
            init TEXT NOT NULL,
            goal TEXT NOT NULL,
            num_objects INTEGER NOT NULL,
            problem_pddl TEXT NOT NULL,
            natural_language TEXT NOT NULL UNIQUE,
            init_is_abstract INTEGER NOT NULL,
            init_num_propositions INTEGER NOT NULL,
            goal_is_abstract INTEGER NOT NULL,
            goal_num_propositions INTEGER NOT NULL,
            is_placeholder INTEGER NOT NULL,
            FOREIGN KEY (domain) REFERENCES domains (name)
        )
        """
    )
    # Create llm_outputs table
    cursor.execute(
        """
        CREATE TABLE llm_outputs (
            problem_id INTEGER NOT NULL,
            output TEXT NOT NULL,
            model_name TEXT NOT NULL,
            config TEXT NOT NULL,
            parseable INTEGER,
            valid INTEGER,
            equivalent INTEGER,
            PRIMARY KEY (problem_id, config)
            FOREIGN KEY (problem_id) REFERENCES problems (id)
        )
        """
    )
    # Create splits table
    cursor.execute(
        """
        CREATE TABLE splits (
            problem_id INTEGER NOT NULL,
            split_type TEXT NOT NULL,
            split TEXT NOT NULL,
            PRIMARY KEY (problem_id, split),
            FOREIGN KEY (problem_id) REFERENCES problems (id)
        )
        """
    )

    conn.commit()
    cursor.close()


def insert_domain(conn: sqlite3.Connection, name: str, domain_pddl: str):
    """Insert domains into the database.

    Args:
        conn (sqlite3.Connection): SQLite database connection.
        name (str): Name of the domain.
        domain_pddl (str): PDDL domain.
    """

    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO domains (name, domain_pddl)
        VALUES (?, ?)
        """,
        (name, domain_pddl),
    )
    conn.commit()
    cursor.close()


def insert_problems(
    conn: sqlite3.Connection,
    args: list[dict],
):
    """Insert problems into the database.

    Args:
        conn (sqlite3.Connection): SQLite database connection.
        args (list[dict]): List of problem arguments.
            name (str): Name of the problem (key).
            domain (str): Domain of the problem.
            init (str): Initial state setting type.
            goal (str): Goal state setting type.
            num_objects (int): Number of objects in the problem.
            problem_pddl (str): PDDL problem.
            natural_language (str): Natural language description of the problem.
            init_is_abstract (bool): Whether the initial state is abstract.
            init_num_propositions (int): Number of predicates in the initial state.
            goal_is_abstract (bool): Whether the goal state is abstract.
            goal_num_propositions (int): Number of predicates in the goal state.
            is_placeholder (bool): Whether the problem requires strict equivalency
                between initial and goal states.
    """
    cursor = conn.cursor()
    cursor.executemany(
        """
        INSERT OR IGNORE INTO problems (
            name,
            domain,
            init,
            goal,
            num_objects,
            problem_pddl,
            natural_language,
            init_is_abstract,
            init_num_propositions,
            goal_is_abstract,
            goal_num_propositions,
            is_placeholder
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                arg["name"],
                arg["domain"],
                arg["init"],
                arg["goal"],
                arg["num_objects"],
                arg["problem_pddl"],
                arg["natural_language"],
                arg["init_is_abstract"],
                arg["init_num_propositions"],
                arg["goal_is_abstract"],
                arg["goal_num_propositions"],
                arg["is_placeholder"],
            )
            for arg in args
        ],
    )
    conn.commit()
    cursor.close()


def generate(
    config: dict[str, str | int | dict[str, list[int]]],
    database_path: str = "dataset.db",
):
    """Generate the dataset.

    Args:
        config (dict[str, str | int | dict[str, list[int]]]): Configuration.
        database_path (str, optional): Path to the database. Defaults to
            "dataset.db".
    Raises:
        ValueError: If the domain is invalid.
    """
    conn = sqlite3.connect(database_path)
    create_tables(conn)

    problems = []
    for domain_cfg in config["domains"]:
        match domain_cfg["name"]:
            case "blocksworld":
                generator = BlocksworldDatasetGenerator()
            case "gripper":
                generator = GripperDatasetGenerator()
            case _:
                raise ValueError(f"Invalid domain: {domain_cfg['name']}")
        insert_domain(
            conn,
            domain_cfg["name"],
            pddl_formatter.domain_to_string(generator.domain),
        )

        layout_cfg: dict = domain_cfg["layouts"]
        initial_layouts = layout_cfg["initial"]
        goal_layouts = layout_cfg["goal"]
        strictly_both = layout_cfg.get("strictly_both", [])
        fully_strict = layout_cfg.get("strict", [])

        with tqdm.tqdm(
            total=len(initial_layouts) * len(goal_layouts),
            desc=f"Generating {domain_cfg['name']} dataset",
        ) as pbar:
            for start in initial_layouts:
                for end in goal_layouts:
                    pbar.update()
                    if (
                        start in strictly_both or end in strictly_both
                    ) and start != end:
                        continue

                    task = f"{start}_to_{end}"
                    for source in (start, end):
                        for args in domain_cfg["tasks"].get(source, []):
                            try:
                                problem, descriptions, data = generator.get_task(
                                    start,
                                    end,
                                    args,
                                    randomize=config.get("randomize_predicates", False),
                                )
                            except ValueError:
                                continue

                            # ground truth PDDL problem
                            problem_str = pddl_formatter.problem_to_string(problem)
                            if isinstance(args, int):
                                arg_str = str(args)
                            else:
                                arg_str = "_".join(str(arg) for arg in args)

                            problem_name = f"{task}_{arg_str}"

                            # problem descriptions for each combination of abstract/explicit initial and goal states
                            for init_desc, goal_desc in itertools.product(
                                ("abstract", "explicit"),
                                ("abstract", "explicit"),
                            ):
                                problem_desc = [
                                    descriptions["init"][init_desc],
                                    descriptions["goal"][goal_desc],
                                ]
                                problem_desc_str = "\n".join(problem_desc)
                                problems.append(
                                    {
                                        "name": domain_cfg["name"] + problem_name,
                                        "domain": domain_cfg["name"],
                                        "init": start,
                                        "goal": end,
                                        "num_objects": data["num_objects"],
                                        "problem_pddl": problem_str,
                                        "natural_language": problem_desc_str,
                                        "init_is_abstract": init_desc == "abstract",
                                        "init_num_propositions": data[
                                            "init_num_propositions"
                                        ],
                                        "goal_is_abstract": goal_desc == "abstract",
                                        "goal_num_propositions": data[
                                            "goal_num_propositions"
                                        ],
                                        "is_placeholder": not (
                                            source in strictly_both
                                            or (
                                                init_desc == "explicit"
                                                and goal_desc == "explicit"
                                            )
                                            or start in fully_strict
                                            or end in fully_strict
                                        ),
                                    }
                                )

    print(f"Upwards of {len(problems):,} problems generated.")
    insert_problems(conn, problems)
    conn.close()


def insert_split(
    conn: sqlite3.Connection,
    name: str,
    split: dict[int | str, list[int]],
):
    """Insert a split into the database.

    Args:
        conn (sqlite3.Connection): SQLite database connection.
        name (str): Name of the split.
        split (dict[int | str, list[int]]): Split data.
    """
    cursor = conn.cursor()
    for split_id, problem_ids in split.items():
        cursor.executemany(
            f"""
            INSERT INTO splits (problem_id, split_type, split)
            VALUES (?, ?, ?)
            """,
            [(problem_id, name, split_id) for problem_id in problem_ids],
        )
    conn.commit()
    cursor.close()


def split(
    config: dict[str, str | int],
    database_path: str = "dataset.db",
    split_path: str = "splits.yaml",
):
    global SPLITS
    random.seed(config.get("random_seed", 42))
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # split by domain
    cursor.execute(
        """
        SELECT name
        FROM domains
        """
    )
    domains = [row[0] for row in cursor.fetchall()]
    domain_splits = {}

    with tqdm.tqdm(total=len(domains) * 4, desc="Splitting dataset") as pbar:
        for domain in domains:
            # split by abstraction
            abstractions = {
                "abstract": True,
                "explicit": False,
            }
            abstraction_splits = {}
            for init_abstraction, init_is_abstract in abstractions.items():
                for goal_abstraction, goal_is_abstract in abstractions.items():
                    cursor.execute(
                        """
                        SELECT id
                        FROM problems
                        WHERE init_is_abstract = ? AND goal_is_abstract = ? AND domain = ?
                        """,
                        (init_is_abstract, goal_is_abstract, domain),
                    )
                    abstraction_splits[f"{init_abstraction}_to_{goal_abstraction}"] = [
                        row[0] for row in cursor.fetchall()
                    ]
            pbar.update()

            # split by predicate size
            size_splits = {}
            split_scales = config.get("split_scales", [5, 10, 15, 20])
            split_scales = [0, *split_scales, float("inf")]

            for lower, upper in zip(split_scales[:-1], split_scales[1:]):
                cursor.execute(
                    """
                    SELECT id
                    FROM problems
                    WHERE init_num_propositions + goal_num_propositions > ?
                    AND init_num_propositions + goal_num_propositions <= ?
                    AND domain = ?
                    """,
                    (lower, upper, domain),
                )
                size_splits[f"{lower}-{upper}"] = [row[0] for row in cursor.fetchall()]
            pbar.update()

            # split by strictness
            strict_splits = {}
            for i in range(2):
                cursor.execute(
                    """
                    SELECT id
                    FROM problems
                    WHERE is_placeholder = ?
                    AND domain = ?
                    """,
                    (i, domain),
                )
                strict_splits["placeholder" if i else "non-placeholder"] = [
                    row[0] for row in cursor.fetchall()
                ]
            pbar.update()

            # split by random
            cursor.execute(
                """
                SELECT id
                FROM problems
                WHERE
                domain = ?
                """,
                (domain,),
            )
            problems = sorted([row[0] for row in cursor.fetchall()])
            random.shuffle(problems)

            num_random_splits = config.get("num_random_splits", 5)
            random_splits = {
                f"{i}": problems[i::num_random_splits] for i in range(num_random_splits)
            }

            splits = {
                "abstraction": abstraction_splits,
                "size": size_splits,
                "placeholder": strict_splits,
                "random": random_splits,
            }
            domain_splits[domain] = splits
            pbar.update()

    cursor.close()
    with open(split_path, "w") as f:
        yaml.safe_dump(domain_splits, f, indent=2, default_flow_style=None)

    SPLITS = domain_splits

    general_splits = defaultdict(lambda: defaultdict(list))
    for domain, domain_split in domain_splits.items():
        for split_type, split in domain_split.items():
            for split_name, split_ids in split.items():
                general_splits[split_type][split_name].extend(split_ids)

    for split_type, split in general_splits.items():
        insert_split(conn, split_type, split)

    conn.close()


def report(split_path: str = "splits.yaml", output_path: str = "report.md"):
    global SPLITS
    if SPLITS is not None:
        splits = SPLITS
    else:
        with open(split_path, "r") as f:
            splits: dict[str, dict[str, dict[str, list[int]]]] = yaml.safe_load(f)

    report = ["# Dataset Report"]

    # All problems report
    total = sum(
        len(split)
        for domain in splits.values()
        for split in domain["placeholder"].values()
    )
    report.append(f"Total number of problems: {total:,}.")

    # Generate table for each split
    domains = list(splits.keys())
    split_types = next(iter(splits.values()))

    for split, split_keys in tqdm.tqdm(
        split_types.items(),
        desc="Generating report",
    ):
        report.append(f"## Split by {split.capitalize()}")
        domains_str = " | ".join(domains) + " |"
        report.append(f"| Domain | {domains_str}")
        report.append("|:---:|" + "---:|" * len(domains))

        row = "|"
        for key in split_keys:
            row += f" {key} |"
            for domain in domains:
                row += f" {len(splits[domain][split][key]):,} |"
            report.append(row)
            row = "|"

    with open(output_path, "w") as f:
        f.write("\n".join(report))


def main(config_path: str):
    with open(config_path, "r") as f:
        config: dict[str, str | int | dict[str, list[int]]] = yaml.safe_load(f)

    if config.get("actions", {}).get("generate", True):
        generate(config["generate"], config.get("database_path", "dataset.db"))
    if config.get("actions", {}).get("split", True):
        split(
            config["split_args"],
            config.get(
                "database_path",
                "dataset.db",
            ),
            config.get(
                "split_dataset_path",
                "splits.yaml",
            ),
        )
    if config.get("actions", {}).get("report", True):
        report(
            config.get(
                "split_dataset_path",
                "splits.yaml",
            ),
            config.get(
                "report_path",
                "report.md",
            ),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Generate PDDL dataset")

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="dataset_config.yaml",
        help="Path to the dataset configuration file",
    )

    args = parser.parse_args()

    main(args.config)
