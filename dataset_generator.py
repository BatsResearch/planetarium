from typing import Iterable

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

from planetarium import DOMAINS

import tqdm


SPLITS = None


def int_to_ordinal(number):
    """Converts an integer to its ordinal string representation (e.g., 1 -> 'first', 2 -> 'second').

    Args:
      number: The integer to convert.

    Returns:
      The ordinal string representation of the number, or None if the number is outside the range 1-20.
    """
    ordinals = [
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "eleventh",
        "twelfth",
        "thirteenth",
        "fourteenth",
        "fifteenth",
        "sixteenth",
        "seventeenth",
        "eighteenth",
        "nineteenth",
        "twentieth",
        "twenty-first",
        "twenty-second",
        "twenty-third",
        "twenty-fourth",
        "twenty-fifth",
        "twenty-sixth",
        "twenty-seventh",
        "twenty-eighth",
        "twenty-ninth",
        "thirtieth",
    ]

    if 1 <= number <= 30:
        return ordinals[number - 1]
    else:
        raise ValueError(f"Number {number} is outside the range 1-20")


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
    ) -> tuple[Problem, dict[str, dict[str, str]], dict[str, int | float | str]]:
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

    def stack(
        self,
        blocks: list[Constant],
        blocks_list: list[int] | None,
        goal: bool = False,
    ) -> list[Predicate]:
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

    def on_table(
        self,
        blocks: list[Constant],
        blocks_list: list[int] | None,
        goal: bool = False,
    ) -> list[Predicate]:
        predicates = [Predicate("arm-empty")]
        for block in blocks:
            predicates.append(Predicate("clear", block))
            predicates.append(Predicate("on-table", block))

        return predicates

    def holding_one(
        self,
        blocks: list[Constant],
        blocks_list: list[int],
        goal: bool = False,
    ) -> list[Predicate]:
        predicates = [Predicate("holding", blocks[0])]
        for block in blocks[1:]:
            predicates.append(Predicate("clear", block))
            predicates.append(Predicate("on-table", block))

        return predicates

    def _staircase_num_steps(self, num_blocks: int) -> int:
        num_steps: float = (2 * num_blocks + 0.25) ** 0.5 - 0.5
        if not num_steps.is_integer():
            raise ValueError(f"Invalid number of blocks for staircase: {num_blocks}")
        return int(num_steps)

    def staircase(
        self,
        blocks: list[Constant],
        blocks_list: list[int] | None,
        goal: bool = False,
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
                    tower_heights = [height] * (num_blocks // height)
                    return tower_heights
            return num_blocks

        if isinstance(num_blocks, int):
            tower_heights = _get_height(num_blocks)
        elif isinstance(num_blocks, list) and len(num_blocks) == 1:
            tower_heights = _get_height(num_blocks[0])
        elif isinstance(num_blocks, list) and any(
            n != num_blocks[0] for n in num_blocks
        ):
            tower_heights = _get_height(sum(num_blocks))
        else:
            raise ValueError("Invalid number of blocks for equal towers")

        return tower_heights

    def equal_towers(
        self,
        blocks: list[Constant],
        blocks_list: list[int] | None,
        goal: bool = False,
    ) -> list[Predicate]:
        tower_heights = self._equal_towers(blocks_list or len(blocks))
        assert sum(tower_heights) > 0, "Invalid number of blocks for equal towers"
        predicates = [Predicate("arm-empty")]
        blocks_iter = iter(blocks)

        for _ in range(len(tower_heights)):
            block = next(blocks_iter)
            predicates.append(Predicate("on-table", block))
            for _ in range(tower_heights[0] - 1):
                next_block = next(blocks_iter)
                predicates.append(Predicate("on", next_block, block))
                block = next_block
            predicates.append(Predicate("clear", block))

        return predicates

    def swap(
        self,
        blocks: list[Constant],
        blocks_list: list[int],
        goal: bool = False,
    ) -> list[Predicate]:
        if len(blocks_list) != 2 or blocks_list[0] < 2 or blocks_list[1] < 2:
            raise ValueError("Swap requires two towers with at least 2 blocks each")

        new_blocks = blocks
        if goal:
            new_blocks[0], new_blocks[1] = new_blocks[1], new_blocks[0]

        predicates = [Predicate("arm-empty")]
        predicates.append(Predicate("on-table", blocks[0]))
        predicates.append(Predicate("on-table", blocks[1]))

        blocks_iter = iter(blocks[2:])
        for i, num in enumerate(blocks_list):
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
        blocks_list: list[int],
        goal: bool = False,
    ) -> list[Predicate]:
        blocks_list = blocks_list or [sum(blocks)]
        if len(blocks) != sum(blocks_list):
            raise ValueError("Number of blocks does not match the sum of block counts")

        if goal:
            blocks = blocks[::-1]
            blocks_list = blocks_list[::-1]

        predicates = [Predicate("arm-empty")]
        idx = 0
        for tower_height in blocks_list:
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
        blocks_list: list[int],
        goal: bool = False,
    ) -> list[Predicate]:
        blocks_list = blocks_list or [sum(blocks)]
        if len(blocks) != sum(blocks_list):
            raise ValueError("Number of blocks does not match the sum of block counts")

        predicates = [Predicate("arm-empty")]
        idx = 0
        for tower_height in blocks_list:
            predicates.append(Predicate("clear", blocks[idx]))
            for _ in range(tower_height - 1):
                idx += 1
                predicates.append(Predicate("on", blocks[idx - 1], blocks[idx]))
            predicates.append(Predicate("on-table", blocks[idx]))
            idx += 1

        return predicates

    def abstract_description(
        self,
        task: str,
        blocks_list: list[int],
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
        num_blocks = sum(blocks_list)
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
                num_blocks = self._equal_towers(blocks_list)
                return f"You have {sum(num_blocks)} blocks, b1 through b{sum(num_blocks)}, stacked into {len(blocks_list)} towers of equal heights, and your arm is empty."
            case ("equal_towers", False):
                num_blocks = self._equal_towers(blocks_list)
                return f"Your goal is to stack the blocks into {len(blocks_list)} towers of equal heights."

            case ("swap", True):
                return f"You have {num_blocks} blocks, b1 through b{num_blocks} in two towers with {blocks_list[0]} blocks in one and {blocks_list[1]} blocks in the other, and your arm is empty."
            case ("swap", False):
                return f"Your goal is to swap all blocks except the bottom blocks from one tower to the other."

            case ("invert", True):
                return f"You have {num_blocks} blocks, stacked into {len(blocks_list)} towers of heights {', '.join(str(h) for h in blocks_list)}, and your arm is empty."
            case ("invert", False):
                return f"Your goal is to invert each individual stack of blocks, such that the block that in each tower that was originally on the bottom will be on the top."

            case ("tower", True):
                return f"You have {num_blocks} blocks, stacked into {len(blocks_list)} towers of heights {', '.join(str(h) for h in blocks_list)}, and your arm is empty."
            case ("tower", False):
                return f"Your goal is to stack the blocks into a towers of heights {', '.join(str(h) for h in blocks_list)}."
            case _:
                raise ValueError(f"Invalid task: {task}")

    def get_task(
        self,
        init: str,
        goal: str,
        blocks_list: list[int],
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

        blocks_list_str = "_".join(str(arg) for arg in blocks_list)
        num_blocks = sum(blocks_list)

        constants = [Constant(f"b{i + 1}") for i in range(num_blocks)]

        init_predicates = getattr(self, init)(
            constants,
            blocks_list=blocks_list,
            goal=False,
        )
        goal_predicates = getattr(self, goal)(
            constants,
            blocks_list=blocks_list,
            goal=True,
        )

        problem = Problem(
            name=f"{init}_to_{goal}_{blocks_list_str}",
            domain=self.domain,
            objects=constants,
            init=init_predicates,
            goal=And(*goal_predicates),
        )

        descriptions = {
            "init": {
                "abstract": self.abstract_description(
                    init,
                    blocks_list=blocks_list,
                    is_init=True,
                ),
                "explicit": self.explicit_description(
                    init_predicates,
                    is_init=True,
                    randomize=randomize,
                    num_blocks=num_blocks,
                ),
            },
            "goal": {
                "abstract": self.abstract_description(
                    goal,
                    blocks_list=blocks_list,
                    is_init=False,
                ),
                "explicit": self.explicit_description(
                    goal_predicates,
                    is_init=False,
                    randomize=randomize,
                    num_blocks=num_blocks,
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
                You have {{ n_grippers }} grippers.
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
        balls_in_grippers: list[int],
        balls_in_rooms: list[int],
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
        if sum(balls_in_grippers) != 0:
            raise ValueError("Grippers should not hold any balls in this task")
        predicates = [Predicate("free", gripper) for gripper in grippers]
        iter_balls = iter(balls)

        for room, num_balls in zip(rooms, balls_in_rooms):
            for _ in range(num_balls):
                predicates.append(Predicate("at", next(iter_balls), room))

        return predicates

    def focus_max(
        self,
        rooms: list[Constant],
        balls: list[Constant],
        grippers: list[Constant],
        balls_in_rooms: list[int],
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
        if len(balls_in_rooms) < 2 or sum(balls_in_rooms) < 2:
            raise ValueError("Focus max requires at least 2 rooms and 2 balls")
        if not goal:
            raise ValueError("Focus max requires a goal state")
        # find maximum and its frequency
        max_balls = max(balls_in_rooms)
        counts = Counter(balls_in_rooms)
        if counts[max_balls] > 1:
            raise ValueError("Focus max requires unique max number of balls in a room")

        predicates = [Predicate("free", gripper) for gripper in grippers]
        max_room = rooms[balls_in_rooms.index(max_balls)]

        for ball in balls:
            predicates.append(Predicate("at", ball, max_room))

        return predicates

    def focus_min(
        self,
        rooms: list[Constant],
        balls: list[Constant],
        grippers: list[Constant],
        balls_in_rooms: list[int],
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
        if len(balls_in_rooms) < 2 or sum(balls_in_rooms) < 2:
            raise ValueError("Focus min requires at least 2 rooms and 2 balls")
        if not goal:
            raise ValueError("Focus min requires a goal state")
        # find maximum and its frequency
        min_balls = min(balls_in_rooms)
        counts = Counter(balls_in_rooms)
        if counts[min_balls] > 1:
            raise ValueError("Focus min requires unique min number of balls in a room")

        predicates = [Predicate("free", gripper) for gripper in grippers]
        min_room = rooms[balls_in_rooms.index(min_balls)]

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
        goal: bool = False,
        **_,
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

    def juggle(
        self,
        rooms: list[Constant],
        balls: list[Constant],
        grippers: list[Constant],
        balls_in_grippers: list[int],
        balls_in_rooms: list[int],
        goal: bool = False,
    ) -> list[Predicate]:
        # only first room can have balls
        if sum(balls_in_rooms[1:]) != 0:
            raise ValueError("Juggle requires all balls to be in the first room")

        used_grippers = [i for i, v in enumerate(balls_in_grippers) if v > 0]
        if len(used_grippers) < 2:
            raise ValueError("Juggle requires at least 2 grippers with balls")

        held_balls = balls[: len(used_grippers)]
        if goal:
            # rotate by 1
            held_balls = held_balls[1:] + held_balls[:1]

        predicates = []
        for i, gripper in enumerate(grippers):
            if i in used_grippers:
                predicates.append(Predicate("carry", held_balls[i], gripper))
            else:
                predicates.append(Predicate("free", gripper))

        for ball in balls[len(used_grippers) :]:
            predicates.append(Predicate("at", ball, rooms[0]))

        return predicates

    def pickup(
        self,
        rooms: list[Constant],
        balls: list[Constant],
        grippers: list[Constant],
        balls_in_grippers: list[int],
        balls_in_rooms: list[int],
        goal: bool = True,
    ) -> list[Predicate]:
        assert goal
        if len(balls) > len(grippers):
            raise ValueError("Not enough grippers to hold all balls")
        # NOTE: since we do not define any specific test cases for this task,
        # all the variables for balls_in_grippres and balls_in_rooms come from
        # the initial scene

        held_balls = balls[: sum(balls_in_grippers)]
        free_balls = balls[sum(balls_in_grippers) :]

        used_grippers = [grippers[i] for i, v in enumerate(balls_in_grippers) if v > 0]
        free_grippers = set(grippers) - set(used_grippers)

        predicates = []
        for gripper, ball in zip(used_grippers, held_balls):
            predicates.append(Predicate("carry", ball, gripper))

        for gripper, ball in zip(free_grippers, free_balls):
            predicates.append(Predicate("carry", ball, gripper))

        return predicates

    def drop_and_pickup(
        self,
        rooms: list[Constant],
        balls: list[Constant],
        grippers: list[Constant],
        balls_in_grippers: list[int],
        balls_in_rooms: list[int],
        goal: bool = True,
    ) -> list[Predicate]:
        assert goal
        # NOTE: since we do not define any specific test cases for this task,
        # all the variables for balls_in_grippres and balls_in_rooms come from
        # the initial scene
        if len(rooms) < 2:
            raise ValueError("Drop and pickup requires at least 2 rooms")
        if not any([b == 0 for b in balls_in_rooms[1:]]):
            pass
        if sum(balls_in_rooms) > len(grippers):
            raise ValueError("Not enough grippers to hold all balls")

        room_idx = balls_in_rooms[1:].index(0) + 1
        free_room = rooms[room_idx]

        held_balls = balls[: sum(balls_in_grippers)]
        free_balls = balls[sum(balls_in_grippers) :]

        assert len(free_balls) == sum(balls_in_rooms)

        # drop all balls into free room
        predicates = []
        for ball in held_balls:
            predicates.append(Predicate("at", ball, free_room))

        for gripper, ball in zip(grippers, free_balls):
            predicates.append(Predicate("carry", ball, gripper))

        return predicates

    def holding(
        self,
        rooms: list[Constant],
        balls: list[Constant],
        grippers: list[Constant],
        balls_in_grippers: list[int],
        balls_in_rooms: list[int],
        goal: bool = False,
    ) -> list[Predicate]:
        if not (len(grippers) and len(balls)):
            raise ValueError("Holding requires at least one gripper and one ball")

        held_balls = balls[: sum(balls_in_grippers)]
        free_balls = balls[sum(balls_in_grippers) :]

        predicates = []

        for gripper, ball in zip(grippers, held_balls):
            predicates.append(Predicate("carry", ball, gripper))

        # free any additional grippers
        for gripper in grippers[len(held_balls) :]:
            predicates.append(Predicate("free", gripper))

        # NOTE: we will ignore the balls in rooms argument for holding if this
        # is a goal state
        if not goal:
            # place free balls according to balls_in_rooms
            balls_iter = iter(free_balls)
            for room_idx, num_balls in enumerate(balls_in_rooms):
                room = rooms[room_idx]
                for _ in range(num_balls):
                    predicates.append(Predicate("at", next(balls_iter), room))

        return predicates

    def abstract_description(
        self,
        task: str,
        n_rooms: int,
        n_balls: int,
        n_grippers: int,
        balls_in_grippers: list[int],
        balls_in_rooms: list[int],
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
        objects = (
            f"You have {n_rooms} rooms, {n_balls} balls, and {n_grippers} grippers"
        )

        def n_room_distributed() -> str:
            ball_counter = Counter(balls_in_rooms)
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

            case ("juggle", True):
                return f"{objects}. {sum(balls_in_grippers)} balls are distributed across the same number of grippers, and the rest are in the first room. The robby is in the first room."
            case ("juggle", False):
                return 'Your goal is to "juggle" the balls between the grippers that started with balls, such that the balls are in the same grippers as before, but shifted by one. The remaining balls should remain untouched.'

            case ("pickup", False):
                return "Your goal is to pick up all the balls with grippers."

            case ("drop_and_pickup", False):
                return "Your goal is to drop all the balls held by grippers in an empty room that the robby didn't start in, and pick up the balls that started in rooms not held by the robby."

            case ("holding", True):
                return f"{objects}. {max(sum(balls_in_grippers), 1)} balls are distributed across the same number of grippers, and the rest are in the first room. The robby is in the first room."
            case ("holding", False):
                return f"Your goal is to make sure robby is holding {max(sum(balls_in_grippers), 1)} balls."

            case _:
                raise ValueError(f"Invalid task: {task}")

    def get_task(
        self,
        init: str,
        goal: str,
        balls_in_grippers: list[int],
        balls_in_rooms: list[int],
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
        num_rooms = len(balls_in_rooms)
        num_balls = sum(balls_in_rooms + balls_in_grippers)
        num_grippers = len(balls_in_grippers)

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
            rooms=rooms,
            balls=balls,
            grippers=grippers,
            balls_in_grippers=balls_in_grippers,
            balls_in_rooms=balls_in_rooms,
            goal=False,
        )
        goal_predicates = getattr(self, goal)(
            rooms=rooms,
            balls=balls,
            grippers=grippers,
            balls_in_grippers=balls_in_grippers,
            balls_in_rooms=balls_in_rooms,
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
                    n_balls=num_balls,
                    n_grippers=num_grippers,
                    balls_in_grippers=balls_in_grippers,
                    balls_in_rooms=balls_in_rooms,
                ),
                "explicit": self.explicit_description(
                    init_predicates,
                    is_init=True,
                    n_rooms=num_rooms,
                    n_balls=num_balls,
                    n_grippers=num_grippers,
                    randomize=randomize,
                ),
            },
            "goal": {
                "abstract": self.abstract_description(
                    goal,
                    is_init=False,
                    n_rooms=num_rooms,
                    n_balls=num_balls,
                    n_grippers=num_grippers,
                    balls_in_grippers=balls_in_grippers,
                    balls_in_rooms=balls_in_rooms,
                ),
                "explicit": self.explicit_description(
                    goal_predicates,
                    is_init=False,
                    n_rooms=num_rooms,
                    n_balls=num_balls,
                    n_grippers=num_grippers,
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


class RoverSingleDatasetGenerator(DatasetGenerator):

    def __init__(self):

        super().__init__(
            "rover",
            """
            {%- set tense = "is" if is_init else "should be" -%}
            {%- if is_init -%}
                You have {{ n_waypoints }} waypoints.
                You have {{ n_stores}} stores.
                You have {{ n_cameras }} cameras.
                You have {{ n_modes }} modes.
                You have {{ n_objectives }} objectives.
            {%- else -%}
                Your goal is to have the following:
            {%- endif -%}
            {%- for predicate in predicates -%}
                {%- if predicate.name == "at_rover" %}
                    The rover {{ tense }} at {{ predicate.terms[0].name }}.
                {%- elif predicate.name == "at_lander" %}
                    The lander {{ tense }} at {{ predicate.terms[0].name }}.
                {%- elif predicate.name == "can_traverse" %}
                    The rover can traverse from {{ predicate.terms[0].name }} to {{ predicate.terms[1].name }}.
                {%- elif predicate.name == "empty" %}
                    Store {{ predicate.terms[0].name }} {{ tense }} empty.
                {%- elif predicate.name == "have_rock_analysis" %}
                    A rock analysis for waypoint {{ predicate.terms[0].name }} {{ tense }} obtained.
                {%- elif predicate.name == "have_soil_analysis" %}
                    A soil analysis for waypoint {{ predicate.terms[0].name }} {{ tense }} obtained.
                {%- elif predicate.name == "full" %}
                    Store {{ predicate.terms[0].name }} {{ tense }} full.
                {%- elif predicate.name == "supports" %}
                    Camera {{ predicate.terms[0].name }} supports {{ predicate.terms[1].name }} mode.
                {%- elif predicate.name == "available" %}
                    The rover {{ tense }} available.
                {%- elif predicate.name == "visible" %}
                    The waypoint {{ predicate.terms[0].name }} {{ tense }} visible from waypoint {{ predicate.terms[1].name }}.
                {%- elif predicate.name == "have_image" %}
                    An image of objective {{ predicate.terms[0].name }} {{ tense }} taken in mode {{ predicate.terms[1].name }}.
                {%- elif predicate.name == "communicated_soil_data" %}
                    The soil data from waypoint {{ predicate.terms[0].name }} {{ tense }} communicated.
                {%- elif predicate.name == "communicated_rock_data" %}
                    The rock data from waypoint {{ predicate.terms[0].name }} {{ tense }} communicated.
                {%- elif predicate.name == "communicated_image_data" %}
                    An image of objective {{ predicate.terms[0].name }} {{ tense }} communicated in mode {{ predicate.terms[1].name }}.
                {%- elif predicate.name == "at_rock_sample" %}
                    A rock sample {{ tense }} available at {{ predicate.terms[0].name }}.
                {%- elif predicate.name == "at_soil_sample" %}
                    A soil sample {{ tense }} available at {{ predicate.terms[0].name }}.
                {%- elif predicate.name == "visible_from" %}
                    The bjective {{ predicate.terms[0].name }} {{ tense }} visible from {{ predicate.terms[1].name }}.
                {%- elif predicate.name == "channel_free" %}
                    Channel {{ tense }} free.
                {%- endif -%}
            {%- endfor -%}""",
        )

    def abstract_description(self, task: str, is_init: bool = False, **kwargs) -> str:
        """Generate an abstract description of the state.

        Args:
            task (str): The task to describe.
            is_init (bool, optional): Whether the description is for the initial
                state. Defaults to False.

        Raises:
            ValueError: If the task is invalid/unsupported.

        Returns:
            str: State description.
        """
        match task, is_init:
            case ("line_navigate", True):
                return ""


class FloorTileDatasetGenerator(DatasetGenerator):

    def __init__(self):
        # predicates are robot-at, up, right, painted, robot-has, and available-color.
        super().__init__(
            "floor-tile",
            """
            {%- set tense = "is" if is_init else "should be" -%}
            {%- set has_robot = "has" if is_init else "should have" -%}
            {%- if is_init -%}
                You have {{ n_robots }} robot{{ "s" if n_robots > 1 else "" }}.
                You have {{ n_tiles }} tile{{ "s" if n_tiles > 1 else "" }}.
                You have {{ n_colors }} color{{ "s" if n_colors > 1 else "" }}.
            {%- else -%}
                Your goal is to have the following:
            {%- endif -%}
            {%- for predicate in predicates -%}
                {%- if predicate.name == "robot-at" %}
                    The robot {{ tense }} at tile {{ predicate.terms[0].name }}.
                {%- elif predicate.name == "up" %}
                    Tile {{ predicate.terms[1].name }} {{ tense }} is above tile {{ predicate.terms[0].name }}.
                {%- elif predicate.name == "right" %}
                    Tile {{ predicate.terms[1].name }} {{ tense }} is to the right of tile {{ predicate.terms[0].name }}.
                {%- elif predicate.name == "painted" %}
                    Tile {{ predicate.terms[0].name }} {{ tense }} painted with color {{ predicate.terms[1].name }}.
                {%- elif predicate.name == "robot-has" %}
                    The robot {{ predicate.terms[0].name }} {{ has_robot }} color {{ predicate.terms[1].name }}.
                {%- elif predicate.name == "available-color" %}
                    Color {{ predicate.terms[0].name }} {{ tense }} available.
                {%- endif -%}
            {%- endfor -%}""",
        )

    def join_elements(self, lst: list) -> str:
        match lst:
            case []:
                return ""
            case [l1]:
                return l1
            case [l1, l2]:
                return f"{l1} and {l2}"
            case _:
                join_str = ", ".join(lst[:-1])
                last = lst[-1]

                return f"{join_str}, and {last}"

    def disconnected_rows(
        self,
        robots: list[Constant],
        tiles: list[Constant],
        colors: list[Constant],
        goal: bool = False,
        num_cols: Iterable[int] = (3, 3, 3),
        **kwargs,
    ) -> list[Predicate]:
        """Task where all tiles are disconnected.

        Args:
            robots (list[Constant]): List of all available robots.
            tiles (list[Constant]): List of all tiles.
            colors (list[Constant]): List of all colors.
            goal (bool, optional): Whether to return goal state. Defaults to False.

        Returns:
            list[Predicate]: List of predicates describing the state.
        """
        assert (
            len(colors) == len(robots) == len(num_cols)
        ), f"Invalid number of robots{len(robots)}, colors{len(colors)}, and num_cols{num_cols}"
        assert sum(num_cols) == len(tiles)
        assert min(num_cols) > 0
        assert len(num_cols) > 1

        tiles_iter = iter(tiles)
        grid = [[next(tiles_iter) for _ in range(n)] for n in num_cols]

        predicates = []

        if not goal:
            # set up the grid
            for row in grid:
                for i in range(len(row) - 1):
                    predicates.append(Predicate("right", row[i + 1], row[i]))

            # place robots
            for robot, color, row in zip(robots, colors, grid):
                predicates.append(Predicate("robot-has", robot, color))
                predicates.append(Predicate("robot-at", robot, row[0]))

        # paint ends of each row

        for row in grid:
            predicates.append(Predicate("painted", row[0], colors[0]))
            if len(row) > 1:
                predicates.append(Predicate("painted", row[-1], colors[0]))

        return predicates

    def grid(
        self,
        robots: list[Constant],
        tiles: list[Constant],
        colors: list[Constant],
        goal: bool = False,
        grid_size: list[int] = None,
        robot_data: list[dict[str, int | list[int]]] = None,
        **kwargs,
    ):
        if grid_size is None or robot_data is None:
            raise ValueError("Grid size and robot data must be provided")
        if goal:
            raise ValueError("Grid task is only supported as an initial state")

        num_rows, num_cols = grid_size
        tiles_iter = iter(tiles)

        grid = [[next(tiles_iter) for _ in range(num_cols)] for _ in range(num_rows)]

        predicates = []

        for i in range(num_rows):
            for j in range(num_cols):
                if i > 0:
                    predicates.append(Predicate("up", grid[i][j], grid[i - 1][j]))
                if j < num_cols - 1:
                    predicates.append(Predicate("right", grid[i][j], grid[i][j + 1]))

        for robot, data in zip(robots, robot_data):
            predicates.append(Predicate("robot-has", robot, colors[data["color"]]))
            predicates.append(Predicate("robot-at", robot, grid[data["pos"][0]][data["pos"][1]]))

        for color in colors:
            predicates.append(Predicate("available-color", color))

        return predicates

    def rings(
        self,
        robots: list[Constant],
        tiles: list[Constant],
        colors: list[Constant],
        goal: bool = False,
        grid_size: list[int] = None,
        robot_data: list[dict[str, int | list[int]]] = None,
        init_task: str = None,
        **kwargs,
    ) -> list[Predicate]:
        """Task where all tiles are arranged in rings.

        Args:
            robots (list[Constant]): List of all available robots.
            tiles (list[Constant]): List of all tiles.
            colors (list[Constant]): List of all colors.
            goal (bool, optional): Whether to return goal state. Defaults to False.
            num_rings (int, optional): Number of rings. Defaults to 3.

        Returns:
            list[Predicate]: List of predicates describing the state.
        """
        if grid_size is None or robot_data is None:
            raise ValueError("Grid size and robot data must be provided")
        if goal and init_task not in ("rings",):
            raise ValueError("Rings goal task must have rings init task")
        assert grid_size[0] == grid_size[1]

        grid_size = grid_size[0]
        num_rings = round(grid_size / 2 + 0.5)
        assert len(tiles) == grid_size**2
        assert len(colors) == num_rings > 1
        assert len(robots) == len(robot_data)

        predicates = [Predicate("available-color", color) for color in colors]

        tiles_iter = iter(tiles)
        grid = [[next(tiles_iter) for _ in range(grid_size)] for _ in range(grid_size)]

        if goal:
            # paint all the tiles in rings
            for i in range(grid_size):
                for j in range(grid_size):
                    ring = min(i, j, grid_size - i - 1, grid_size - j - 1)
                    predicates.append(Predicate("painted", grid[i][j], colors[ring]))
        else:
            # robot positions, top-left corner and center
            for robot, data in zip(robots, robot_data):
                predicates.append(Predicate("robot-has", robot, colors[data["color"]]))
                predicates.append(
                    Predicate("robot-at", robot, grid[data["pos"][0]][data["pos"][1]])
                )

            # grid position predicates
            for i in range(grid_size):
                for j in range(grid_size):
                    if i > 0:
                        # second element is above the first
                        predicates.append(Predicate("up", grid[i][j], grid[i - 1][j]))
                    if j < grid_size - 1:
                        # second element is to the right of the first
                        predicates.append(
                            Predicate("right", grid[i][j], grid[i][j + 1])
                        )

        return predicates

    def checkerboard(
        self,
        robots: list[Constant],
        tiles: list[Constant],
        colors: list[Constant],
        goal: bool = False,
        grid_size: list[int] = None,
        **kwargs,
    ) -> list[Predicate]:
        if grid_size is None:
            raise ValueError("Grid size must be provided")
        if not goal:
            raise ValueError("Checkerboard task is only supported as a goal state")
        if len(colors) != 2:
            raise ValueError("Checkerboard task requires exactly 2 colors")

        grid_size_x, grid_size_y = grid_size
        assert len(tiles) == grid_size_x * grid_size_y

        tiles_iter = iter(tiles)
        grid = [[next(tiles_iter) for _ in range(grid_size_x)] for _ in range(grid_size_y)]

        predicates = []

        for i in range(grid_size_y):
            for j in range(grid_size_x):
                if (i + j) % 2 == 0:
                    color = colors[0]
                else:
                    color = colors[1]
                predicates.append(Predicate("painted", grid[i][j], color))

        return predicates

    def all_different(
        self,
        robots: list[Constant],
        tiles: list[Constant],
        colors: list[Constant],
        goal: bool = False,
        **kwargs,
    ) -> list[Predicate]:
        if not goal:
            raise ValueError("All different task is only supported as a goal state")
        if len(colors) != len(tiles):
            raise ValueError("All different task requires exactly one color per tile")

        predicates = []
        for tile, color in zip(tiles, colors):
            predicates.append(Predicate("painted", tile, color))

        return predicates

    def paint_all(
        self,
        robots: list[Constant],
        tiles: list[Constant],
        colors: list[Constant],
        goal: bool = False,
        **kwargs,
    ) -> list[Predicate]:
        if not goal:
            raise ValueError("Paint all task is only supported as a goal state")
        if len(colors) != 1:
            raise ValueError("Paint all task requires exactly one color")

        predicates = []
        for tile in tiles:
            predicates.append(Predicate("painted", tile, colors[0]))

        return predicates

    def abstract_description(
        self,
        task: str,
        n_robots: int,
        n_tiles: int,
        n_colors: int,
        is_init: bool = False,
        **kwargs,
    ) -> str:
        """Generate an abstract description of the state.

        Args:
            task (str): The task to describe.
            n_robots (int): Number of robots.
            n_tiles (int): Number of tiles.
            n_colors (int): Number of colors.
            is_init (bool, optional): Whether the description is for the initial
                state. Defaults to False.

        Returns:
            str: State description.
        """

        def get_robot_ring_string(
            grid_size_x: int,
            grid_size_y: int,
            robot_data: list[dict, str, int | list[int]],
        ) -> str:
            robot_ring_string = ""
            for i, data in enumerate(robot_data):
                # TODO: check strings
                pos_x, pos_y = data["pos"]
                if pos_x == pos_y and pos_x <= grid_size_x / 2:
                    pos_str = f"the top-left corner of the {int_to_ordinal(pos_x + 1)} ring from the outside"
                elif pos_x == pos_y and pos_x > grid_size_x / 2:
                    pos_str = f"the bottom-right corner of the {int_to_ordinal(pos_x + 1)} ring from the outside"
                elif pos_x == grid_size_x - pos_y - 1:
                    pos_str = f"the top-right corner of the {int_to_ordinal(pos_x + 1)} ring from the outside"
                elif pos_y == grid_size_y - pos_x - 1:
                    pos_str = f"the bottom-left corner of the {int_to_ordinal(pos_x + 1)} ring from the outside"
                else:
                    pos_str = f"{int_to_ordinal(pos_x + 1)} row and {int_to_ordinal(pos_y + 1)} column"
                robot_ring_string += f"The {int_to_ordinal(i + 1)} robot is at the {pos_str}, and has the {int_to_ordinal(data['color'] + 1)} color. "

            return robot_ring_string

        def get_robot_grid_string(
            grid_size_x: int,
            grid_size_y: int,
            robot_data: list[dict, str, int | list[int]],
        ) -> str:
            robot_grid_string = ""
            for i, data in enumerate(robot_data):
                pos_x, pos_y = data["pos"]
                robot_grid_string += f"The {int_to_ordinal(i + 1)} robot is at the {int_to_ordinal(pos_y + 1)} row and {int_to_ordinal(pos_x + 1)} column, and has the {int_to_ordinal(data['color'] + 1)} color. "

            return robot_grid_string

        match task, is_init:
            case ("disconnected_rows", True):
                num_cols = kwargs.get("num_cols")
                if all(n == num_cols[0] for n in num_cols):
                    num_cols_str = f"{num_cols[0]}"
                else:
                    num_cols_str = self.join_elements([str(n) for n in num_cols])
                return f"You have {n_tiles} unpainted tiles arranged in disconnected rows. There are {len(num_cols)} rows, each with {num_cols_str} tiles. At the start (tile on the left end) of each row is a robot, each of which has a different color. No colors are available."
            case ("disconnected_rows", False):
                return "Your goal is to paint the ends of each row."

            case ("grid", True):
                num_rows, num_cols = kwargs.get("grid_size")
                # TODO: check x vs y for col vs row
                return f"You have {n_colors} colors and {n_tiles} unpainted tiles arranged in a grid with {num_rows} rows and {num_cols} columns. All colors are available."

            case ("rings", True):
                grid_size_x, grid_size_y = kwargs.get("grid_size")
                assert grid_size_x == grid_size_y
                num_rings = round(grid_size_x / 2 + 0.5)
                robot_ring_string = get_robot_ring_string(
                    grid_size_x,
                    grid_size_y,
                    kwargs.get("robot_data"),
                )
                return f"You have {n_colors} colors and {n_tiles} unpainted tiles arranged in a {grid_size_x}x{grid_size_y} grid, in {num_rings} rings. {robot_ring_string}All colors are available."

            case ("rings", False):
                return "Your goal is to paint all the tiles such that each ring has a different color."

            case ("paint_all", False):
                return "Your goal is to paint all the tiles with the same color."

            case ("checkerboard", False):
                return "Your goal is to paint the tiles in a checkerboard pattern."

            case ("all_different", False):
                return "Your goal is to paint each tile with a different color."

            case _:
                raise ValueError(f"Invalid task: {task}")

    def get_task(
        self,
        init: str,
        goal: str,
        n_robots: int,
        n_tiles: int,
        n_colors: int,
        randomize: bool = True,
        **kwargs,
    ) -> tuple[Problem, dict[str, dict[str, str]]]:
        """Generate a floor tile task.

        Args:
            init (str): Initial state setting type.
            goal (str): Goal state setting type.
            n_robots (int): Number of robots.
            n_tiles (int): Number of tiles.
            n_colors (int): Number of colors.
            randomize (bool, optional): Whether to randomize the order of the
                tiles. Defaults to True.

        Returns:
            tuple[Problem, dict[str, dict[str, str]]]: PDDL problem and descriptions.
        """
        robots = [Constant(f"robot{i + 1}", type_tag="robot") for i in range(n_robots)]
        tiles = [Constant(f"tile{i + 1}", type_tag="tile") for i in range(n_tiles)]
        colors = [Constant(f"color{i + 1}", type_tag="color") for i in range(n_colors)]
        constants = robots + tiles + colors

        init_predicates = getattr(self, init)(
            robots=robots,
            tiles=tiles,
            colors=colors,
            init_task=init,
            goal_task=goal,
            **kwargs,
        )
        goal_predicates = getattr(self, goal)(
            robots=robots,
            tiles=tiles,
            colors=colors,
            goal=True,
            init_task=init,
            goal_task=goal,
            **kwargs,
        )

        problem = Problem(
            name=f"{init}_to_{goal}_{n_robots}_{n_tiles}_{n_colors}",
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
                    n_robots=n_robots,
                    n_tiles=n_tiles,
                    n_colors=n_colors,
                    **kwargs,
                ),
                "explicit": self.explicit_description(
                    init_predicates,
                    is_init=True,
                    n_robots=n_robots,
                    n_tiles=n_tiles,
                    n_colors=n_colors,
                    randomize=randomize,
                ),
            },
            "goal": {
                "abstract": self.abstract_description(
                    goal,
                    is_init=False,
                    n_robots=n_robots,
                    n_tiles=n_tiles,
                    n_colors=n_colors,
                    **kwargs,
                ),
                "explicit": self.explicit_description(
                    goal_predicates,
                    is_init=False,
                    n_robots=n_robots,
                    n_tiles=n_tiles,
                    n_colors=n_colors,
                    randomize=randomize,
                ),
            },
        }

        data = {
            "num_objects": len(constants),
            "init_num_propositions": len(init_predicates),
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
            case "rover-single":
                generator = RoverSingleDatasetGenerator()
            case "floor-tile":
                generator = FloorTileDatasetGenerator()
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
                        kwargs: dict = {}
                        for kwargs in domain_cfg["tasks"].get(source, kwargs):
                            try:
                                problem, descriptions, data = generator.get_task(
                                    start,
                                    end,
                                    **kwargs,
                                    randomize=config.get("randomize_predicates", False),
                                )
                            except ValueError:
                                continue

                            # ground truth PDDL problem
                            problem_str = pddl_formatter.problem_to_string(problem)
                            args = [
                                arg
                                for kw, args in kwargs.items()
                                for arg in (
                                    (kw, *args)
                                    if isinstance(args, list)
                                    else [kw, args]
                                )
                            ]
                            arg_str = "_".join(str(arg) for arg in args)

                            problem_name = f"{task}_{arg_str}"

                            # problem descriptions for each combination of abstract/explicit initial and goal states
                            for init_desc, goal_desc in itertools.product(
                                ("abstract", "explicit"),
                                ("abstract", "explicit"),
                            ):
                                # hard-code juggle right now
                                # TODO: make sure this is correct for all fully_strict or just juggle
                                if (
                                    start in fully_strict
                                    or end in fully_strict
                                    or start in strictly_both
                                ) and init_desc != goal_desc:
                                    continue
                                problem_desc = [
                                    descriptions["init"][init_desc],
                                    descriptions["goal"][goal_desc],
                                ]
                                problem_desc_str = "\n".join(problem_desc)
                                problems.append(
                                    {
                                        "name": f"{domain_cfg['name']}_{problem_name}",
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
