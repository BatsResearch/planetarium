from itertools import product
import pytest

import planetarium

from .test_oracle import (
    blocksworld_underspecified,
    blocksworld_missing_clears,
    blocksworld_missing_ontables,
    blocksworld_fully_specified,
    blocksworld_invalid_1,
)


@pytest.fixture
def blocksworld_wrong_init():
    """
    Fixture providing a fully specified blocksworld problem with wrong init.
    """
    return """
    (define (problem staircase)
        (:domain blocksworld)
        (:objects
            b1 b2 b3 b4 b5 b6
        )
        (:init
            (arm-empty)
            (on-table b1)
            (clear b1)
            (on-table b2)
            (clear b2)
            (on-table b3)
            (clear b3)
            (on-table b4)
            (clear b4)
            (on-table b5)
            (on b6 b5)
            (clear b6)
        )
        (:goal
            (and
            (on-table b1)
            (clear b1)

            (on-table b2)
            (on b3 b2)
            (clear b3)

            (on-table b4)
            (on b5 b4)
            (on b6 b5)
            (clear b6)
            )
        )
    )
    """


@pytest.fixture
def blocksworld_unsolveable():
    """
    Fixture providing a fully specified blocksworld problem with wrong init.
    """
    return """
    (define (problem staircase)
        (:domain blocksworld)
        (:objects
            b1 b2 b3 b4 b5 b6
        )
        (:init
            (arm-empty)
            (on-table b1)
            (clear b1)
            (on-table b2)
            (clear b2)
            (on-table b3)
            (clear b3)
            (on-table b4)
            (clear b4)
            (on-table b5)
            (on b6 b5)
            (clear b6)
        )
        (:goal
            (and
            (on-table b1)
            (clear b1)

            (on-table b2)
            (on b3 b2)
            (clear b3)

            (on-table b4)
            (on b5 b4)
            (clear b5)
            (on b6 b5)
            (clear b6)
            )
        )
    )
    """


@pytest.fixture
def blocksworld_unparseable():
    """
    Fixture providing an unparseable blocksworld problem.
    """
    return """
    (define (problem staircase)
        (:domain blocksworld)
        (:objects
            b1 b2 b3 b4 b5 b6
        ))
        (:init
            (arm-empty)
            (on-table b1)
            (clear b1)
            (on-table b2)
            (clear b2)
            (on-table b3)
            (clear b3)
            (on-table b4)
            (clear b4)
            (on-table b5)
            (on b6 b5)
            (clear b6)
        )
        (:goal
            (and
            (on-table b1)
            (clear b1)

            (on-table b2)
            (on b3 b2)
            (clear b3)

            (on-table b4)
            (on b5 b4)
            (on b6 b5)
            (clear b6)
            )
        )
    )
    """


class TestEvaluate:
    """
    Test suite for the evaluation of PDDL problem descriptions.
    """

    def test_evaluate_equivalent(
        self,
        subtests,
        blocksworld_missing_clears,
        blocksworld_fully_specified,
        blocksworld_missing_ontables,
        blocksworld_underspecified,
    ):
        """
        Test if the evaluation of PDDL problem descriptions is correct.
        """
        descs = [
            ("blocksworld_missing_clears", blocksworld_missing_clears),
            ("blocksworld_fully_specified", blocksworld_fully_specified),
            ("blocksworld_missing_ontables", blocksworld_missing_ontables),
        ]
        for (name1, desc1), (name2, desc2) in product(descs, descs):
            with subtests.test(f"{name1} equals {name2}"):
                assert all(planetarium.evaluate(desc1, desc2))

        with subtests.test(
            "blocksworld_underspecified equals blocksworld_underspecified"
        ):
            assert all(
                planetarium.evaluate(
                    blocksworld_underspecified, blocksworld_underspecified
                )
            )

    def test_evaluate_inequivalent(
        self,
        subtests,
        blocksworld_missing_clears,
        blocksworld_fully_specified,
        blocksworld_missing_ontables,
        blocksworld_underspecified,
        blocksworld_wrong_init,
        blocksworld_unparseable,
        blocksworld_unsolveable,
    ):
        """
        Test if the evaluation of PDDL problem descriptions is correct.
        """
        descs = [
            ("blocksworld_missing_clears", blocksworld_missing_clears),
            ("blocksworld_fully_specified", blocksworld_fully_specified),
            ("blocksworld_missing_ontables", blocksworld_missing_ontables),
        ]
        for name, desc in descs:
            with subtests.test(f"{name} not equals blocksworld_underspecified"):
                assert planetarium.evaluate(desc, blocksworld_underspecified) == (
                    True,
                    True,
                    False,
                )

        with subtests.test(f"{name} not equals blocksworld_wrong_init"):
            assert planetarium.evaluate(desc, blocksworld_wrong_init) == (
                True,
                True,
                False,
            )
        with subtests.test(f"{name} not equals blocksworld_unparseable"):
            assert planetarium.evaluate(desc, blocksworld_unparseable) == (
                False,
                False,
                False,
            )
        with subtests.test(f"{name} not equals blocksworld_unsolveable"):
            assert planetarium.evaluate(desc, blocksworld_unsolveable) == (
                True,
                False,
                False,
            )

        with subtests.test(
            "blocksworld_underspecified not equals blocksworld_wrong_init"
        ):
            assert planetarium.evaluate(
                blocksworld_underspecified, blocksworld_wrong_init
            ) == (
                True,
                True,
                False,
            )
