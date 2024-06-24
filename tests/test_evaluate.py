from itertools import product

from planetarium import evaluate

from .test_oracle import (
    blocksworld_underspecified,
    blocksworld_missing_clears,
    blocksworld_missing_ontables,
    blocksworld_fully_specified,
)


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
                assert all(evaluate.evaluate(desc1, desc2))

    def test_evaluate_inequivalent(
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
        for name1, desc1 in descs:
            with subtests.test(f"{name1} not equals blocksworld_underspecified"):
                assert evaluate.evaluate(desc1, blocksworld_underspecified) == (
                    True,
                    True,
                    False,
                )
