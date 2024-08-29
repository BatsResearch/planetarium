import pytest

from planetarium import builder, graph, oracle

from .problem_fixtures import (
    blocksworld_fully_specified,
    blocksworld_missing_clears,
    blocksworld_missing_ontables,
    blocksworld_underspecified,
    blocksworld_underspecified_arm,
    blocksworld_holding,
    gripper_fully_specified,
    gripper_no_goal_types,
    gripper_fully_specified_not_strict,
    gripper_no_robby,
    gripper_underspecified_1,
    gripper_underspecified_2,
    gripper_underspecified_3,
    gripper_no_robby_init,
    rover_line,
    rover_line_fully_specified,
    rover_line_fully_specified_1,
    rover_line_missing_visible,
    rover_line_1,
    rover_line_2,
)


def reduce_and_inflate(scene: graph.SceneGraph) -> bool:
    """Respecify a scene and check if it is equal to the original.

    Args:
        scene (graph.SceneGraph): The scene to test

    Returns:
        bool: True if the respecified scene is equal to the original.
    """
    reduced = oracle.reduce(scene, domain=scene.domain)
    respecified = oracle.inflate(reduced, domain=scene.domain)
    return scene == respecified


class TestBlocksworldOracle:
    """
    Test suite for the blocksworld oracle.
    """

    def test_fully_specified(self, blocksworld_fully_specified):
        """
        Test the fully specified blocksworld problem.
        """
        problem = builder.build(blocksworld_fully_specified)
        full = oracle.fully_specify(problem)
        assert oracle.fully_specify(full) == full

    def test_missing_clears(self, blocksworld_missing_clears):
        """
        Test the fully specified blocksworld problem with missing clears.
        """
        problem = builder.build(blocksworld_missing_clears)
        full = oracle.fully_specify(problem)
        assert oracle.fully_specify(full) == full

    def test_missing_ontables(self, blocksworld_missing_ontables):
        """
        Test the fully specified blocksworld problem with missing clears.
        """
        problem = builder.build(blocksworld_missing_ontables)
        full = oracle.fully_specify(problem)
        assert oracle.fully_specify(full) == full

    def test_missing_ontables_and_clears(self, blocksworld_underspecified):
        """
        Test the fully specified blocksworld problem with missing clears.
        """
        problem = builder.build(blocksworld_underspecified)
        full = oracle.fully_specify(problem)
        assert oracle.fully_specify(full) == full

    def test_inflate(
        self,
        subtests,
        blocksworld_fully_specified,
        blocksworld_missing_clears,
        blocksworld_missing_ontables,
        blocksworld_underspecified,
        blocksworld_underspecified_arm,
        blocksworld_holding,
    ):
        """
        Test the inflate function.
        """

        for name, desc in {
            "blocksworld_fully_specified": blocksworld_fully_specified,
            "blocksworld_missing_clears": blocksworld_missing_clears,
            "blocksworld_missing_ontables": blocksworld_missing_ontables,
            "blocksworld_underspecified": blocksworld_underspecified,
            "blocksworld_underspecified_arm": blocksworld_underspecified_arm,
            "blocksworld_holding": blocksworld_holding,
        }.items():
            problem = builder.build(desc)
            init, goal = problem.decompose()
            with subtests.test(name):
                assert reduce_and_inflate(init)
                assert reduce_and_inflate(goal)
                assert reduce_and_inflate(problem)

                assert problem == oracle.inflate(
                    oracle.ReducedProblemGraph.join(
                        oracle.reduce(init),
                        oracle.reduce(goal),
                    )
                )


class TestGripperOracle:
    """
    Test suite for the gripper oracle.
    """

    def test_fully_specified(
        self,
        subtests,
        gripper_fully_specified,
        gripper_no_goal_types,
        gripper_fully_specified_not_strict,
    ):
        """
        Test the fully specified gripper problem.
        """
        descs = [
            ("gripper_fully_specified", gripper_fully_specified),
            ("gripper_no_goal_types", gripper_no_goal_types),
            ("gripper_fully_specified_not_strict", gripper_fully_specified_not_strict),
        ]
        for name, desc in descs:
            with subtests.test(name):
                problem = builder.build(desc)
                full = oracle.fully_specify(problem)
                assert oracle.fully_specify(full) == full

    def test_inflate(
        self,
        subtests,
        gripper_fully_specified,
        gripper_no_robby,
        gripper_underspecified_1,
        gripper_underspecified_2,
        gripper_underspecified_3,
        gripper_no_robby_init,
    ):
        """
        Test the inflate function.
        """

        descs = [
            ("gripper_fully_specified", gripper_fully_specified),
            ("gripper_no_robby", gripper_no_robby),
            ("gripper_underspecified_1", gripper_underspecified_1),
            ("gripper_underspecified_2", gripper_underspecified_2),
            ("gripper_underspecified_3", gripper_underspecified_3),
            ("gripper_no_robby_init", gripper_no_robby_init),
        ]

        for name, desc in descs:
            problem = builder.build(desc)
            init, goal = problem.decompose()
            with subtests.test(name):
                assert reduce_and_inflate(init)
                assert reduce_and_inflate(goal)
                assert reduce_and_inflate(problem)

    def test_underspecified(
        self,
        gripper_underspecified_1,
        gripper_underspecified_2,
    ):
        problem = builder.build(gripper_underspecified_1)
        full = oracle.fully_specify(problem)
        assert oracle.fully_specify(full) == full

        problem = builder.build(gripper_underspecified_2)
        full = oracle.fully_specify(problem)
        assert oracle.fully_specify(full) == full


class TestRoverOracle:
    """
    Test suite for the rover oracle.
    """

    def test_fully_specified(self, subtests, rover_line_fully_specified, rover_line_fully_specified_1):
        """
        Test the fully specified rover problem.
        """
        descs = [
            ("rover_line", rover_line_fully_specified),
            ("rover_line_1", rover_line_fully_specified_1),
        ]
        for name, desc in descs:
            with subtests.test(name):
                problem = builder.build(desc)
                full = oracle.fully_specify(problem)
                assert oracle.fully_specify(full) == problem

    def test_under_specified(
        self, subtests, rover_line, rover_line_missing_visible, rover_line_1,
        rover_line_2,
    ):
        """
        Test the under specified rover problem.
        """
        descs = [
            ("rover_line", rover_line),
            ("rover_line_missing_visible", rover_line_missing_visible),
            ("rover_line_1", rover_line_1),
        ]
        for name, desc in descs:
            with subtests.test(name):
                problem = builder.build(desc)
                full = oracle.fully_specify(problem)
                assert full != problem
                assert oracle.fully_specify(full) == full

    def test_inflate(self):
        pass


class TestUnsupportedDomain:
    def test_reduce_and_inflate(self, gripper_fully_specified):
        problem = builder.build(gripper_fully_specified)
        init, goal = problem.decompose()

        with pytest.raises(oracle.DomainNotSupportedError):
            oracle.reduce(init, domain="gripper-modified")
        with pytest.raises(oracle.DomainNotSupportedError):
            reduced = oracle.reduce(goal, domain="gripper")
            oracle.inflate(reduced, domain="gripper-modified")

    def test_fully_specify(self, gripper_fully_specified):
        problem = builder.build(gripper_fully_specified)
        with pytest.raises(oracle.DomainNotSupportedError):
            oracle.fully_specify(problem, domain="gripper-modified")

    def test_plan(self, gripper_fully_specified):
        problem = builder.build(gripper_fully_specified)
        with pytest.raises(oracle.DomainNotSupportedError):
            oracle.plan(problem, domain="gripper-modified")
