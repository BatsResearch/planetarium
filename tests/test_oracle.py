import pytest

from planetarium import builder, graph, oracle


@pytest.fixture
def blocksworld_fully_specified():
    """
    Fixture providing a fully specified blocksworld problem.
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
            (clear b5)
            (on-table b6)
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
def blocksworld_missing_clears():
    """
    Fixture providing a fully specified blocksworld problem missing nonessential predicates.
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
            (clear b5)
            (on-table b6)
            (clear b6)
        )
        (:goal
            (and
            (on-table b1)

            (on-table b2)
            (on b3 b2)

            (on-table b4)
            (on b5 b4)
            (on b6 b5)
            )
        )
    )
    """


@pytest.fixture
def blocksworld_missing_ontables():
    """
    Fixture providing a fully specified blocksworld problem missing nonessential predicates.
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
            (clear b5)
            (on-table b6)
            (clear b6)
        )
        (:goal
            (and
            (clear b1)

            (clear b3)
            (on b3 b2)

            (clear b6)
            (on b5 b4)
            (on b6 b5)
            (arm-empty)
            )
        )
    )
    """


@pytest.fixture
def blocksworld_underspecified():
    """
    Fixture providing an underspecified blocksworld problem.
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
            (clear b5)
            (on-table b6)
            (clear b6)
        )
        (:goal
            (and
            (on-table b1)

            (on b3 b2)

            (on b5 b4)
            (on b6 b5)
            )
        )
    )
    """


@pytest.fixture
def blocksworld_underspecified_arm():
    """
    Fixture providing an underspecified blocksworld problem.
    """
    return """
    (define (problem staircase)
        (:domain blocksworld)
        (:objects
            b1 b2 b4 b5 b6
        )
        (:init
            (arm-empty)
            (on-table b1)
            (clear b1)
            (on-table b2)
            (clear b2)
            (on-table b4)
            (clear b4)
            (on-table b5)
            (clear b5)
            (on-table b6)
            (clear b6)
        )
        (:goal
            (and

            (on b5 b4)
            (on b6 b5)
            )
        )
    )
    """


@pytest.fixture
def blocksworld_invalid_1():
    """
    Fixture providing an invalid blocksworld problem.
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
            (clear b5)
            (on-table b6)
            (clear b6)
        )
        (:goal
            (and
            (on-table b1)

            (on b3 b2)

            (on b5 b4)
            (on b6 b5)
            (on b5 b2)
            )
        )
    )
    """


@pytest.fixture
def blocksworld_invalid_2():
    """
    Fixture providing an invalid blocksworld problem.
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
            (clear b5)
            (on-table b6)
            (clear b6)
        )
        (:goal
            (and
            (on b1 b2)
            (on b2 b3)
            (on b3 b4)
            (on b4 b5)
            (on b5 b6)
            (on b6 b1)
            )
        )
    )
    """


@pytest.fixture
def blocksworld_invalid_3():
    """
    Fixture providing an invalid blocksworld problem.
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
            (clear b5)
            (on-table b6)
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
            (clear b4)

            (holding b5)
            (on b6 b5)
            (clear b6)
            )
        )
    )
    """


@pytest.fixture
def blocksworld_holding():
    """
    Fixture providing a fully specified blocksworld problem.
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
            (clear b5)
            (on-table b6)
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
            (holding b6)
            )
        )
    )
    """


@pytest.fixture
def blocksworld_stack_to_holding():
    """
    Fixture providing a fully specified blocksworld problem.
    """
    return """
    (define (problem staircase)
        (:domain blocksworld)
        (:objects
            b1 b2 b3 b4 b5 b6
        )
        (:init
            (holding b1)
            (on-table b2)
            (on b3 b2)
            (on b4 b3)
            (on b5 b4)
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
            (holding b6)
            )
        )
    )
    """


"""
GRIPPER FIXTURES
"""


@pytest.fixture
def gripper_fully_specified():
    return """
    (define (problem gripper)
        (:domain gripper)
        (:objects
            room1 room2 room3 ball1 ball2 ball3 gripper1 gripper2
        )
        (:init
            (room room1)
            (room room2)
            (room room3)
            (ball ball1)
            (ball ball2)
            (ball ball3)
            (gripper gripper1)
            (gripper gripper2)
            (at-robby room1)
            (at ball2 room2)
            (at ball3 room3)
            (free gripper1)
            (carry ball1 gripper2)
        )
        (:goal
            (and
                (room room1)
                (room room2)
                (room room3)
                (ball ball1)
                (ball ball2)
                (ball ball3)
                (gripper gripper1)
                (gripper gripper2)
                (at ball1 room3)
                (at ball2 room3)
                (at ball3 room3)
                (free gripper1)
                (free gripper2)
                (at-robby room1)
            )
        )
    )
    """


@pytest.fixture
def gripper_no_robby():
    return """
    (define (problem gripper)
        (:domain gripper)
        (:objects
            room1 room2 room3 ball1 ball2 ball3 gripper1 gripper2
        )
        (:init
            (room room1)
            (room room2)
            (room room3)
            (ball ball1)
            (ball ball2)
            (ball ball3)
            (gripper gripper1)
            (gripper gripper2)
            (at-robby room1)
            (at ball2 room2)
            (at ball3 room3)
            (free gripper1)
            (carry ball1 gripper2)
        )
        (:goal
            (and
                (room room1)
                (room room2)
                (room room3)
                (ball ball1)
                (ball ball2)
                (ball ball3)
                (gripper gripper1)
                (gripper gripper2)
                (at ball1 room3)
                (at ball2 room3)
                (at ball3 room3)
                (free gripper1)
                (free gripper2)
            )
        )
    )
    """


@pytest.fixture
def gripper_no_robby_init():
    return """
    (define (problem gripper)
        (:domain gripper)
        (:objects
            room1 room2 room3 ball1 ball2 ball3 gripper1 gripper2
        )
        (:init
            (room room1)
            (room room2)
            (room room3)
            (ball ball1)
            (ball ball2)
            (ball ball3)
            (gripper gripper1)
            (gripper gripper2)
            (at ball2 room2)
            (at ball3 room3)
            (free gripper1)
            (carry ball1 gripper2)
        )
        (:goal
            (and
                (room room1)
                (room room2)
                (room room3)
                (ball ball1)
                (ball ball2)
                (ball ball3)
                (gripper gripper1)
                (gripper gripper2)
                (at ball1 room3)
                (at ball2 room3)
                (at ball3 room3)
                (free gripper1)
                (free gripper2)
            )
        )
    )
    """


@pytest.fixture
def gripper_no_goal_types():
    return """
    (define (problem gripper)
        (:domain gripper)
        (:objects
            room1 room2 room3 ball1 ball2 ball3 gripper1 gripper2
        )
        (:init
            (room room1)
            (room room2)
            (room room3)
            (ball ball1)
            (ball ball2)
            (ball ball3)
            (gripper gripper1)
            (gripper gripper2)
            (at-robby room1)
            (at ball2 room2)
            (at ball3 room3)
            (free gripper1)
            (carry ball1 gripper2)
        )
        (:goal
            (and
                (at ball1 room3)
                (at ball2 room3)
                (at ball3 room3)
                (free gripper1)
                (free gripper2)
                (at-robby room1)
            )
        )
    )
    """


@pytest.fixture
def gripper_fully_specified_not_strict():
    return """
    (define (problem gripper)
        (:domain gripper)
        (:objects
            room1 room2 room3 ball1 ball2 ball3 gripper1 gripper2
        )
        (:init
            (room room1)
            (room room2)
            (room room3)
            (ball ball1)
            (ball ball2)
            (ball ball3)
            (gripper gripper1)
            (gripper gripper2)
            (at-robby room1)
            (at ball2 room2)
            (at ball3 room3)
            (free gripper1)
            (carry ball1 gripper2)
        )
        (:goal
            (and
                (at ball1 room3)
                (at ball2 room3)
                (at ball3 room3)
                (free gripper1)
                (at-robby room1)
            )
        )
    )
    """


@pytest.fixture
def gripper_underspecified_1():
    return """
    (define (problem gripper)
        (:domain gripper)
        (:objects
            room1 room2 room3 ball1 ball2 ball3 gripper1 gripper2
        )
        (:init
            (room room1)
            (room room2)
            (room room3)
            (ball ball1)
            (ball ball2)
            (ball ball3)
            (gripper gripper1)
            (gripper gripper2)
            (at-robby room1)
            (at ball2 room2)
            (at ball3 room3)
            (free gripper1)
            (carry ball1 gripper2)
        )
        (:goal
            (and
                (room room1)
                (room room2)
                (room room3)
                (ball ball1)
                (ball ball2)
                (ball ball3)
                (gripper gripper1)
                (gripper gripper2)
                (at ball2 room3)
                (at ball3 room3)
                (free gripper1)
                (free gripper2)
                (at-robby room1)
            )
        )
    )
    """


@pytest.fixture
def gripper_underspecified_2():
    return """
    (define (problem gripper)
        (:domain gripper)
        (:objects
            room1 room2 room3 ball1 ball2 ball3 gripper1 gripper2
        )
        (:init
            (room room1)
            (room room2)
            (room room3)
            (ball ball1)
            (ball ball2)
            (ball ball3)
            (gripper gripper1)
            (gripper gripper2)
            (at-robby room1)
            (at ball2 room2)
            (at ball3 room3)
            (free gripper1)
            (carry ball1 gripper2)
        )
        (:goal
            (and
                (room room1)
                (room room2)
                (room room3)
                (ball ball1)
                (ball ball2)
                (ball ball3)
                (gripper gripper1)
                (gripper gripper2)
                (at ball2 room3)
                (at ball3 room3)
                (free gripper1)
                (free gripper2)
            )
        )
    )
    """


@pytest.fixture
def gripper_underspecified_3():
    return """
    (define (problem gripper)
        (:domain gripper)
        (:objects
            room1 room2 room3 ball1 ball2 ball3 gripper1 gripper2
        )
        (:init
            (room room1)
            (room room2)
            (room room3)
            (ball ball1)
            (ball ball2)
            (ball ball3)
            (gripper gripper1)
            (gripper gripper2)
            (at-robby room1)
            (at ball2 room2)
            (at ball3 room3)
            (free gripper1)
            (carry ball1 gripper2)
        )
        (:goal
            (and
                (room room1)
                (room room2)
                (room room3)
                (gripper gripper1)
                (gripper gripper2)
                (at ball2 room3)
                (at ball3 room3)
                (free gripper1)
            )
        )
    )
    """


@pytest.fixture
def gripper_invalid():
    return """
    (define (problem gripper)
        (:domain gripper)
        (:objects
            room1 room2 room3 ball1 ball2 ball3 gripper1 gripper2
        )
        (:init
            (room room1)
            (room room2)
            (room room3)
            (ball ball1)
            (ball ball2)
            (ball ball3)
            (gripper gripper1)
            (gripper gripper2)
            (at-robby room1)
            (at ball2 room2)
            (at ball3 room3)
            (free gripper1)
            (carry ball1 gripper2)
        )
        (:goal
            (and
                (room room1)
                (room room2)
                (room room3)
                (gripper gripper1)
                (gripper gripper2)
                (at ball2 room3)
                (at ball3 room3)
                (at room3 ball3)
                (free gripper1)
            )
        )
    )
    """


@pytest.fixture
def gripper_inconsistent_typing():
    return """
    (define (problem gripper)
        (:domain gripper)
        (:objects
            room1 room2 room3 ball1 ball2 ball3 gripper1 gripper2
        )
        (:init
            (room room1)
            (room room2)
            (room room3)
            (ball ball1)
            (ball ball2)
            (ball ball3)
            (gripper gripper1)
            (gripper gripper2)
            (at-robby room1)
            (at ball2 room2)
            (at ball3 room3)
            (free gripper1)
            (carry ball1 gripper2)
        )
        (:goal
            (and
                (room room1)
                (ball room2)
                (room room3)
                (ball ball1)
                (ball ball2)
                (ball ball3)
                (gripper gripper1)
                (gripper gripper2)
                (at ball1 room3)
                (at ball2 room3)
                (at ball3 room3)
                (free gripper1)
                (free gripper2)
                (at-robby room1)
            )
        )
    )
    """


@pytest.fixture
def gripper_multiple_typing():
    return """
    (define (problem gripper)
        (:domain gripper)
        (:objects
            room1 room2 room3 ball1 ball2 ball3 gripper1 gripper2
        )
        (:init
            (room room1)
            (room room2)
            (room room3)
            (ball ball1)
            (ball ball2)
            (room ball2)
            (ball ball3)
            (gripper gripper1)
            (gripper gripper2)
            (at-robby room1)
            (at ball2 room2)
            (at ball3 room3)
            (free gripper1)
            (carry ball1 gripper2)
        )
        (:goal
            (and
                (room room1)
                (room room2)
                (room room3)
                (ball ball1)
                (ball ball2)
                (ball ball3)
                (gripper gripper1)
                (gripper gripper2)
                (at ball1 room3)
                (at ball2 room3)
                (at ball3 room3)
                (free gripper1)
                (free gripper2)
                (at-robby room1)
            )
        )
    )
    """


@pytest.fixture
def gripper_missing_typing():
    return """
    (define (problem gripper)
        (:domain gripper)
        (:objects
            room1 room2 room3 ball1 ball2 ball3 gripper1 gripper2
        )
        (:init
            (room room1)
            (room room2)
            (room room3)
            (ball ball1)
            (ball ball3)
            (gripper gripper1)
            (gripper gripper2)
            (at-robby room1)
            (at ball2 room2)
            (at ball3 room3)
            (free gripper1)
            (carry ball1 gripper2)
        )
        (:goal
            (and
                (room room1)
                (room room2)
                (room room3)
                (ball ball1)
                (ball ball2)
                (ball ball3)
                (gripper gripper1)
                (gripper gripper2)
                (at ball1 room3)
                (at ball2 room3)
                (at ball3 room3)
                (free gripper1)
                (free gripper2)
                (at-robby room1)
            )
        )
    )
    """


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
