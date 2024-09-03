import pytest


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
def gripper_robby_at_last():
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
            (at-robby room3)
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


@pytest.fixture
def rover_line_fully_specified():
    return """
    (define (problem rover)
        (:domain rover)
        (:objects
            rover1 rover2 rover3 - rover
            site1 site2 site3 site4 site5 site6 - waypoint
        )
        (:init
            (can_traverse rover1 site1 site2)
            (can_traverse rover1 site2 site3)
            (can_traverse rover1 site3 site4)
            (can_traverse rover1 site4 site5)
            (can_traverse rover1 site5 site6)
            (can_traverse rover2 site1 site2)
            (can_traverse rover2 site2 site3)
            (can_traverse rover2 site3 site4)
            (can_traverse rover2 site4 site5)
            (can_traverse rover2 site5 site6)
            (can_traverse rover3 site1 site2)
            (can_traverse rover3 site2 site3)
            (can_traverse rover3 site3 site4)
            (can_traverse rover3 site4 site5)
            (can_traverse rover3 site5 site6)
            (visible site1 site2)
            (visible site2 site3)
            (visible site3 site4)
            (visible site4 site5)
            (visible site5 site6)
            (at rover1 site1)
            (at rover2 site1)
            (at rover3 site5)
            (available rover1)
            (available rover2)
            (available rover3)
        )
        (:goal
            (and
                (can_traverse rover1 site1 site2)
                (can_traverse rover1 site2 site3)
                (can_traverse rover1 site3 site4)
                (can_traverse rover1 site4 site5)
                (can_traverse rover1 site5 site6)
                (can_traverse rover2 site1 site2)
                (can_traverse rover2 site2 site3)
                (can_traverse rover2 site3 site4)
                (can_traverse rover2 site4 site5)
                (can_traverse rover2 site5 site6)
                (can_traverse rover3 site1 site2)
                (can_traverse rover3 site2 site3)
                (can_traverse rover3 site3 site4)
                (can_traverse rover3 site4 site5)
                (can_traverse rover3 site5 site6)
                (visible site1 site2)
                (visible site2 site3)
                (visible site3 site4)
                (visible site4 site5)
                (visible site5 site6)
                (at rover1 site6)
                (at rover2 site6)
                (at rover3 site6)
                (available rover1)
                (available rover2)
                (available rover3)
            )
        )
    )"""


@pytest.fixture
def rover_line():
    return """
    (define (problem rover)
        (:domain rover)
        (:objects
            rover1 rover2 rover3 - rover
            site1 site2 site3 site4 site5 site6 - waypoint
        )
        (:init
            (can_traverse rover1 site1 site2)
            (can_traverse rover1 site2 site3)
            (can_traverse rover1 site3 site4)
            (can_traverse rover1 site4 site5)
            (can_traverse rover1 site5 site6)
            (can_traverse rover2 site1 site2)
            (can_traverse rover2 site2 site3)
            (can_traverse rover2 site3 site4)
            (can_traverse rover2 site4 site5)
            (can_traverse rover2 site5 site6)
            (can_traverse rover3 site1 site2)
            (can_traverse rover3 site2 site3)
            (can_traverse rover3 site3 site4)
            (can_traverse rover3 site4 site5)
            (can_traverse rover3 site5 site6)
            (visible site1 site2)
            (visible site2 site3)
            (visible site3 site4)
            (visible site4 site5)
            (visible site5 site6)
            (at rover1 site1)
            (at rover2 site1)
            (at rover3 site5)
            (available rover1)
            (available rover2)
            (available rover3)
        )
        (:goal
            (and
                (at rover1 site6)
                (at rover2 site6)
                (at rover3 site6)
            )
        )
    )"""


@pytest.fixture
def rover_line_fully_specified_1():
    return """
    (define (problem rover)
        (:domain rover)
        (:objects
            rover1 rover2 rover3 - rover
            site1 site2 site3 site4 site5 site6 - waypoint
            lander1 lander2 - lander
            store1 store2 - store
            rgb - mode
            camera1 camera2 camera3 - camera
            objective1 objective2 objective3 - objective
        )
        (:init
            (can_traverse rover1 site1 site2)
            (can_traverse rover1 site2 site3)
            (can_traverse rover1 site3 site4)
            (can_traverse rover1 site4 site5)
            (can_traverse rover1 site5 site6)
            (can_traverse rover2 site1 site2)
            (can_traverse rover2 site2 site3)
            (can_traverse rover2 site3 site4)
            (can_traverse rover2 site4 site5)
            (can_traverse rover2 site5 site6)
            (can_traverse rover3 site1 site2)
            (can_traverse rover3 site2 site3)
            (can_traverse rover3 site3 site4)
            (can_traverse rover3 site4 site5)
            (can_traverse rover3 site5 site6)
            (visible site1 site2)
            (visible site2 site3)
            (visible site3 site4)
            (visible site4 site5)
            (visible site5 site6)
            (at rover1 site1)
            (at rover2 site1)
            (at rover3 site5)
            (available rover1)
            (available rover2)
            (available rover3)
            (at_lander lander1 site1)
            (at_lander lander2 site6)
            (empty store1)
            (full store2)
            (store_of store1 rover1)
            (store_of store2 rover2)
            (equipped_for_imaging rover3)
            (on_board camera1 rover3)
            (on_board camera2 rover3)
            (on_board camera3 rover3)
            (supports camera1 rgb)
            (supports camera2 rgb)
            (supports camera3 rgb)
            (equipped_for_soil_analysis rover1)
            (equipped_for_soil_analysis rover2)
            (equipped_for_soil_analysis rover3)
            (equipped_for_rock_analysis rover1)
            (equipped_for_rock_analysis rover2)
            (equipped_for_rock_analysis rover3)
            (visible_from objective1 site5)
            (calibration_target camera1 objective1)
            (channel_free lander2)
        )
        (:goal
            (and
                (can_traverse rover1 site1 site2)
                (can_traverse rover1 site2 site3)
                (can_traverse rover1 site3 site4)
                (can_traverse rover1 site4 site5)
                (can_traverse rover1 site5 site6)
                (can_traverse rover2 site1 site2)
                (can_traverse rover2 site2 site3)
                (can_traverse rover2 site3 site4)
                (can_traverse rover2 site4 site5)
                (can_traverse rover2 site5 site6)
                (can_traverse rover3 site1 site2)
                (can_traverse rover3 site2 site3)
                (can_traverse rover3 site3 site4)
                (can_traverse rover3 site4 site5)
                (can_traverse rover3 site5 site6)
                (visible site1 site2)
                (visible site2 site3)
                (visible site3 site4)
                (visible site5 site6)
                (available rover1)
                (available rover2)
                (available rover3)
                (at rover1 site6)
                (at rover2 site6)
                (at rover3 site6)
                (at_lander lander1 site1)
                (at_lander lander2 site6)
                (empty store1)
                (full store2)
                (store_of store1 rover1)
                (store_of store2 rover2)
                (equipped_for_imaging rover3)
                (on_board camera1 rover3)
                (on_board camera2 rover3)
                (on_board camera3 rover3)
                (supports camera1 rgb)
                (supports camera2 rgb)
                (supports camera3 rgb)
                (equipped_for_soil_analysis rover1)
                (equipped_for_soil_analysis rover2)
                (equipped_for_soil_analysis rover3)
                (equipped_for_rock_analysis rover1)
                (equipped_for_rock_analysis rover2)
                (equipped_for_rock_analysis rover3)
                (visible_from objective1 site5)
                (calibration_target camera1 objective1)
                (channel_free lander2)
                (have_image rover3 objective1 rgb)
                (communicated_image_data objective1 rgb)
            )
        )
    )"""


@pytest.fixture
def rover_line_fully_specified_2():
    return """
    (define (problem rover)
        (:domain rover)
        (:objects
            rover1 rover2 rover3 - rover
            site1 site2 site3 site4 site5 site6 - waypoint
            lander1 lander2 - lander
            store1 store2 - store
            grayscale rgb - mode
            camera1 camera2 camera3 - camera
            objective1 objective2 objective3 - objective
        )
        (:init
            (can_traverse rover1 site1 site2)
            (can_traverse rover1 site2 site3)
            (can_traverse rover1 site3 site4)
            (can_traverse rover1 site4 site5)
            (can_traverse rover1 site5 site6)
            (can_traverse rover2 site1 site2)
            (can_traverse rover2 site2 site3)
            (can_traverse rover2 site3 site4)
            (can_traverse rover2 site4 site5)
            (can_traverse rover2 site5 site6)
            (can_traverse rover3 site1 site2)
            (can_traverse rover3 site2 site3)
            (can_traverse rover3 site3 site4)
            (can_traverse rover3 site4 site5)
            (can_traverse rover3 site5 site6)
            (visible site1 site2)
            (visible site2 site3)
            (visible site3 site4)
            (visible site4 site5)
            (visible site5 site6)
            (at rover1 site1)
            (at rover2 site1)
            (at rover3 site5)
            (available rover1)
            (available rover2)
            (available rover3)
            (at_lander lander1 site1)
            (at_lander lander2 site6)
            (empty store1)
            (full store2)
            (store_of store1 rover1)
            (store_of store2 rover2)
            (equipped_for_imaging rover3)
            (on_board camera1 rover3)
            (on_board camera2 rover3)
            (on_board camera3 rover3)
            (supports camera1 grayscale)
            (supports camera2 grayscale)
            (supports camera3 grayscale)
            (equipped_for_soil_analysis rover1)
            (equipped_for_soil_analysis rover2)
            (equipped_for_soil_analysis rover3)
            (equipped_for_rock_analysis rover1)
            (equipped_for_rock_analysis rover2)
            (equipped_for_rock_analysis rover3)
            (visible_from objective1 site5)
            (calibration_target camera1 objective1)
            (channel_free lander2)
        )
        (:goal
            (and
                (can_traverse rover1 site1 site2)
                (can_traverse rover1 site2 site3)
                (can_traverse rover1 site3 site4)
                (can_traverse rover1 site4 site5)
                (can_traverse rover1 site5 site6)
                (can_traverse rover2 site1 site2)
                (can_traverse rover2 site2 site3)
                (can_traverse rover2 site3 site4)
                (can_traverse rover2 site4 site5)
                (can_traverse rover2 site5 site6)
                (can_traverse rover3 site1 site2)
                (can_traverse rover3 site2 site3)
                (can_traverse rover3 site3 site4)
                (can_traverse rover3 site4 site5)
                (can_traverse rover3 site5 site6)
                (visible site1 site2)
                (visible site2 site3)
                (visible site3 site4)
                (visible site5 site6)
                (available rover1)
                (available rover2)
                (available rover3)
                (at rover1 site6)
                (at rover2 site6)
                (at rover3 site6)
                (at_lander lander1 site1)
                (at_lander lander2 site6)
                (empty store1)
                (full store2)
                (store_of store1 rover1)
                (store_of store2 rover2)
                (equipped_for_imaging rover3)
                (on_board camera1 rover3)
                (on_board camera2 rover3)
                (on_board camera3 rover3)
                (supports camera1 grayscale)
                (supports camera2 grayscale)
                (supports camera3 grayscale)
                (equipped_for_soil_analysis rover1)
                (equipped_for_soil_analysis rover2)
                (equipped_for_soil_analysis rover3)
                (equipped_for_rock_analysis rover1)
                (equipped_for_rock_analysis rover2)
                (equipped_for_rock_analysis rover3)
                (visible_from objective1 site5)
                (communicated_image_data objective1 rgb)
                (have_image rover3 objective1 rgb)
                (calibration_target camera1 objective1)
                (channel_free lander2)
            )
        )
    )"""


@pytest.fixture
def rover_line_underspecified_1():
    return """
    (define (problem rover)
        (:domain rover)
        (:objects
            rover1 rover2 rover3 - rover
            site1 site2 site3 site4 site5 site6 - waypoint
            lander1 lander2 - lander
            store1 store2 - store
            rgb - mode
            camera1 camera2 camera3 - camera
            objective1 objective2 objective3 - objective
        )
        (:init
            (can_traverse rover1 site1 site2)
            (can_traverse rover1 site2 site3)
            (can_traverse rover1 site3 site4)
            (can_traverse rover1 site4 site5)
            (can_traverse rover1 site5 site6)
            (can_traverse rover2 site1 site2)
            (can_traverse rover2 site2 site3)
            (can_traverse rover2 site3 site4)
            (can_traverse rover2 site4 site5)
            (can_traverse rover2 site5 site6)
            (can_traverse rover3 site1 site2)
            (can_traverse rover3 site2 site3)
            (can_traverse rover3 site3 site4)
            (can_traverse rover3 site4 site5)
            (can_traverse rover3 site5 site6)
            (visible site1 site2)
            (visible site2 site3)
            (visible site3 site4)
            (visible site4 site5)
            (visible site5 site6)
            (at rover1 site1)
            (at rover2 site1)
            (at rover3 site5)
            (available rover1)
            (available rover2)
            (available rover3)
            (at_lander lander1 site1)
            (at_lander lander2 site6)
            (empty store1)
            (full store2)
            (store_of store1 rover1)
            (store_of store2 rover2)
            (equipped_for_imaging rover3)
            (on_board camera1 rover3)
            (on_board camera2 rover3)
            (on_board camera3 rover3)
            (supports camera1 rgb)
            (supports camera2 rgb)
            (supports camera3 rgb)
            (calibration_target camera1 objective1)
            (equipped_for_soil_analysis rover1)
            (equipped_for_soil_analysis rover2)
            (equipped_for_soil_analysis rover3)
            (equipped_for_rock_analysis rover1)
            (equipped_for_rock_analysis rover2)
            (equipped_for_rock_analysis rover3)
            (visible_from objective1 site5)
        )
        (:goal
            (and
                (at rover1 site6)
                (at rover2 site6)
                (at rover3 site6)
                (communicated_image_data objective1 rgb)
            )
        )
    )"""


@pytest.fixture
def rover_line_missing_visible():
    # if it's missing a visible, then rover can't reach their goal
    return """
    (define (problem rover)
        (:domain rover)
        (:objects
            rover1 rover2 rover3 - rover
            site1 site2 site3 site4 site5 site6 - waypoint
        )
        (:init
            (can_traverse rover1 site1 site2)
            (can_traverse rover1 site2 site3)
            (can_traverse rover1 site3 site4)
            (can_traverse rover1 site4 site5)
            (can_traverse rover1 site5 site6)
            (can_traverse rover2 site1 site2)
            (can_traverse rover2 site2 site3)
            (can_traverse rover2 site3 site4)
            (can_traverse rover2 site4 site5)
            (can_traverse rover2 site5 site6)
            (can_traverse rover3 site1 site2)
            (can_traverse rover3 site2 site3)
            (can_traverse rover3 site3 site4)
            (can_traverse rover3 site4 site5)
            (can_traverse rover3 site5 site6)
            (visible site1 site2)
            (visible site2 site3)
            (visible site3 site4)
            (visible site5 site6)
            (at rover1 site1)
            (at rover2 site1)
            (at rover3 site5)
            (available rover1)
            (available rover2)
            (available rover3)
        )
        (:goal
            (and
                (at rover1 site6)
                (at rover2 site6)
                (at rover3 site6)
            )
        )
    )"""


@pytest.fixture
def rover_line_1():
    return """
    (define (problem rover)
        (:domain rover)
        (:objects
            rover1 rover2 rover3 - rover
            site1 site2 site3 site4 site5 site6 - waypoint
            lander1 lander2 - lander
        )
        (:init
            (can_traverse rover1 site1 site2)
            (can_traverse rover1 site2 site3)
            (can_traverse rover1 site3 site4)
            (can_traverse rover1 site4 site5)
            (can_traverse rover1 site5 site6)
            (can_traverse rover2 site1 site2)
            (can_traverse rover2 site2 site3)
            (can_traverse rover2 site3 site4)
            (can_traverse rover2 site4 site5)
            (can_traverse rover2 site5 site6)
            (can_traverse rover3 site1 site2)
            (can_traverse rover3 site2 site3)
            (can_traverse rover3 site3 site4)
            (can_traverse rover3 site4 site5)
            (can_traverse rover3 site5 site6)
            (visible site1 site2)
            (visible site2 site3)
            (visible site3 site4)
            (visible site4 site5)
            (visible site5 site6)
            (at rover1 site1)
            (at rover2 site1)
            (at rover3 site5)
            (available rover1)
            (available rover2)
            (available rover3)
            (at_lander lander1 site1)
            (at_lander lander2 site6)
        )
        (:goal
            (and
                (at rover1 site6)
                (at rover2 site6)
                (at rover3 site6)
            )
        )
    )"""


@pytest.fixture
def rover_line_2():
    return """
    (define (problem rover)
        (:domain rover)
        (:objects
            rover1 rover2 rover3 - rover
            site1 site2 site3 site4 site5 site6 - waypoint
            lander1 lander2 - lander
        )
        (:init
            (can_traverse rover1 site1 site2)
            (can_traverse rover1 site2 site3)
            (can_traverse rover1 site3 site4)
            (can_traverse rover1 site4 site5)
            (can_traverse rover1 site5 site6)
            (can_traverse rover2 site1 site2)
            (can_traverse rover2 site2 site3)
            (can_traverse rover2 site3 site4)
            (can_traverse rover2 site4 site5)
            (can_traverse rover2 site5 site6)
            (can_traverse rover3 site1 site2)
            (can_traverse rover3 site2 site3)
            (can_traverse rover3 site3 site4)
            (can_traverse rover3 site4 site5)
            (can_traverse rover3 site5 site6)
            (visible site1 site2)
            (visible site2 site3)
            (visible site3 site4)
            (visible site5 site6)
            (at rover1 site1)
            (at rover2 site1)
            (at rover3 site5)
            (available rover1)
            (available rover2)
            (available rover3)
            (at_lander lander1 site1)
            (at_lander lander2 site6)
        )
        (:goal
            (and
                (at rover1 site6)
                (at rover2 site6)
                (at rover3 site6)
            )
        )
    )"""


@pytest.fixture
def rover_single_line_fully_specified():
    return """
    (define (problem rover)
        (:domain rover-single)
        (:objects
            site1 site2 site3 site4 site5 site6 - waypoint
            store1 store2 - store
            rgb - mode
            camera1 camera2 camera3 - camera
            objective1 objective2 objective3 - objective
        )
        (:init
            (can_traverse site1 site2)
            (can_traverse site2 site3)
            (can_traverse site3 site4)
            (can_traverse site4 site5)
            (can_traverse site5 site6)
            (visible site1 site2)
            (visible site2 site3)
            (visible site3 site4)
            (visible site4 site5)
            (visible site5 site6)
            (at_rover site1)
            (available)
            (at_lander site6)
            (empty store1)
            (full store2)
            (supports camera1 rgb)
            (supports camera2 rgb)
            (supports camera3 rgb)
            (visible_from objective1 site5)
            (channel_free)
        )
        (:goal
            (and
                (can_traverse site1 site2)
                (can_traverse site2 site3)
                (can_traverse site3 site4)
                (can_traverse site4 site5)
                (can_traverse site5 site6)
                (visible site1 site2)
                (visible site2 site3)
                (visible site3 site4)
                (visible site4 site5)
                (visible site5 site6)
                (available)
                (at_lander site6)
                (empty store1)
                (full store2)
                (supports camera1 rgb)
                (supports camera2 rgb)
                (supports camera3 rgb)
                (visible_from objective1 site5)
                (channel_free)
                (have_image objective1 rgb)
                (communicated_image_data objective1 rgb)
            )
        )
    )"""


@pytest.fixture
def rover_single_line_fully_specified_1():
    return """
    (define (problem rover)
        (:domain rover-single)
        (:objects
            site1 site2 site3 site4 site5 site6 - waypoint
            store1 store2 - store
            rgb - mode
            camera1 camera2 camera3 - camera
            objective1 objective2 objective3 - objective
        )
        (:init
            (can_traverse site1 site2)
            (can_traverse site2 site3)
            (can_traverse site3 site4)
            (can_traverse site4 site5)
            (can_traverse site5 site6)
            (visible site1 site2)
            (visible site2 site3)
            (visible site3 site4)
            (visible site4 site5)
            (visible site5 site6)
            (at_rover site1)
            (available)
            (at_lander site6)
            (empty store1)
            (full store2)
            (supports camera1 rgb)
            (supports camera2 rgb)
            (supports camera3 rgb)
            (visible_from objective1 site5)
            (channel_free)
        )
        (:goal
            (and
                (can_traverse site1 site2)
                (can_traverse site2 site3)
                (can_traverse site3 site4)
                (can_traverse site4 site5)
                (can_traverse site5 site6)
                (visible site1 site2)
                (visible site2 site3)
                (visible site3 site4)
                (visible site4 site5)
                (visible site5 site6)
                (available)
                (at_lander site6)
                (empty store1)
                (full store2)
                (supports camera1 rgb)
                (supports camera2 rgb)
                (supports camera3 rgb)
                (visible_from objective1 site5)
                (channel_free)
            )
        )
    )"""


@pytest.fixture
def rover_single_line_fully_specified_2():
    return """
    (define (problem rover)
        (:domain rover-single)
        (:objects
            site1 site2 site3 site4 site5 site6 - waypoint
            store1 store2 - store
            rgb - mode
            camera1 camera2 camera3 - camera
            objective1 objective2 objective3 - objective
        )
        (:init
            (can_traverse site1 site2)
            (can_traverse site2 site3)
            (can_traverse site3 site4)
            (can_traverse site4 site5)
            (can_traverse site5 site6)
            (visible site1 site2)
            (visible site2 site3)
            (visible site3 site4)
            (visible site4 site5)
            (visible site5 site6)
            (at_rover site6)
            (available)
            (at_lander site6)
            (empty store1)
            (full store2)
            (supports camera1 rgb)
            (supports camera2 rgb)
            (supports camera3 rgb)
            (visible_from objective1 site5)
            (channel_free)
        )
        (:goal
            (and
                (can_traverse site1 site2)
                (can_traverse site2 site3)
                (can_traverse site3 site4)
                (can_traverse site4 site5)
                (can_traverse site5 site6)
                (visible site1 site2)
                (visible site2 site3)
                (visible site3 site4)
                (visible site4 site5)
                (visible site5 site6)
                (available)
                (at_rover site6)
                (at_lander site6)
                (empty store1)
                (full store2)
                (supports camera1 rgb)
                (supports camera2 rgb)
                (supports camera3 rgb)
                (visible_from objective1 site5)
                (channel_free)
            )
        )
    )"""


@pytest.fixture
def rover_single_line_fully_specified_3():
    return """
    (define (problem rover)
        (:domain rover-single)
        (:objects
            site1 site2 site3 site4 site5 site6 - waypoint
            store1 store2 - store
            rgb - mode
            camera1 camera2 camera3 - camera
            objective1 objective2 objective3 - objective
        )
        (:init
            (can_traverse site1 site2)
            (can_traverse site2 site3)
            (can_traverse site3 site4)
            (can_traverse site4 site5)
            (can_traverse site5 site6)
            (visible site1 site2)
            (visible site2 site3)
            (visible site3 site4)
            (visible site4 site5)
            (visible site5 site6)
            (at_rover site1)
            (available)
            (at_lander site6)
            (empty store1)
            (full store2)
            (supports camera1 rgb)
            (supports camera2 rgb)
            (supports camera3 rgb)
            (visible_from objective1 site5)
            (channel_free)
        )
        (:goal
            (and
                (can_traverse site1 site2)
                (can_traverse site2 site3)
                (can_traverse site3 site4)
                (can_traverse site4 site5)
                (can_traverse site5 site6)
                (visible site1 site2)
                (visible site2 site3)
                (visible site3 site4)
                (visible site4 site5)
                (visible site5 site6)
                (available)
                (at_rover site6)
                (at_lander site6)
                (empty store1)
                (full store2)
                (supports camera1 rgb)
                (supports camera2 rgb)
                (supports camera3 rgb)
                (visible_from objective1 site5)
                (channel_free)
                (have_image objective1 rgb)
                (communicated_image_data objective1 rgb)
            )
        )
    )"""


@pytest.fixture
def rover_single_line_fully_specified_4():
    # rock and soil samples after lander
    return """
    (define (problem rover)
        (:domain rover-single)
        (:objects
            site1 site2 site3 site4 site5 site6 - waypoint
            store1 store2 - store
            rgb - mode
            camera1 camera2 camera3 - camera
            objective1 objective2 objective3 - objective
        )
        (:init
            (can_traverse site1 site2)
            (can_traverse site2 site3)
            (can_traverse site3 site4)
            (can_traverse site4 site5)
            (can_traverse site5 site6)
            (visible site1 site2)
            (visible site2 site3)
            (visible site3 site4)
            (visible site4 site5)
            (visible site5 site6)
            (at_rover site1)
            (available)
            (at_lander site6)
            (empty store1)
            (full store2)
            (supports camera1 rgb)
            (supports camera2 rgb)
            (supports camera3 rgb)
            (visible_from objective1 site5)
            (channel_free)
            (at_rock_sample site6)
            (at_soil_sample site6)
        )
        (:goal
            (and
                (can_traverse site1 site2)
                (can_traverse site2 site3)
                (can_traverse site3 site4)
                (can_traverse site4 site5)
                (can_traverse site5 site6)
                (visible site1 site2)
                (visible site2 site3)
                (visible site3 site4)
                (visible site4 site5)
                (visible site5 site6)
                (available)
                (at_rover site6)
                (at_lander site6)
                (empty store1)
                (full store2)
                (supports camera1 rgb)
                (supports camera2 rgb)
                (supports camera3 rgb)
                (visible_from objective1 site5)
                (channel_free)
                (have_image objective1 rgb)
                (communicated_image_data objective1 rgb)
                (have_rock_analysis site6)
                (communicated_rock_data site6)
                (have_soil_analysis site6)
                (communicated_soil_data site6)
                (at_rock_sample site6)
                (at_soil_sample site6)
            )
        )
    )"""


@pytest.fixture
def rover_single_line_fully_specified_4a():
    # rock and soil samples after lander
    return """
    (define (problem rover)
        (:domain rover-single)
        (:objects
            site1 site2 site3 site4 site5 site6 - waypoint
            store1 store2 - store
            rgb - mode
            camera1 camera2 camera3 - camera
            objective1 objective2 objective3 - objective
        )
        (:init
            (can_traverse site1 site2)
            (can_traverse site2 site3)
            (can_traverse site3 site4)
            (can_traverse site4 site5)
            (can_traverse site5 site6)
            (visible site1 site2)
            (visible site2 site3)
            (visible site3 site4)
            (visible site4 site5)
            (visible site5 site6)
            (at_rover site1)
            (available)
            (at_lander site6)
            (empty store1)
            (full store2)
            (supports camera1 rgb)
            (supports camera2 rgb)
            (supports camera3 rgb)
            (visible_from objective1 site5)
            (channel_free)
            (at_rock_sample site6)
            (at_soil_sample site6)
        )
        (:goal
            (and
                (have_image objective1 rgb)
                (communicated_image_data objective1 rgb)
                (have_rock_analysis site6)
                (communicated_rock_data site6)
                (have_soil_analysis site6)
                (communicated_soil_data site6)
            )
        )
    )"""


@pytest.fixture
def rover_single_line_equiv():
    return """
    (define (problem rover)
        (:domain rover-single)
        (:objects
            site1 site2 site3 site4 site5 site6 - waypoint
            store1 store2 - store
            rgb - mode
            camera1 camera2 camera3 - camera
            objective1 objective2 objective3 - objective
        )
        (:init
            (can_traverse site1 site2)
            (can_traverse site2 site3)
            (can_traverse site3 site4)
            (can_traverse site4 site5)
            (can_traverse site5 site6)
            (visible site1 site2)
            (visible site2 site3)
            (visible site3 site4)
            (visible site4 site5)
            (visible site5 site6)
            (at_rover site1)
            (available)
            (at_lander site6)
            (empty store1)
            (full store2)
            (supports camera1 rgb)
            (supports camera2 rgb)
            (supports camera3 rgb)
            (visible_from objective1 site5)
            (channel_free)
        )
        (:goal
            (and
                (at_lander site6)
                (communicated_image_data objective1 rgb)
            )
        )
    )"""


@pytest.fixture
def rover_single_line_equiva():
    return """
    (define (problem rover)
        (:domain rover-single)
        (:objects
            site1 site2 site3 site4 site5 site6 - waypoint
            store1 store2 - store
            rgb - mode
            camera1 camera2 camera3 - camera
            objective1 objective2 objective3 - objective
        )
        (:init
            (can_traverse site1 site2)
            (can_traverse site2 site3)
            (can_traverse site3 site4)
            (can_traverse site4 site5)
            (can_traverse site5 site6)
            (visible site1 site2)
            (visible site2 site3)
            (visible site3 site4)
            (visible site4 site5)
            (visible site5 site6)
            (at_rover site1)
            (available)
            (at_lander site6)
            (empty store1)
            (full store2)
            (supports camera1 rgb)
            (supports camera2 rgb)
            (supports camera3 rgb)
            (visible_from objective1 site5)
            (channel_free)
        )
        (:goal
            (and
                (can_traverse site1 site2)
                (can_traverse site2 site3)
                (can_traverse site3 site4)
                (can_traverse site4 site5)
                (can_traverse site5 site6)
                (visible site1 site2)
                (visible site2 site3)
                (visible site3 site4)
                (visible site4 site5)
                (visible site5 site6)
                (available)
                (supports camera1 rgb)
                (supports camera2 rgb)
                (supports camera3 rgb)
                (visible_from objective1 site5)
                (channel_free)
                (have_image objective1 rgb)
                (communicated_image_data objective1 rgb)
            )
        )
    )"""


@pytest.fixture
def rover_single_line_equiv_1():
    return """
    (define (problem rover)
        (:domain rover-single)
        (:objects
            site1 site2 site3 site4 site5 site6 - waypoint
            store1 store2 - store
            rgb - mode
            camera1 camera2 camera3 - camera
            objective1 objective2 objective3 - objective
        )
        (:init
            (can_traverse site1 site2)
            (can_traverse site2 site3)
            (can_traverse site3 site4)
            (can_traverse site4 site5)
            (can_traverse site5 site6)
            (visible site1 site2)
            (visible site2 site3)
            (visible site3 site4)
            (visible site4 site5)
            (visible site5 site6)
            (visible site6 site6)
            (at_rover site1)
            (available)
            (at_lander site6)
            (at_rock_sample site6)
            (empty store1)
            (full store2)
            (supports camera1 rgb)
            (supports camera2 rgb)
            (supports camera3 rgb)
            (visible_from objective1 site5)
            (channel_free)
        )
        (:goal
            (and
                (can_traverse site1 site2)
                (can_traverse site2 site3)
                (can_traverse site3 site4)
                (can_traverse site4 site5)
                (can_traverse site5 site6)
                (visible site1 site2)
                (visible site2 site3)
                (visible site3 site4)
                (visible site4 site5)
                (visible site5 site6)
                (visible site6 site6)
                (available)
                (at_lander site6)
                (supports camera1 rgb)
                (supports camera2 rgb)
                (supports camera3 rgb)
                (visible_from objective1 site5)
                (channel_free)
                (at_rover site6)
                (have_rock_analysis site6)
                (communicated_rock_data site6)
                (have_image objective1 rgb)
                (communicated_image_data objective1 rgb)
            )
        )
    )"""


@pytest.fixture
def rover_single_line_equiv_1a():
    return """
    (define (problem rover)
        (:domain rover-single)
        (:objects
            site1 site2 site3 site4 site5 site6 - waypoint
            store1 store2 - store
            rgb - mode
            camera1 camera2 camera3 - camera
            objective1 objective2 objective3 - objective
        )
        (:init
            (can_traverse site1 site2)
            (can_traverse site2 site3)
            (can_traverse site3 site4)
            (can_traverse site4 site5)
            (can_traverse site5 site6)
            (visible site1 site2)
            (visible site2 site3)
            (visible site3 site4)
            (visible site4 site5)
            (visible site5 site6)
            (visible site6 site6)
            (at_rover site1)
            (available)
            (at_lander site6)
            (at_rock_sample site6)
            (empty store1)
            (full store2)
            (supports camera1 rgb)
            (supports camera2 rgb)
            (supports camera3 rgb)
            (visible_from objective1 site5)
            (channel_free)
        )
        (:goal
            (and
                (at_rover site6)
                (communicated_rock_data site6)
                (communicated_image_data objective1 rgb)
            )
        )
    )"""


@pytest.fixture
def rover_single_line_equiv_1b():
    return """
    (define (problem rover)
        (:domain rover-single)
        (:objects
            site1 site2 site3 site4 site5 site6 - waypoint
            store1 store2 - store
            rgb - mode
            camera1 camera2 camera3 - camera
            objective1 objective2 objective3 - objective
        )
        (:init
            (can_traverse site1 site2)
            (can_traverse site2 site3)
            (can_traverse site3 site4)
            (can_traverse site4 site5)
            (can_traverse site5 site6)
            (visible site1 site2)
            (visible site2 site3)
            (visible site3 site4)
            (visible site4 site5)
            (visible site5 site6)
            (visible site6 site6)
            (at_rover site1)
            (available)
            (at_lander site6)
            (at_rock_sample site6)
            (empty store1)
            (full store2)
            (supports camera1 rgb)
            (supports camera2 rgb)
            (supports camera3 rgb)
            (visible_from objective1 site5)
            (channel_free)
        )
        (:goal
            (and
                (communicated_rock_data site6)
                (communicated_image_data objective1 rgb)
            )
        )
    )"""
