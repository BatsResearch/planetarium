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
def rover_single_line_fully_specified():
    return """
    (define (problem rover)
        (:domain rover-single)
        (:objects
            site1 site2 site3 site4 site5 site6 - waypoint
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
                (at_rock_sample site6)
                (at_soil_sample site6)
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


@pytest.fixture
def floortile_fully_specified():
    return """
 (define (problem two_colors_1)
 (:domain floor-tile)
 (:objects tile_0-1 tile_0-2 tile_0-3 tile_0-4 tile_0-5 tile_0-6
           tile_1-1 tile_1-2 tile_1-3 tile_1-4 tile_1-5 tile_1-6
           tile_2-1 tile_2-2 tile_2-3 tile_2-4 tile_2-5 tile_2-6
           tile_3-1 tile_3-2 tile_3-3 tile_3-4 tile_3-5 tile_3-6
           tile_4-1 tile_4-2 tile_4-3 tile_4-4 tile_4-5 tile_4-6 - tile
           robot1 robot2 robot3 - robot
           white black - color
)
 (:init
   (robot-at robot1 tile_0-5)
   (robot-has robot1 white)
   (robot-at robot2 tile_2-1)
   (robot-has robot2 black)
   (robot-at robot3 tile_1-4)
   (robot-has robot3 white)
   (available-color white)
   (available-color black)
   (up tile_1-1 tile_0-1)
   (up tile_1-2 tile_0-2)
   (up tile_1-3 tile_0-3)
   (up tile_1-4 tile_0-4)
   (up tile_1-5 tile_0-5)
   (up tile_1-6 tile_0-6)
   (up tile_2-1 tile_1-1)
   (up tile_2-2 tile_1-2)
   (up tile_2-3 tile_1-3)
   (up tile_2-4 tile_1-4)
   (up tile_2-5 tile_1-5)
   (up tile_2-6 tile_1-6)
   (up tile_3-1 tile_2-1)
   (up tile_3-2 tile_2-2)
   (up tile_3-3 tile_2-3)
   (up tile_3-4 tile_2-4)
   (up tile_3-5 tile_2-5)
   (up tile_3-6 tile_2-6)
   (up tile_4-1 tile_3-1)
   (up tile_4-2 tile_3-2)
   (up tile_4-3 tile_3-3)
   (up tile_4-4 tile_3-4)
   (up tile_4-5 tile_3-5)
   (up tile_4-6 tile_3-6)
   (right tile_0-2 tile_0-1)
   (right tile_0-3 tile_0-2)
   (right tile_0-4 tile_0-3)
   (right tile_0-5 tile_0-4)
   (right tile_0-6 tile_0-5)
   (right tile_1-2 tile_1-1)
   (right tile_1-3 tile_1-2)
   (right tile_1-4 tile_1-3)
   (right tile_1-5 tile_1-4)
   (right tile_1-6 tile_1-5)
   (right tile_2-2 tile_2-1)
   (right tile_2-3 tile_2-2)
   (right tile_2-4 tile_2-3)
   (right tile_2-5 tile_2-4)
   (right tile_2-6 tile_2-5)
   (right tile_3-2 tile_3-1)
   (right tile_3-3 tile_3-2)
   (right tile_3-4 tile_3-3)
   (right tile_3-5 tile_3-4)
   (right tile_3-6 tile_3-5)
   (right tile_4-2 tile_4-1)
   (right tile_4-3 tile_4-2)
   (right tile_4-4 tile_4-3)
   (right tile_4-5 tile_4-4)
   (right tile_4-6 tile_4-5)
)
 (:goal (and
    (painted tile_1-1 white)
    (painted tile_1-2 black)
    (painted tile_1-3 white)
    (painted tile_1-4 black)
    (painted tile_1-5 white)
    (painted tile_1-6 black)
    (painted tile_2-1 black)
    (painted tile_2-2 white)
    (painted tile_2-3 black)
    (painted tile_2-4 white)
    (painted tile_2-5 black)
    (painted tile_2-6 white)
    (painted tile_3-1 white)
    (painted tile_3-2 black)
    (painted tile_3-3 white)
    (painted tile_3-4 black)
    (painted tile_3-5 white)
    (painted tile_3-6 black)
    (painted tile_4-1 black)
    (painted tile_4-2 white)
    (painted tile_4-3 black)
    (painted tile_4-4 white)
    (painted tile_4-5 black)
    (painted tile_4-6 white)
    (up tile_1-1 tile_0-1)
   (up tile_1-2 tile_0-2)
   (up tile_1-3 tile_0-3)
   (up tile_1-4 tile_0-4)
   (up tile_1-5 tile_0-5)
   (up tile_1-6 tile_0-6)
   (up tile_2-1 tile_1-1)
   (up tile_2-2 tile_1-2)
   (up tile_2-3 tile_1-3)
   (up tile_2-4 tile_1-4)
   (up tile_2-5 tile_1-5)
   (up tile_2-6 tile_1-6)
   (up tile_3-1 tile_2-1)
   (up tile_3-2 tile_2-2)
   (up tile_3-3 tile_2-3)
   (up tile_3-4 tile_2-4)
   (up tile_3-5 tile_2-5)
   (up tile_3-6 tile_2-6)
   (up tile_4-1 tile_3-1)
   (up tile_4-2 tile_3-2)
   (up tile_4-3 tile_3-3)
   (up tile_4-4 tile_3-4)
   (up tile_4-5 tile_3-5)
   (up tile_4-6 tile_3-6)
   (right tile_0-2 tile_0-1)
   (right tile_0-3 tile_0-2)
   (right tile_0-4 tile_0-3)
   (right tile_0-5 tile_0-4)
   (right tile_0-6 tile_0-5)
   (right tile_1-2 tile_1-1)
   (right tile_1-3 tile_1-2)
   (right tile_1-4 tile_1-3)
   (right tile_1-5 tile_1-4)
   (right tile_1-6 tile_1-5)
   (right tile_2-2 tile_2-1)
   (right tile_2-3 tile_2-2)
   (right tile_2-4 tile_2-3)
   (right tile_2-5 tile_2-4)
   (right tile_2-6 tile_2-5)
   (right tile_3-2 tile_3-1)
   (right tile_3-3 tile_3-2)
   (right tile_3-4 tile_3-3)
   (right tile_3-5 tile_3-4)
   (right tile_3-6 tile_3-5)
   (right tile_4-2 tile_4-1)
   (right tile_4-3 tile_4-2)
   (right tile_4-4 tile_4-3)
   (right tile_4-5 tile_4-4)
   (right tile_4-6 tile_4-5)
   (available-color white)
   (available-color black)
))
)"""


@pytest.fixture
def floortile_underspecified_directions():
    return """
 (define (problem two_colors_1)
 (:domain floor-tile)
 (:objects tile_0-1 tile_0-2 tile_0-3 tile_0-4 tile_0-5 tile_0-6
           tile_1-1 tile_1-2 tile_1-3 tile_1-4 tile_1-5 tile_1-6
           tile_2-1 tile_2-2 tile_2-3 tile_2-4 tile_2-5 tile_2-6
           tile_3-1 tile_3-2 tile_3-3 tile_3-4 tile_3-5 tile_3-6
           tile_4-1 tile_4-2 tile_4-3 tile_4-4 tile_4-5 tile_4-6 - tile
           robot1 robot2 robot3 - robot
           white black - color
)
 (:init
   (robot-at robot1 tile_0-5)
   (robot-has robot1 white)
   (robot-at robot2 tile_2-1)
   (robot-has robot2 black)
   (robot-at robot3 tile_1-4)
   (robot-has robot3 white)
   (available-color white)
   (available-color black)
   (up tile_1-1 tile_0-1)
   (up tile_1-2 tile_0-2)
   (up tile_1-3 tile_0-3)
   (up tile_1-4 tile_0-4)
   (up tile_1-5 tile_0-5)
   (up tile_1-6 tile_0-6)
   (up tile_2-1 tile_1-1)
   (up tile_2-2 tile_1-2)
   (up tile_2-3 tile_1-3)
   (up tile_2-4 tile_1-4)
   (up tile_2-5 tile_1-5)
   (up tile_2-6 tile_1-6)
   (up tile_3-1 tile_2-1)
   (up tile_3-2 tile_2-2)
   (up tile_3-3 tile_2-3)
   (up tile_3-4 tile_2-4)
   (up tile_3-5 tile_2-5)
   (up tile_3-6 tile_2-6)
   (up tile_4-1 tile_3-1)
   (up tile_4-2 tile_3-2)
   (up tile_4-3 tile_3-3)
   (up tile_4-4 tile_3-4)
   (up tile_4-5 tile_3-5)
   (up tile_4-6 tile_3-6)
   (right tile_0-2 tile_0-1)
   (right tile_0-3 tile_0-2)
   (right tile_0-4 tile_0-3)
   (right tile_0-5 tile_0-4)
   (right tile_0-6 tile_0-5)
   (right tile_1-2 tile_1-1)
   (right tile_1-3 tile_1-2)
   (right tile_1-4 tile_1-3)
   (right tile_1-5 tile_1-4)
   (right tile_1-6 tile_1-5)
   (right tile_2-2 tile_2-1)
   (right tile_2-3 tile_2-2)
   (right tile_2-4 tile_2-3)
   (right tile_2-5 tile_2-4)
   (right tile_2-6 tile_2-5)
   (right tile_3-2 tile_3-1)
   (right tile_3-3 tile_3-2)
   (right tile_3-4 tile_3-3)
   (right tile_3-5 tile_3-4)
   (right tile_3-6 tile_3-5)
   (right tile_4-2 tile_4-1)
   (right tile_4-3 tile_4-2)
   (right tile_4-4 tile_4-3)
   (right tile_4-5 tile_4-4)
   (right tile_4-6 tile_4-5)
)
 (:goal (and
    (painted tile_1-1 white)
    (painted tile_1-2 black)
    (painted tile_1-3 white)
    (painted tile_1-4 black)
    (painted tile_1-5 white)
    (painted tile_1-6 black)
    (painted tile_2-1 black)
    (painted tile_2-2 white)
    (painted tile_2-3 black)
    (painted tile_2-4 white)
    (painted tile_2-5 black)
    (painted tile_2-6 white)
    (painted tile_3-1 white)
    (painted tile_3-2 black)
    (painted tile_3-3 white)
    (painted tile_3-4 black)
    (painted tile_3-5 white)
    (painted tile_3-6 black)
    (painted tile_4-1 black)
    (painted tile_4-2 white)
    (painted tile_4-3 black)
    (painted tile_4-4 white)
    (painted tile_4-5 black)
    (painted tile_4-6 white)
))
)"""


@pytest.fixture
def floortile_no_white1():
    return """
 (define (problem two_colors_1)
 (:domain floor-tile)
 (:objects tile_0-1 tile_0-2 tile_0-3 tile_0-4 tile_0-5 tile_0-6
           tile_1-1 tile_1-2 tile_1-3 tile_1-4 tile_1-5 tile_1-6
           tile_2-1 tile_2-2 tile_2-3 tile_2-4 tile_2-5 tile_2-6
           tile_3-1 tile_3-2 tile_3-3 tile_3-4 tile_3-5 tile_3-6
           tile_4-1 tile_4-2 tile_4-3 tile_4-4 tile_4-5 tile_4-6 - tile
           robot1 - robot
           white black - color
)
 (:init
   (robot-at robot1 tile_0-5)
   (robot-has robot1 white)
   (available-color black)
   (up tile_1-1 tile_0-1)
   (up tile_1-2 tile_0-2)
   (up tile_1-3 tile_0-3)
   (up tile_1-4 tile_0-4)
   (up tile_1-5 tile_0-5)
   (up tile_1-6 tile_0-6)
   (up tile_2-1 tile_1-1)
   (up tile_2-2 tile_1-2)
   (up tile_2-3 tile_1-3)
   (up tile_2-4 tile_1-4)
   (up tile_2-5 tile_1-5)
   (up tile_2-6 tile_1-6)
   (up tile_3-1 tile_2-1)
   (up tile_3-2 tile_2-2)
   (up tile_3-3 tile_2-3)
   (up tile_3-4 tile_2-4)
   (up tile_3-5 tile_2-5)
   (up tile_3-6 tile_2-6)
   (up tile_4-1 tile_3-1)
   (up tile_4-2 tile_3-2)
   (up tile_4-3 tile_3-3)
   (up tile_4-4 tile_3-4)
   (up tile_4-5 tile_3-5)
   (up tile_4-6 tile_3-6)
   (right tile_0-2 tile_0-1)
   (right tile_0-3 tile_0-2)
   (right tile_0-4 tile_0-3)
   (right tile_0-5 tile_0-4)
   (right tile_0-6 tile_0-5)
   (right tile_1-2 tile_1-1)
   (right tile_1-3 tile_1-2)
   (right tile_1-4 tile_1-3)
   (right tile_1-5 tile_1-4)
   (right tile_1-6 tile_1-5)
   (right tile_2-2 tile_2-1)
   (right tile_2-3 tile_2-2)
   (right tile_2-4 tile_2-3)
   (right tile_2-5 tile_2-4)
   (right tile_2-6 tile_2-5)
   (right tile_3-2 tile_3-1)
   (right tile_3-3 tile_3-2)
   (right tile_3-4 tile_3-3)
   (right tile_3-5 tile_3-4)
   (right tile_3-6 tile_3-5)
   (right tile_4-2 tile_4-1)
   (right tile_4-3 tile_4-2)
   (right tile_4-4 tile_4-3)
   (right tile_4-5 tile_4-4)
   (right tile_4-6 tile_4-5)
)
 (:goal (and
    (painted tile_1-1 white)
    (painted tile_1-2 black)
    (painted tile_1-3 white)
    (painted tile_1-4 black)
    (painted tile_1-5 white)
    (painted tile_1-6 black)
    (painted tile_2-1 black)
    (painted tile_2-2 white)
    (painted tile_2-3 black)
    (painted tile_2-4 white)
    (painted tile_2-5 black)
    (painted tile_2-6 white)
    (painted tile_3-1 white)
    (painted tile_3-2 black)
    (painted tile_3-3 white)
    (painted tile_3-4 black)
    (painted tile_3-5 white)
    (painted tile_3-6 black)
    (painted tile_4-1 black)
    (painted tile_4-2 white)
    (painted tile_4-3 black)
    (painted tile_4-4 white)
    (painted tile_4-5 black)
    (painted tile_4-6 white)
    (robot-has robot1 black)
    (available-color black)
    (up tile_1-1 tile_0-1)
    (up tile_1-2 tile_0-2)
    (up tile_1-3 tile_0-3)
    (up tile_1-4 tile_0-4)
    (up tile_1-5 tile_0-5)
    (up tile_1-6 tile_0-6)
    (up tile_2-1 tile_1-1)
    (up tile_2-2 tile_1-2)
    (up tile_2-3 tile_1-3)
    (up tile_2-4 tile_1-4)
    (up tile_2-5 tile_1-5)
    (up tile_2-6 tile_1-6)
    (up tile_3-1 tile_2-1)
    (up tile_3-2 tile_2-2)
    (up tile_3-3 tile_2-3)
    (up tile_3-4 tile_2-4)
    (up tile_3-5 tile_2-5)
    (up tile_3-6 tile_2-6)
    (up tile_4-1 tile_3-1)
    (up tile_4-2 tile_3-2)
    (up tile_4-3 tile_3-3)
    (up tile_4-4 tile_3-4)
    (up tile_4-5 tile_3-5)
    (up tile_4-6 tile_3-6)
    (right tile_0-2 tile_0-1)
    (right tile_0-3 tile_0-2)
    (right tile_0-4 tile_0-3)
    (right tile_0-5 tile_0-4)
    (right tile_0-6 tile_0-5)
    (right tile_1-2 tile_1-1)
    (right tile_1-3 tile_1-2)
    (right tile_1-4 tile_1-3)
    (right tile_1-5 tile_1-4)
    (right tile_1-6 tile_1-5)
    (right tile_2-2 tile_2-1)
    (right tile_2-3 tile_2-2)
    (right tile_2-4 tile_2-3)
    (right tile_2-5 tile_2-4)
    (right tile_2-6 tile_2-5)
    (right tile_3-2 tile_3-1)
    (right tile_3-3 tile_3-2)
    (right tile_3-4 tile_3-3)
    (right tile_3-5 tile_3-4)
    (right tile_3-6 tile_3-5)
    (right tile_4-2 tile_4-1)
    (right tile_4-3 tile_4-2)
    (right tile_4-4 tile_4-3)
    (right tile_4-5 tile_4-4)
    (right tile_4-6 tile_4-5)
))
)"""


@pytest.fixture
def floortile_no_white1a():
    return """
 (define (problem two_colors_1)
 (:domain floor-tile)
 (:objects tile_0-1 tile_0-2 tile_0-3 tile_0-4 tile_0-5 tile_0-6
           tile_1-1 tile_1-2 tile_1-3 tile_1-4 tile_1-5 tile_1-6
           tile_2-1 tile_2-2 tile_2-3 tile_2-4 tile_2-5 tile_2-6
           tile_3-1 tile_3-2 tile_3-3 tile_3-4 tile_3-5 tile_3-6
           tile_4-1 tile_4-2 tile_4-3 tile_4-4 tile_4-5 tile_4-6 - tile
           robot1 - robot
           white black - color
)
 (:init
   (robot-at robot1 tile_0-5)
   (robot-has robot1 white)
   (available-color black)
   (up tile_1-1 tile_0-1)
   (up tile_1-2 tile_0-2)
   (up tile_1-3 tile_0-3)
   (up tile_1-4 tile_0-4)
   (up tile_1-5 tile_0-5)
   (up tile_1-6 tile_0-6)
   (up tile_2-1 tile_1-1)
   (up tile_2-2 tile_1-2)
   (up tile_2-3 tile_1-3)
   (up tile_2-4 tile_1-4)
   (up tile_2-5 tile_1-5)
   (up tile_2-6 tile_1-6)
   (up tile_3-1 tile_2-1)
   (up tile_3-2 tile_2-2)
   (up tile_3-3 tile_2-3)
   (up tile_3-4 tile_2-4)
   (up tile_3-5 tile_2-5)
   (up tile_3-6 tile_2-6)
   (up tile_4-1 tile_3-1)
   (up tile_4-2 tile_3-2)
   (up tile_4-3 tile_3-3)
   (up tile_4-4 tile_3-4)
   (up tile_4-5 tile_3-5)
   (up tile_4-6 tile_3-6)
   (right tile_0-2 tile_0-1)
   (right tile_0-3 tile_0-2)
   (right tile_0-4 tile_0-3)
   (right tile_0-5 tile_0-4)
   (right tile_0-6 tile_0-5)
   (right tile_1-2 tile_1-1)
   (right tile_1-3 tile_1-2)
   (right tile_1-4 tile_1-3)
   (right tile_1-5 tile_1-4)
   (right tile_1-6 tile_1-5)
   (right tile_2-2 tile_2-1)
   (right tile_2-3 tile_2-2)
   (right tile_2-4 tile_2-3)
   (right tile_2-5 tile_2-4)
   (right tile_2-6 tile_2-5)
   (right tile_3-2 tile_3-1)
   (right tile_3-3 tile_3-2)
   (right tile_3-4 tile_3-3)
   (right tile_3-5 tile_3-4)
   (right tile_3-6 tile_3-5)
   (right tile_4-2 tile_4-1)
   (right tile_4-3 tile_4-2)
   (right tile_4-4 tile_4-3)
   (right tile_4-5 tile_4-4)
   (right tile_4-6 tile_4-5)
)
 (:goal (and
    (painted tile_1-1 white)
    (painted tile_1-2 black)
    (painted tile_1-3 white)
    (painted tile_1-4 black)
    (painted tile_1-5 white)
    (painted tile_1-6 black)
    (painted tile_2-1 black)
    (painted tile_2-2 white)
    (painted tile_2-3 black)
    (painted tile_2-4 white)
    (painted tile_2-5 black)
    (painted tile_2-6 white)
    (painted tile_3-1 white)
    (painted tile_3-2 black)
    (painted tile_3-3 white)
    (painted tile_3-4 black)
    (painted tile_3-5 white)
    (painted tile_3-6 black)
    (painted tile_4-1 black)
    (painted tile_4-2 white)
    (painted tile_4-3 black)
    (painted tile_4-4 white)
    (painted tile_4-5 black)
    (painted tile_4-6 white)
))
)"""


@pytest.fixture
def floortile_no_white2():
    return """
 (define (problem two_colors_2)
 (:domain floor-tile)
 (:objects tile_0-1 tile_0-2 tile_0-3 tile_0-4 tile_0-5 tile_0-6
           tile_1-1 tile_1-2 tile_1-3 tile_1-4 tile_1-5 tile_1-6
           tile_2-1 tile_2-2 tile_2-3 tile_2-4 tile_2-5 tile_2-6
           tile_3-1 tile_3-2 tile_3-3 tile_3-4 tile_3-5 tile_3-6
           tile_4-1 tile_4-2 tile_4-3 tile_4-4 tile_4-5 tile_4-6 - tile
           robot1 robot2 - robot
           white black - color
)
 (:init
   (robot-at robot1 tile_0-5)
   (robot-has robot1 white)
   (robot-has robot2 black)
   (robot-at robot2 tile_0-1)
   (available-color black)
   (up tile_1-1 tile_0-1)
   (up tile_1-2 tile_0-2)
   (up tile_1-3 tile_0-3)
   (up tile_1-4 tile_0-4)
   (up tile_1-5 tile_0-5)
   (up tile_1-6 tile_0-6)
   (up tile_2-1 tile_1-1)
   (up tile_2-2 tile_1-2)
   (up tile_2-3 tile_1-3)
   (up tile_2-4 tile_1-4)
   (up tile_2-5 tile_1-5)
   (up tile_2-6 tile_1-6)
   (up tile_3-1 tile_2-1)
   (up tile_3-2 tile_2-2)
   (up tile_3-3 tile_2-3)
   (up tile_3-4 tile_2-4)
   (up tile_3-5 tile_2-5)
   (up tile_3-6 tile_2-6)
   (up tile_4-1 tile_3-1)
   (up tile_4-2 tile_3-2)
   (up tile_4-3 tile_3-3)
   (up tile_4-4 tile_3-4)
   (up tile_4-5 tile_3-5)
   (up tile_4-6 tile_3-6)
   (right tile_0-2 tile_0-1)
   (right tile_0-3 tile_0-2)
   (right tile_0-4 tile_0-3)
   (right tile_0-5 tile_0-4)
   (right tile_0-6 tile_0-5)
   (right tile_1-2 tile_1-1)
   (right tile_1-3 tile_1-2)
   (right tile_1-4 tile_1-3)
   (right tile_1-5 tile_1-4)
   (right tile_1-6 tile_1-5)
   (right tile_2-2 tile_2-1)
   (right tile_2-3 tile_2-2)
   (right tile_2-4 tile_2-3)
   (right tile_2-5 tile_2-4)
   (right tile_2-6 tile_2-5)
   (right tile_3-2 tile_3-1)
   (right tile_3-3 tile_3-2)
   (right tile_3-4 tile_3-3)
   (right tile_3-5 tile_3-4)
   (right tile_3-6 tile_3-5)
   (right tile_4-2 tile_4-1)
   (right tile_4-3 tile_4-2)
   (right tile_4-4 tile_4-3)
   (right tile_4-5 tile_4-4)
   (right tile_4-6 tile_4-5)
)
 (:goal (and
    (painted tile_1-1 white)
    (painted tile_1-2 black)
    (painted tile_1-3 white)
    (painted tile_1-4 black)
    (painted tile_1-5 white)
    (painted tile_1-6 black)
    (painted tile_2-1 black)
    (painted tile_2-2 white)
    (painted tile_2-3 black)
    (painted tile_2-4 white)
    (painted tile_2-5 black)
    (painted tile_2-6 white)
    (painted tile_3-1 white)
    (painted tile_3-2 black)
    (painted tile_3-3 white)
    (painted tile_3-4 black)
    (painted tile_3-5 white)
    (painted tile_3-6 black)
    (painted tile_4-1 black)
    (painted tile_4-2 white)
    (painted tile_4-3 black)
    (painted tile_4-4 white)
    (painted tile_4-5 black)
    (painted tile_4-6 white)
    (robot-has robot2 black)
    (available-color black)
   (up tile_1-1 tile_0-1)
   (up tile_1-2 tile_0-2)
   (up tile_1-3 tile_0-3)
   (up tile_1-4 tile_0-4)
   (up tile_1-5 tile_0-5)
   (up tile_1-6 tile_0-6)
   (up tile_2-1 tile_1-1)
   (up tile_2-2 tile_1-2)
   (up tile_2-3 tile_1-3)
   (up tile_2-4 tile_1-4)
   (up tile_2-5 tile_1-5)
   (up tile_2-6 tile_1-6)
   (up tile_3-1 tile_2-1)
   (up tile_3-2 tile_2-2)
   (up tile_3-3 tile_2-3)
   (up tile_3-4 tile_2-4)
   (up tile_3-5 tile_2-5)
   (up tile_3-6 tile_2-6)
   (up tile_4-1 tile_3-1)
   (up tile_4-2 tile_3-2)
   (up tile_4-3 tile_3-3)
   (up tile_4-4 tile_3-4)
   (up tile_4-5 tile_3-5)
   (up tile_4-6 tile_3-6)
   (right tile_0-2 tile_0-1)
   (right tile_0-3 tile_0-2)
   (right tile_0-4 tile_0-3)
   (right tile_0-5 tile_0-4)
   (right tile_0-6 tile_0-5)
   (right tile_1-2 tile_1-1)
   (right tile_1-3 tile_1-2)
   (right tile_1-4 tile_1-3)
   (right tile_1-5 tile_1-4)
   (right tile_1-6 tile_1-5)
   (right tile_2-2 tile_2-1)
   (right tile_2-3 tile_2-2)
   (right tile_2-4 tile_2-3)
   (right tile_2-5 tile_2-4)
   (right tile_2-6 tile_2-5)
   (right tile_3-2 tile_3-1)
   (right tile_3-3 tile_3-2)
   (right tile_3-4 tile_3-3)
   (right tile_3-5 tile_3-4)
   (right tile_3-6 tile_3-5)
   (right tile_4-2 tile_4-1)
   (right tile_4-3 tile_4-2)
   (right tile_4-4 tile_4-3)
   (right tile_4-5 tile_4-4)
   (right tile_4-6 tile_4-5)
))
)"""


@pytest.fixture
def floortile_disconnected_tile1():
    return """(define (problem one_tile)
     (:domain floor-tile)
     (:objects tile_0-1 tile_0-2 tile_0-3 tile_0-4 tile_0-5
               tile_1-1 tile_1-2 tile_1-3 tile_1-4 tile_1-5
               tile_2-1 tile_2-2 tile_2-3 tile_2-4 tile_2-5
               tile_3-1 tile_3-2 tile_3-3 tile_3-4 tile_3-5
               tile_4-1 tile_4-2 tile_4-3 tile_4-4 tile_4-5
               tile_5-1 tile_5-2 tile_5-3 tile_5-4 tile_5-5
               tile_6-1 - tile
               robot1 robot2 robot3 - robot
               white black - color
    )
     (:init
       (robot-at robot1 tile_2-4)
       (robot-has robot1 white)
       (robot-at robot2 tile_0-3)
       (robot-has robot2 black)
       (robot-at robot3 tile_6-1)
       (robot-has robot3 white)
       (available-color white)
       (available-color black)
       (up tile_1-1 tile_0-1)
       (up tile_1-2 tile_0-2)
       (up tile_1-3 tile_0-3)
       (up tile_1-4 tile_0-4)
       (up tile_1-5 tile_0-5)
       (up tile_2-1 tile_1-1)
       (up tile_2-2 tile_1-2)
       (up tile_2-3 tile_1-3)
       (up tile_2-4 tile_1-4)
       (up tile_2-5 tile_1-5)
       (up tile_3-1 tile_2-1)
       (up tile_3-2 tile_2-2)
       (up tile_3-3 tile_2-3)
       (up tile_3-4 tile_2-4)
       (up tile_3-5 tile_2-5)
       (up tile_4-1 tile_3-1)
       (up tile_4-2 tile_3-2)
       (up tile_4-3 tile_3-3)
       (up tile_4-4 tile_3-4)
       (up tile_4-5 tile_3-5)
       (up tile_5-1 tile_4-1)
       (up tile_5-2 tile_4-2)
       (up tile_5-3 tile_4-3)
       (up tile_5-4 tile_4-4)
       (up tile_5-5 tile_4-5)
       (right tile_0-2 tile_0-1)
       (right tile_0-3 tile_0-2)
       (right tile_0-4 tile_0-3)
       (right tile_0-5 tile_0-4)
       (right tile_1-2 tile_1-1)
       (right tile_1-3 tile_1-2)
       (right tile_1-4 tile_1-3)
       (right tile_1-5 tile_1-4)
       (right tile_2-2 tile_2-1)
       (right tile_2-3 tile_2-2)
       (right tile_2-4 tile_2-3)
       (right tile_2-5 tile_2-4)
       (right tile_3-2 tile_3-1)
       (right tile_3-3 tile_3-2)
       (right tile_3-4 tile_3-3)
       (right tile_3-5 tile_3-4)
       (right tile_4-2 tile_4-1)
       (right tile_4-3 tile_4-2)
       (right tile_4-4 tile_4-3)
       (right tile_4-5 tile_4-4)
       (right tile_5-2 tile_5-1)
       (right tile_5-3 tile_5-2)
       (right tile_5-4 tile_5-3)
       (right tile_5-5 tile_5-4)
    )
     (:goal (and
        (painted tile_1-1 white)
        (painted tile_1-2 black)
        (painted tile_1-3 white)
        (painted tile_1-4 black)
        (painted tile_1-5 white)
        (painted tile_2-1 black)
        (painted tile_2-2 white)
        (painted tile_2-3 black)
        (painted tile_2-4 white)
        (painted tile_2-5 black)
        (painted tile_3-1 white)
        (painted tile_3-2 black)
        (painted tile_3-3 white)
        (painted tile_3-4 black)
        (painted tile_3-5 white)
        (painted tile_4-1 black)
        (painted tile_4-2 white)
        (painted tile_4-3 black)
        (painted tile_4-4 white)
        (painted tile_4-5 black)
        (painted tile_5-1 white)
        (painted tile_5-2 black)
        (painted tile_5-3 white)
        (painted tile_5-4 black)
        (painted tile_5-5 white)
        (up tile_1-1 tile_0-1)
        (up tile_1-2 tile_0-2)
        (up tile_1-3 tile_0-3)
        (up tile_1-4 tile_0-4)
        (up tile_1-5 tile_0-5)
        (up tile_2-1 tile_1-1)
        (up tile_2-2 tile_1-2)
        (up tile_2-3 tile_1-3)
        (up tile_2-4 tile_1-4)
        (up tile_2-5 tile_1-5)
        (up tile_3-1 tile_2-1)
        (up tile_3-2 tile_2-2)
        (up tile_3-3 tile_2-3)
        (up tile_3-4 tile_2-4)
        (up tile_3-5 tile_2-5)
        (up tile_4-1 tile_3-1)
        (up tile_4-2 tile_3-2)
        (up tile_4-3 tile_3-3)
        (up tile_4-4 tile_3-4)
        (up tile_4-5 tile_3-5)
        (up tile_5-1 tile_4-1)
        (up tile_5-2 tile_4-2)
        (up tile_5-3 tile_4-3)
        (up tile_5-4 tile_4-4)
        (up tile_5-5 tile_4-5)
        (right tile_0-2 tile_0-1)
        (right tile_0-3 tile_0-2)
        (right tile_0-4 tile_0-3)
        (right tile_0-5 tile_0-4)
        (right tile_1-2 tile_1-1)
        (right tile_1-3 tile_1-2)
        (right tile_1-4 tile_1-3)
        (right tile_1-5 tile_1-4)
        (right tile_2-2 tile_2-1)
        (right tile_2-3 tile_2-2)
        (right tile_2-4 tile_2-3)
        (right tile_2-5 tile_2-4)
        (right tile_3-2 tile_3-1)
        (right tile_3-3 tile_3-2)
        (right tile_3-4 tile_3-3)
        (right tile_3-5 tile_3-4)
        (right tile_4-2 tile_4-1)
        (right tile_4-3 tile_4-2)
        (right tile_4-4 tile_4-3)
        (right tile_4-5 tile_4-4)
        (right tile_5-2 tile_5-1)
        (right tile_5-3 tile_5-2)
        (right tile_5-4 tile_5-3)
        (right tile_5-5 tile_5-4)
        (robot-at robot3 tile_6-1)
        (available-color white)
        (available-color black)
    ))
    )
    """


@pytest.fixture
def floortile_disconnected_tile1a():
    return """(define (problem one_tile)
     (:domain floor-tile)
     (:objects tile_0-1 tile_0-2 tile_0-3 tile_0-4 tile_0-5
               tile_1-1 tile_1-2 tile_1-3 tile_1-4 tile_1-5
               tile_2-1 tile_2-2 tile_2-3 tile_2-4 tile_2-5
               tile_3-1 tile_3-2 tile_3-3 tile_3-4 tile_3-5
               tile_4-1 tile_4-2 tile_4-3 tile_4-4 tile_4-5
               tile_5-1 tile_5-2 tile_5-3 tile_5-4 tile_5-5
               tile_6-1 - tile
               robot1 robot2 robot3 - robot
               white black - color
    )
     (:init
       (robot-at robot1 tile_2-4)
       (robot-has robot1 white)
       (robot-at robot2 tile_0-3)
       (robot-has robot2 black)
       (robot-at robot3 tile_6-1)
       (robot-has robot3 white)
       (available-color white)
       (available-color black)
       (up tile_1-1 tile_0-1)
       (up tile_1-2 tile_0-2)
       (up tile_1-3 tile_0-3)
       (up tile_1-4 tile_0-4)
       (up tile_1-5 tile_0-5)
       (up tile_2-1 tile_1-1)
       (up tile_2-2 tile_1-2)
       (up tile_2-3 tile_1-3)
       (up tile_2-4 tile_1-4)
       (up tile_2-5 tile_1-5)
       (up tile_3-1 tile_2-1)
       (up tile_3-2 tile_2-2)
       (up tile_3-3 tile_2-3)
       (up tile_3-4 tile_2-4)
       (up tile_3-5 tile_2-5)
       (up tile_4-1 tile_3-1)
       (up tile_4-2 tile_3-2)
       (up tile_4-3 tile_3-3)
       (up tile_4-4 tile_3-4)
       (up tile_4-5 tile_3-5)
       (up tile_5-1 tile_4-1)
       (up tile_5-2 tile_4-2)
       (up tile_5-3 tile_4-3)
       (up tile_5-4 tile_4-4)
       (up tile_5-5 tile_4-5)
       (right tile_0-2 tile_0-1)
       (right tile_0-3 tile_0-2)
       (right tile_0-4 tile_0-3)
       (right tile_0-5 tile_0-4)
       (right tile_1-2 tile_1-1)
       (right tile_1-3 tile_1-2)
       (right tile_1-4 tile_1-3)
       (right tile_1-5 tile_1-4)
       (right tile_2-2 tile_2-1)
       (right tile_2-3 tile_2-2)
       (right tile_2-4 tile_2-3)
       (right tile_2-5 tile_2-4)
       (right tile_3-2 tile_3-1)
       (right tile_3-3 tile_3-2)
       (right tile_3-4 tile_3-3)
       (right tile_3-5 tile_3-4)
       (right tile_4-2 tile_4-1)
       (right tile_4-3 tile_4-2)
       (right tile_4-4 tile_4-3)
       (right tile_4-5 tile_4-4)
       (right tile_5-2 tile_5-1)
       (right tile_5-3 tile_5-2)
       (right tile_5-4 tile_5-3)
       (right tile_5-5 tile_5-4)
    )
     (:goal (and
        (painted tile_1-1 white)
        (painted tile_1-2 black)
        (painted tile_1-3 white)
        (painted tile_1-4 black)
        (painted tile_1-5 white)
        (painted tile_2-1 black)
        (painted tile_2-2 white)
        (painted tile_2-3 black)
        (painted tile_2-4 white)
        (painted tile_2-5 black)
        (painted tile_3-1 white)
        (painted tile_3-2 black)
        (painted tile_3-3 white)
        (painted tile_3-4 black)
        (painted tile_3-5 white)
        (painted tile_4-1 black)
        (painted tile_4-2 white)
        (painted tile_4-3 black)
        (painted tile_4-4 white)
        (painted tile_4-5 black)
        (painted tile_5-1 white)
        (painted tile_5-2 black)
        (painted tile_5-3 white)
        (painted tile_5-4 black)
        (painted tile_5-5 white)
    ))
    )
    """


@pytest.fixture
def floortile_disconnected_tile_no_white():
    return """(define (problem two_tile_no_white)
     (:domain floor-tile)
     (:objects tile_0-1 tile_0-2 tile_0-3 tile_0-4 tile_0-5
               tile_1-1 tile_1-2 tile_1-3 tile_1-4 tile_1-5
               tile_2-1 tile_2-2 tile_2-3 tile_2-4 tile_2-5
               tile_3-1 tile_3-2 tile_3-3 tile_3-4 tile_3-5
               tile_4-1 tile_4-2 tile_4-3 tile_4-4 tile_4-5
               tile_5-1 tile_5-2 tile_5-3 tile_5-4 tile_5-5
               tile_6-1 tile_6-2 - tile
               robot1 robot2 robot3 - robot
               white black - color
    )
     (:init
       (robot-at robot1 tile_2-4)
       (robot-has robot1 white)
       (robot-at robot2 tile_0-3)
       (robot-has robot2 black)
       (robot-at robot3 tile_6-1)
       (robot-has robot3 white)
       (available-color black)
       (up tile_1-1 tile_0-1)
       (up tile_1-2 tile_0-2)
       (up tile_1-3 tile_0-3)
       (up tile_1-4 tile_0-4)
       (up tile_1-5 tile_0-5)
       (up tile_2-1 tile_1-1)
       (up tile_2-2 tile_1-2)
       (up tile_2-3 tile_1-3)
       (up tile_2-4 tile_1-4)
       (up tile_2-5 tile_1-5)
       (up tile_3-1 tile_2-1)
       (up tile_3-2 tile_2-2)
       (up tile_3-3 tile_2-3)
       (up tile_3-4 tile_2-4)
       (up tile_3-5 tile_2-5)
       (up tile_4-1 tile_3-1)
       (up tile_4-2 tile_3-2)
       (up tile_4-3 tile_3-3)
       (up tile_4-4 tile_3-4)
       (up tile_4-5 tile_3-5)
       (up tile_5-1 tile_4-1)
       (up tile_5-2 tile_4-2)
       (up tile_5-3 tile_4-3)
       (up tile_5-4 tile_4-4)
       (up tile_6-1 tile_6-2)
       (right tile_0-2 tile_0-1)
       (right tile_0-3 tile_0-2)
       (right tile_0-4 tile_0-3)
       (right tile_0-5 tile_0-4)
       (right tile_1-2 tile_1-1)
       (right tile_1-3 tile_1-2)
       (right tile_1-4 tile_1-3)
       (right tile_1-5 tile_1-4)
       (right tile_2-2 tile_2-1)
       (right tile_2-3 tile_2-2)
       (right tile_2-4 tile_2-3)
       (right tile_2-5 tile_2-4)
       (right tile_3-2 tile_3-1)
       (right tile_3-3 tile_3-2)
       (right tile_3-4 tile_3-3)
       (right tile_3-5 tile_3-4)
       (right tile_4-2 tile_4-1)
       (right tile_4-3 tile_4-2)
       (right tile_4-4 tile_4-3)
       (right tile_4-5 tile_4-4)
       (right tile_5-2 tile_5-1)
       (right tile_5-3 tile_5-2)
       (right tile_5-4 tile_5-3)
       (right tile_5-5 tile_5-4)
    )
     (:goal (and
        (painted tile_1-1 white)
        (painted tile_1-2 black)
        (painted tile_1-3 white)
        (painted tile_1-4 black)
        (painted tile_1-5 white)
        (painted tile_2-1 black)
        (painted tile_2-2 white)
        (painted tile_2-3 black)
        (painted tile_2-4 white)
        (painted tile_2-5 black)
        (painted tile_3-1 white)
        (painted tile_3-2 black)
        (painted tile_3-3 white)
        (painted tile_3-4 black)
        (painted tile_3-5 white)
        (painted tile_4-1 black)
        (painted tile_4-2 white)
        (painted tile_4-3 black)
        (painted tile_4-4 white)
        (painted tile_4-5 black)
        (painted tile_5-1 white)
        (painted tile_5-2 black)
        (painted tile_5-3 white)
        (painted tile_5-4 black)
        (painted tile_5-5 white)
        (painted tile_6-1 black)
        (painted tile_6-2 white)
        (up tile_1-1 tile_0-1)
        (up tile_1-2 tile_0-2)
        (up tile_1-3 tile_0-3)
        (up tile_1-4 tile_0-4)
        (up tile_1-5 tile_0-5)
        (up tile_2-1 tile_1-1)
        (up tile_2-2 tile_1-2)
        (up tile_2-3 tile_1-3)
        (up tile_2-4 tile_1-4)
        (up tile_2-5 tile_1-5)
        (up tile_3-1 tile_2-1)
        (up tile_3-2 tile_2-2)
        (up tile_3-3 tile_2-3)
        (up tile_3-4 tile_2-4)
        (up tile_3-5 tile_2-5)
        (up tile_4-1 tile_3-1)
        (up tile_4-2 tile_3-2)
        (up tile_4-3 tile_3-3)
        (up tile_4-4 tile_3-4)
        (up tile_4-5 tile_3-5)
        (up tile_5-1 tile_4-1)
        (up tile_5-2 tile_4-2)
        (up tile_5-3 tile_4-3)
        (up tile_5-4 tile_4-4)
        (up tile_6-1 tile_6-2)
        (right tile_0-2 tile_0-1)
        (right tile_0-3 tile_0-2)
        (right tile_0-4 tile_0-3)
        (right tile_0-5 tile_0-4)
        (right tile_1-2 tile_1-1)
        (right tile_1-3 tile_1-2)
        (right tile_1-4 tile_1-3)
        (right tile_1-5 tile_1-4)
        (right tile_2-2 tile_2-1)
        (right tile_2-3 tile_2-2)
        (right tile_2-4 tile_2-3)
        (right tile_2-5 tile_2-4)
        (right tile_3-2 tile_3-1)
        (right tile_3-3 tile_3-2)
        (right tile_3-4 tile_3-3)
        (right tile_3-5 tile_3-4)
        (right tile_4-2 tile_4-1)
        (right tile_4-3 tile_4-2)
        (right tile_4-4 tile_4-3)
        (right tile_4-5 tile_4-4)
        (right tile_5-2 tile_5-1)
        (right tile_5-3 tile_5-2)
        (right tile_5-4 tile_5-3)
        (right tile_5-5 tile_5-4)
        (robot-has robot2 black)
        (robot-has robot3 black)
        (available-color black)
    ))
    )
    """


@pytest.fixture
def floortile_no_available_colors():
    return """(define (problem no_available_colors)
     (:domain floor-tile)
     (:objects tile_0-1 tile_0-2 tile_0-3 tile_0-4 tile_0-5
               tile_1-1 tile_1-2 tile_1-3 tile_1-4 tile_1-5
               tile_2-1 tile_2-2 tile_2-3 tile_2-4 tile_2-5
               tile_3-1 tile_3-2 tile_3-3 tile_3-4 tile_3-5
               tile_4-1 tile_4-2 tile_4-3 tile_4-4 tile_4-5
               tile_5-1 tile_5-2 tile_5-3 tile_5-4 tile_5-5 - tile
               robot1 robot2 robot3 - robot
               white black - color
    )
     (:init
       (robot-at robot1 tile_2-4)
       (robot-has robot1 white)
       (robot-at robot2 tile_0-3)
       (robot-has robot2 black)
       (robot-at robot3 tile_5-5)
       (robot-has robot3 white)
       (up tile_1-1 tile_0-1)
       (up tile_1-2 tile_0-2)
       (up tile_1-3 tile_0-3)
       (up tile_1-4 tile_0-4)
       (up tile_1-5 tile_0-5)
       (up tile_2-1 tile_1-1)
       (up tile_2-2 tile_1-2)
       (up tile_2-3 tile_1-3)
       (up tile_2-4 tile_1-4)
       (up tile_2-5 tile_1-5)
       (up tile_3-1 tile_2-1)
       (up tile_3-2 tile_2-2)
       (up tile_3-3 tile_2-3)
       (up tile_3-4 tile_2-4)
       (up tile_3-5 tile_2-5)
       (up tile_4-1 tile_3-1)
       (up tile_4-2 tile_3-2)
       (up tile_4-3 tile_3-3)
       (up tile_4-4 tile_3-4)
       (up tile_4-5 tile_3-5)
       (up tile_5-1 tile_4-1)
       (up tile_5-2 tile_4-2)
       (up tile_5-3 tile_4-3)
       (up tile_5-4 tile_4-4)
       (right tile_0-2 tile_0-1)
       (right tile_0-3 tile_0-2)
       (right tile_0-4 tile_0-3)
       (right tile_0-5 tile_0-4)
       (right tile_1-2 tile_1-1)
       (right tile_1-3 tile_1-2)
       (right tile_1-4 tile_1-3)
       (right tile_1-5 tile_1-4)
       (right tile_2-2 tile_2-1)
       (right tile_2-3 tile_2-2)
       (right tile_2-4 tile_2-3)
       (right tile_2-5 tile_2-4)
       (right tile_3-2 tile_3-1)
       (right tile_3-3 tile_3-2)
       (right tile_3-4 tile_3-3)
       (right tile_3-5 tile_3-4)
       (right tile_4-2 tile_4-1)
       (right tile_4-3 tile_4-2)
       (right tile_4-4 tile_4-3)
       (right tile_4-5 tile_4-4)
       (right tile_5-2 tile_5-1)
       (right tile_5-3 tile_5-2)
       (right tile_5-4 tile_5-3)
       (right tile_5-5 tile_5-4)
    )
     (:goal (and
        (painted tile_1-1 white)
        (painted tile_1-2 black)
        (painted tile_1-3 white)
        (painted tile_1-4 black)
        (painted tile_1-5 white)
        (painted tile_2-1 black)
        (painted tile_2-2 white)
        (painted tile_2-3 black)
        (painted tile_2-4 white)
        (painted tile_2-5 black)
        (painted tile_3-1 white)
        (painted tile_3-2 black)
        (painted tile_3-3 white)
        (painted tile_3-4 black)
        (painted tile_3-5 white)
        (painted tile_4-1 black)
        (painted tile_4-2 white)
        (painted tile_4-3 black)
        (painted tile_4-4 white)
        (painted tile_4-5 black)
        (painted tile_5-1 white)
        (painted tile_5-2 black)
        (painted tile_5-3 white)
        (painted tile_5-4 black)
        (painted tile_5-5 white)
       (up tile_1-1 tile_0-1)
       (up tile_1-2 tile_0-2)
       (up tile_1-3 tile_0-3)
       (up tile_1-4 tile_0-4)
       (up tile_1-5 tile_0-5)
       (up tile_2-1 tile_1-1)
       (up tile_2-2 tile_1-2)
       (up tile_2-3 tile_1-3)
       (up tile_2-4 tile_1-4)
       (up tile_2-5 tile_1-5)
       (up tile_3-1 tile_2-1)
       (up tile_3-2 tile_2-2)
       (up tile_3-3 tile_2-3)
       (up tile_3-4 tile_2-4)
       (up tile_3-5 tile_2-5)
       (up tile_4-1 tile_3-1)
       (up tile_4-2 tile_3-2)
       (up tile_4-3 tile_3-3)
       (up tile_4-4 tile_3-4)
       (up tile_4-5 tile_3-5)
       (up tile_5-1 tile_4-1)
       (up tile_5-2 tile_4-2)
       (up tile_5-3 tile_4-3)
       (up tile_5-4 tile_4-4)
       (right tile_0-2 tile_0-1)
       (right tile_0-3 tile_0-2)
       (right tile_0-4 tile_0-3)
       (right tile_0-5 tile_0-4)
       (right tile_1-2 tile_1-1)
       (right tile_1-3 tile_1-2)
       (right tile_1-4 tile_1-3)
       (right tile_1-5 tile_1-4)
       (right tile_2-2 tile_2-1)
       (right tile_2-3 tile_2-2)
       (right tile_2-4 tile_2-3)
       (right tile_2-5 tile_2-4)
       (right tile_3-2 tile_3-1)
       (right tile_3-3 tile_3-2)
       (right tile_3-4 tile_3-3)
       (right tile_3-5 tile_3-4)
       (right tile_4-2 tile_4-1)
       (right tile_4-3 tile_4-2)
       (right tile_4-4 tile_4-3)
       (right tile_4-5 tile_4-4)
       (right tile_5-2 tile_5-1)
       (right tile_5-3 tile_5-2)
       (right tile_5-4 tile_5-3)
       (right tile_5-5 tile_5-4)
       (robot-has robot1 white)
       (robot-has robot2 black)
       (robot-has robot3 white)
    ))
    )
    """


@pytest.fixture
def floortile_no_available_colors_a():
    return """(define (problem no_available_colors)
     (:domain floor-tile)
     (:objects tile_0-1 tile_0-2 tile_0-3 tile_0-4 tile_0-5
               tile_1-1 tile_1-2 tile_1-3 tile_1-4 tile_1-5
               tile_2-1 tile_2-2 tile_2-3 tile_2-4 tile_2-5
               tile_3-1 tile_3-2 tile_3-3 tile_3-4 tile_3-5
               tile_4-1 tile_4-2 tile_4-3 tile_4-4 tile_4-5
               tile_5-1 tile_5-2 tile_5-3 tile_5-4 tile_5-5 - tile
               robot1 robot2 robot3 - robot
               white black - color
    )
     (:init
       (robot-at robot1 tile_2-4)
       (robot-has robot1 white)
       (robot-at robot2 tile_0-3)
       (robot-has robot2 black)
       (robot-at robot3 tile_5-5)
       (robot-has robot3 white)
       (up tile_1-1 tile_0-1)
       (up tile_1-2 tile_0-2)
       (up tile_1-3 tile_0-3)
       (up tile_1-4 tile_0-4)
       (up tile_1-5 tile_0-5)
       (up tile_2-1 tile_1-1)
       (up tile_2-2 tile_1-2)
       (up tile_2-3 tile_1-3)
       (up tile_2-4 tile_1-4)
       (up tile_2-5 tile_1-5)
       (up tile_3-1 tile_2-1)
       (up tile_3-2 tile_2-2)
       (up tile_3-3 tile_2-3)
       (up tile_3-4 tile_2-4)
       (up tile_3-5 tile_2-5)
       (up tile_4-1 tile_3-1)
       (up tile_4-2 tile_3-2)
       (up tile_4-3 tile_3-3)
       (up tile_4-4 tile_3-4)
       (up tile_4-5 tile_3-5)
       (up tile_5-1 tile_4-1)
       (up tile_5-2 tile_4-2)
       (up tile_5-3 tile_4-3)
       (up tile_5-4 tile_4-4)
       (right tile_0-2 tile_0-1)
       (right tile_0-3 tile_0-2)
       (right tile_0-4 tile_0-3)
       (right tile_0-5 tile_0-4)
       (right tile_1-2 tile_1-1)
       (right tile_1-3 tile_1-2)
       (right tile_1-4 tile_1-3)
       (right tile_1-5 tile_1-4)
       (right tile_2-2 tile_2-1)
       (right tile_2-3 tile_2-2)
       (right tile_2-4 tile_2-3)
       (right tile_2-5 tile_2-4)
       (right tile_3-2 tile_3-1)
       (right tile_3-3 tile_3-2)
       (right tile_3-4 tile_3-3)
       (right tile_3-5 tile_3-4)
       (right tile_4-2 tile_4-1)
       (right tile_4-3 tile_4-2)
       (right tile_4-4 tile_4-3)
       (right tile_4-5 tile_4-4)
       (right tile_5-2 tile_5-1)
       (right tile_5-3 tile_5-2)
       (right tile_5-4 tile_5-3)
       (right tile_5-5 tile_5-4)
    )
     (:goal (and
        (painted tile_1-1 white)
        (painted tile_1-2 black)
        (painted tile_1-3 white)
        (painted tile_1-4 black)
        (painted tile_1-5 white)
        (painted tile_2-1 black)
        (painted tile_2-2 white)
        (painted tile_2-3 black)
        (painted tile_2-4 white)
        (painted tile_2-5 black)
        (painted tile_3-1 white)
        (painted tile_3-2 black)
        (painted tile_3-3 white)
        (painted tile_3-4 black)
        (painted tile_3-5 white)
        (painted tile_4-1 black)
        (painted tile_4-2 white)
        (painted tile_4-3 black)
        (painted tile_4-4 white)
        (painted tile_4-5 black)
        (painted tile_5-1 white)
        (painted tile_5-2 black)
        (painted tile_5-3 white)
        (painted tile_5-4 black)
        (painted tile_5-5 white)
    ))
    )
    """


@pytest.fixture
def floortile_one_color_one_robot1():
    return """(define (problem one_available_color)
     (:domain floor-tile)
     (:objects tile_0-1 tile_0-2 tile_0-3 tile_0-4 tile_0-5
               tile_1-1 tile_1-2 tile_1-3 tile_1-4 tile_1-5
               tile_2-1 tile_2-2 tile_2-3 tile_2-4 tile_2-5
               tile_3-1 tile_3-2 tile_3-3 tile_3-4 tile_3-5
               tile_4-1 tile_4-2 tile_4-3 tile_4-4 tile_4-5
               tile_5-1 tile_5-2 tile_5-3 tile_5-4 tile_5-5 - tile
               robot1 - robot
               white black - color
    )
     (:init
       (robot-at robot1 tile_2-4)
       (robot-has robot1 white)
       (available-color white)
       (up tile_1-1 tile_0-1)
       (up tile_1-2 tile_0-2)
       (up tile_1-3 tile_0-3)
       (up tile_1-4 tile_0-4)
       (up tile_1-5 tile_0-5)
       (up tile_2-1 tile_1-1)
       (up tile_2-2 tile_1-2)
       (up tile_2-3 tile_1-3)
       (up tile_2-4 tile_1-4)
       (up tile_2-5 tile_1-5)
       (up tile_3-1 tile_2-1)
       (up tile_3-2 tile_2-2)
       (up tile_3-3 tile_2-3)
       (up tile_3-4 tile_2-4)
       (up tile_3-5 tile_2-5)
       (up tile_4-1 tile_3-1)
       (up tile_4-2 tile_3-2)
       (up tile_4-3 tile_3-3)
       (up tile_4-4 tile_3-4)
       (up tile_4-5 tile_3-5)
       (up tile_5-1 tile_4-1)
       (up tile_5-2 tile_4-2)
       (up tile_5-3 tile_4-3)
       (up tile_5-4 tile_4-4)
       (right tile_0-2 tile_0-1)
       (right tile_0-3 tile_0-2)
       (right tile_0-4 tile_0-3)
       (right tile_0-5 tile_0-4)
       (right tile_1-2 tile_1-1)
       (right tile_1-3 tile_1-2)
       (right tile_1-4 tile_1-3)
       (right tile_1-5 tile_1-4)
       (right tile_2-2 tile_2-1)
       (right tile_2-3 tile_2-2)
       (right tile_2-4 tile_2-3)
       (right tile_2-5 tile_2-4)
       (right tile_3-2 tile_3-1)
       (right tile_3-3 tile_3-2)
       (right tile_3-4 tile_3-3)
       (right tile_3-5 tile_3-4)
       (right tile_4-2 tile_4-1)
       (right tile_4-3 tile_4-2)
       (right tile_4-4 tile_4-3)
       (right tile_4-5 tile_4-4)
       (right tile_5-2 tile_5-1)
       (right tile_5-3 tile_5-2)
       (right tile_5-4 tile_5-3)
       (right tile_5-5 tile_5-4)
    )
     (:goal (and
       (up tile_1-1 tile_0-1)
       (up tile_1-2 tile_0-2)
       (up tile_1-3 tile_0-3)
       (up tile_1-4 tile_0-4)
       (up tile_1-5 tile_0-5)
       (up tile_2-1 tile_1-1)
       (up tile_2-2 tile_1-2)
       (up tile_2-3 tile_1-3)
       (up tile_2-4 tile_1-4)
       (up tile_2-5 tile_1-5)
       (up tile_3-1 tile_2-1)
       (up tile_3-2 tile_2-2)
       (up tile_3-3 tile_2-3)
       (up tile_3-4 tile_2-4)
       (up tile_3-5 tile_2-5)
       (up tile_4-1 tile_3-1)
       (up tile_4-2 tile_3-2)
       (up tile_4-3 tile_3-3)
       (up tile_4-4 tile_3-4)
       (up tile_4-5 tile_3-5)
       (up tile_5-1 tile_4-1)
       (up tile_5-2 tile_4-2)
       (up tile_5-3 tile_4-3)
       (up tile_5-4 tile_4-4)
       (right tile_0-2 tile_0-1)
       (right tile_0-3 tile_0-2)
       (right tile_0-4 tile_0-3)
       (right tile_0-5 tile_0-4)
       (right tile_1-2 tile_1-1)
       (right tile_1-3 tile_1-2)
       (right tile_1-4 tile_1-3)
       (right tile_1-5 tile_1-4)
       (right tile_2-2 tile_2-1)
       (right tile_2-3 tile_2-2)
       (right tile_2-4 tile_2-3)
       (right tile_2-5 tile_2-4)
       (right tile_3-2 tile_3-1)
       (right tile_3-3 tile_3-2)
       (right tile_3-4 tile_3-3)
       (right tile_3-5 tile_3-4)
       (right tile_4-2 tile_4-1)
       (right tile_4-3 tile_4-2)
       (right tile_4-4 tile_4-3)
       (right tile_4-5 tile_4-4)
       (right tile_5-2 tile_5-1)
       (right tile_5-3 tile_5-2)
       (right tile_5-4 tile_5-3)
       (right tile_5-5 tile_5-4)
       (robot-has robot1 white)
       (available-color white)
    ))
    )
    """


@pytest.fixture
def floortile_one_color_one_robot1a():
    return """(define (problem one_available_color)
     (:domain floor-tile)
     (:objects tile_0-1 tile_0-2 tile_0-3 tile_0-4 tile_0-5
               tile_1-1 tile_1-2 tile_1-3 tile_1-4 tile_1-5
               tile_2-1 tile_2-2 tile_2-3 tile_2-4 tile_2-5
               tile_3-1 tile_3-2 tile_3-3 tile_3-4 tile_3-5
               tile_4-1 tile_4-2 tile_4-3 tile_4-4 tile_4-5
               tile_5-1 tile_5-2 tile_5-3 tile_5-4 tile_5-5 - tile
               robot1 - robot
               white black - color
    )
     (:init
       (robot-at robot1 tile_2-4)
       (robot-has robot1 white)
       (available-color white)
       (up tile_1-1 tile_0-1)
       (up tile_1-2 tile_0-2)
       (up tile_1-3 tile_0-3)
       (up tile_1-4 tile_0-4)
       (up tile_1-5 tile_0-5)
       (up tile_2-1 tile_1-1)
       (up tile_2-2 tile_1-2)
       (up tile_2-3 tile_1-3)
       (up tile_2-4 tile_1-4)
       (up tile_2-5 tile_1-5)
       (up tile_3-1 tile_2-1)
       (up tile_3-2 tile_2-2)
       (up tile_3-3 tile_2-3)
       (up tile_3-4 tile_2-4)
       (up tile_3-5 tile_2-5)
       (up tile_4-1 tile_3-1)
       (up tile_4-2 tile_3-2)
       (up tile_4-3 tile_3-3)
       (up tile_4-4 tile_3-4)
       (up tile_4-5 tile_3-5)
       (up tile_5-1 tile_4-1)
       (up tile_5-2 tile_4-2)
       (up tile_5-3 tile_4-3)
       (up tile_5-4 tile_4-4)
       (right tile_0-2 tile_0-1)
       (right tile_0-3 tile_0-2)
       (right tile_0-4 tile_0-3)
       (right tile_0-5 tile_0-4)
       (right tile_1-2 tile_1-1)
       (right tile_1-3 tile_1-2)
       (right tile_1-4 tile_1-3)
       (right tile_1-5 tile_1-4)
       (right tile_2-2 tile_2-1)
       (right tile_2-3 tile_2-2)
       (right tile_2-4 tile_2-3)
       (right tile_2-5 tile_2-4)
       (right tile_3-2 tile_3-1)
       (right tile_3-3 tile_3-2)
       (right tile_3-4 tile_3-3)
       (right tile_3-5 tile_3-4)
       (right tile_4-2 tile_4-1)
       (right tile_4-3 tile_4-2)
       (right tile_4-4 tile_4-3)
       (right tile_4-5 tile_4-4)
       (right tile_5-2 tile_5-1)
       (right tile_5-3 tile_5-2)
       (right tile_5-4 tile_5-3)
       (right tile_5-5 tile_5-4)
    )
     (:goal (and
       (right tile_5-5 tile_5-4)
    ))
    )
    """
