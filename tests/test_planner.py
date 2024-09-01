import os
import pytest

VALIDATE = os.getenv("VALIDATE", "Validate")

from planetarium import builder, downward, oracle

from .problem_fixtures import (
    blocksworld_fully_specified,
    blocksworld_holding,
    blocksworld_missing_clears,
    blocksworld_missing_ontables,
    blocksworld_underspecified,
    blocksworld_underspecified_arm,
    blocksworld_stack_to_holding,
    blocksworld_invalid_1,
    blocksworld_invalid_2,
    blocksworld_invalid_3,
    gripper_fully_specified,
    gripper_fully_specified_not_strict,
    gripper_inconsistent_typing,
    gripper_missing_typing,
    gripper_multiple_typing,
    gripper_no_goal_types,
    gripper_no_robby,
    gripper_robby_at_last,
    gripper_underspecified_1,
    gripper_underspecified_2,
    gripper_underspecified_3,
    gripper_invalid,
)

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


class TestBlocksworldOracle:
    """
    Test suite for the blocksworld oracle.
    """

    def test_plan(
        self,
        subtests,
        blocksworld_missing_clears,
        blocksworld_fully_specified,
        blocksworld_holding,
        blocksworld_missing_ontables,
        blocksworld_underspecified,
        blocksworld_underspecified_arm,
        blocksworld_stack_to_holding,
    ):
        """
        Test if the oracle can plan for a fully specified blocksworld problem.
        """
        for name, desc in {
            "blocksworld_fully_specified": blocksworld_fully_specified,
            "blocksworld_holding": blocksworld_holding,
            "blocksworld_missing_clears": blocksworld_missing_clears,
            "blocksworld_missing_ontables": blocksworld_missing_ontables,
            "blocksworld_underspecified": blocksworld_underspecified,
            "blocksworld_underspecified_arm": blocksworld_underspecified_arm,
            "blocksworld_stack_to_holding": blocksworld_stack_to_holding,
        }.items():
            plan = oracle.plan(builder.build(desc))
            with subtests.test(name):
                assert plan != [], name

                assert downward.validate(
                    DOMAINS["blocksworld"],
                    desc,
                    oracle.plan_to_string(plan),
                    VALIDATE,
                )

        with subtests.test(name):
            assert not downward.validate(
                DOMAINS["gripper"],
                desc,
                oracle.plan_to_string(plan),
                VALIDATE,
            )

    def test_invalid_plan(
        self,
        subtests,
        blocksworld_invalid_2,
    ):
        """
        Test if the oracle can plan for an invalid blocksworld problem.
        """
        domain = DOMAINS["blocksworld"]
        for name, desc in {
            "blocksworld_invalid_2": blocksworld_invalid_2,
        }.items():
            with subtests.test(name):
                try:
                    plan = oracle.plan(builder.build(desc))
                except Exception as e:
                    plan = []
                assert plan == [], f"{name}: {plan}"

                plan_str = oracle.plan_to_string(plan)
                assert not downward.validate(domain, desc, plan_str, VALIDATE)


class TestGripperOracle:
    """
    Test suite for the gripper oracle.
    """

    def test_plan(
        self,
        subtests,
        gripper_fully_specified,
        gripper_fully_specified_not_strict,
        gripper_no_goal_types,
        gripper_no_robby,
        gripper_robby_at_last,
        gripper_underspecified_1,
        gripper_underspecified_2,
        gripper_underspecified_3,
    ):
        """
        Test if the oracle can plan for a fully specified gripper problem.
        """
        domain = DOMAINS["gripper"]
        for name, desc in {
            "gripper_fully_specified": gripper_fully_specified,
            "gripper_fully_specified_not_strict": gripper_fully_specified_not_strict,
            "gripper_no_goal_types": gripper_no_goal_types,
            "gripper_no_robby": gripper_no_robby,
            "gripper_robby_at_last": gripper_robby_at_last,
            "gripper_underspecified_1": gripper_underspecified_1,
            "gripper_underspecified_2": gripper_underspecified_2,
            "gripper_underspecified_3": gripper_underspecified_3,
        }.items():
            with subtests.test(name):
                plan = oracle.plan(builder.build(desc))
                assert plan != [], name

                assert downward.validate(
                    domain,
                    desc,
                    oracle.plan_to_string(plan),
                    VALIDATE,
                ), name

        with subtests.test(name):
            assert not downward.validate(
                DOMAINS["blocksworld"],
                desc,
                oracle.plan_to_string(plan),
                VALIDATE,
            )


class TestUnsupportedDomain:
    """
    Test suite for unsupported domain.
    """

    def test_plan(self, mocker, blocksworld_fully_specified):
        """
        Test if the oracle can plan for an unsupported domain.
        """
        problem = builder.build(blocksworld_fully_specified)
        mocker.patch("planetarium.oracle.fully_specify", return_value=problem)
        with pytest.raises(oracle.DomainNotSupportedError):
            oracle.plan(problem, domain="unsupported_domain")
