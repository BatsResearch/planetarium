import pytest

from planetarium import builder, graph, metric, oracle

# pylint: disable=unused-import
from .test_pddl import (
    problem_string,
    two_initial_problem_string,
    renamed_problem_string,
    wrong_problem_string,
    swap_problem_string,
    wrong_swap_problem_string,
    move_problem_string,
    wrong_move_problem_string,
    wrong_initial_problem_string,
)

from .problem_fixtures import (
    blocksworld_underspecified,
    blocksworld_missing_clears,
    blocksworld_missing_ontables,
    blocksworld_fully_specified,
    gripper_fully_specified,
    gripper_no_robby,
    rover_single_line_fully_specified_4,
    rover_single_line_fully_specified_4a,
)

from pddl import requirements


def problem_states(
    problem_init: graph.SceneGraph, problem_goals: set[graph.SceneGraph]
) -> set[graph.ProblemGraph]:
    return set([graph.ProblemGraph.join(problem_init, goal) for goal in problem_goals])


class TestConstantMatching:
    """
    Test suite for constant matching functions in the metric module.
    """

    @pytest.fixture
    def source(self):
        """Fixture for a valid source constant."""
        return graph.PlanGraphNode(
            "o1", "o1", typing=["t1", "t2"], label=graph.Label.CONSTANT
        )

    @pytest.fixture
    def target(self):
        """Fixture for a valid target constant."""
        return graph.PlanGraphNode(
            "c1", "c1", typing=["t1", "t2"], label=graph.Label.CONSTANT
        )

    @pytest.fixture
    def source_incorrect_label(self):
        """Fixture for a source constant with an incorrect label."""
        return graph.PlanGraphNode(
            "o1", "o1", typing=["t1", "t2"], label=graph.Label.PREDICATE
        )

    @pytest.fixture
    def target_incorrect_label(self):
        """Fixture for a target constant with an incorrect label."""
        return graph.PlanGraphNode(
            "c1", "c1", typing=["t1", "t2"], label=graph.Label.PREDICATE
        )

    @pytest.fixture
    def source_incorrect_typing(self):
        """Fixture for a source constant with incorrect typing."""
        return graph.PlanGraphNode(
            "o1", "o1", typing=["ty1", "ty2"], label=graph.Label.CONSTANT
        )

    @pytest.fixture
    def target_incorrect_typing(self):
        """Fixture for a target constant with incorrect typing."""
        return graph.PlanGraphNode(
            "c1", "c1", typing=["ty1", "ty2"], label=graph.Label.CONSTANT
        )

    @pytest.fixture
    def mapping(self):
        """Fixture for a valid mapping between source and target constants."""
        return {"o1": "c1"}

    @pytest.fixture
    def mapping_incorrect(self):
        """Fixture for an incorrect mapping between source and target constants."""
        return {"o1": "o1"}

    def test_correct_matching(self, source, target, mapping):
        """Test correct matching between source and target constants."""
        assert metric._node_matching(source, target, None)
        assert metric._node_matching(source, target, mapping)

        assert metric._same_typing(source, target)
        assert metric._preserves_mapping(source, target, mapping)

    def test_incorrect_label(
        self, source, target, source_incorrect_label, target_incorrect_label, mapping
    ):
        """Test incorrect label matching between source and target constants."""
        assert not metric._node_matching(source, target_incorrect_label, None)
        assert not metric._node_matching(source_incorrect_label, target, None)
        assert not metric._node_matching(source, target_incorrect_label, mapping)
        assert not metric._node_matching(source_incorrect_label, target, mapping)

        assert not metric._preserves_mapping(source, target_incorrect_label, mapping)
        assert not metric._preserves_mapping(source_incorrect_label, target, mapping)

        assert not metric._same_typing(source, target_incorrect_label)
        assert not metric._same_typing(source_incorrect_label, target)

    def test_incorrect_typing(
        self, source, target, source_incorrect_typing, target_incorrect_typing, mapping
    ):
        """Test incorrect typing between source and target constants."""
        assert not metric._node_matching(source, target_incorrect_typing, None)
        assert not metric._node_matching(source_incorrect_typing, target, None)
        assert not metric._node_matching(source, target_incorrect_typing, mapping)
        assert not metric._node_matching(source_incorrect_typing, target, mapping)

        assert metric._preserves_mapping(source, target_incorrect_typing, mapping)
        assert metric._preserves_mapping(source_incorrect_typing, target, mapping)

        assert not metric._same_typing(source, target_incorrect_typing)
        assert not metric._same_typing(source_incorrect_typing, target)

    def test_incorrect_mapping(self, source, target, mapping_incorrect):
        """Test incorrect mapping between source and target constants."""
        assert not metric._node_matching(source, target, mapping_incorrect)
        assert not metric._preserves_mapping(source, target, mapping_incorrect)


class TestPredicateMatching:
    """
    Test suite for predicate matching functions in the metric module.
    """

    @pytest.fixture
    def source(self):
        """Fixture for a valid source predicate node."""
        return graph.PlanGraphNode(
            "f-a1-a2",
            "f-a1-a2",
            typing="f",
            label=graph.Label.PREDICATE,
        )

    @pytest.fixture
    def target(self):
        """Fixture for a valid target predicate node."""
        return graph.PlanGraphNode(
            "f-a1-a2",
            "f-a1-a2",
            typing="f",
            label=graph.Label.PREDICATE,
        )

    @pytest.fixture
    def source_incorrect_label(self):
        """Fixture for a source predicate node with an incorrect label."""
        return graph.PlanGraphNode(
            "f-a1-a2",
            "f-a1-a2",
            typing="f",
            label=graph.Label.CONSTANT,
        )

    @pytest.fixture
    def target_incorrect_label(self):
        """Fixture for a target predicate node with an incorrect label."""
        return graph.PlanGraphNode(
            "f-a1-a2",
            "f-a1-a2",
            typing="f",
            label=graph.Label.CONSTANT,
        )

    def test_correct_matching(self, source, target):
        """Test correct matching between source and target predicate nodes."""
        assert metric._node_matching(source, target, None)

    def test_incorrect_label(
        self,
        source,
        target,
        source_incorrect_label,
        target_incorrect_label,
    ):
        """Test incorrect label matching between source and target predicate nodes."""
        assert not metric._node_matching(source, target_incorrect_label, None)
        assert not metric._node_matching(source_incorrect_label, target, None)


class TestMetrics:
    """
    Test suite for metrics functions in the metric module.
    """

    def test_map(self, problem_string, two_initial_problem_string):
        """Test the mapping function on graph pairs."""
        problem_graph = builder.build(problem_string)
        problem_graph2 = builder.build(two_initial_problem_string)

        assert metric.isomorphic(problem_graph, problem_graph)
        assert not metric.isomorphic(problem_graph, problem_graph2)

    def test_validate(self, problem_string, two_initial_problem_string):
        """Test the validation function on graph pairs."""
        problem_graph = builder.build(problem_string)
        problem_graph2 = builder.build(two_initial_problem_string)

        assert metric.equals(problem_graph, problem_graph, is_placeholder=True)
        assert not metric.equals(
            problem_graph,
            problem_graph2,
            is_placeholder=True,
        )

    def test_swap(self, swap_problem_string, wrong_swap_problem_string):
        """
        Test the distance function on graph pairs.
        """
        swap_problem = builder.build(swap_problem_string)
        wrong_swap = builder.build(wrong_swap_problem_string)

        # Test validate
        assert metric.equals(swap_problem, swap_problem, is_placeholder=False)
        assert not metric.equals(swap_problem, wrong_swap, is_placeholder=False)
        assert metric.equals(swap_problem, wrong_swap, is_placeholder=True)

    def test_move(self, move_problem_string, wrong_move_problem_string):
        """
        Test the distance function on graph pairs.
        """
        move_problem = builder.build(move_problem_string)
        wrong_move = builder.build(wrong_move_problem_string)

        # Test validate
        assert metric.equals(move_problem, move_problem, is_placeholder=True)
        assert not metric.equals(move_problem, wrong_move, is_placeholder=True)

    def test_blocksworld_equivalence(
        self,
        subtests,
        blocksworld_fully_specified,
        blocksworld_missing_clears,
        blocksworld_missing_ontables,
        blocksworld_underspecified,
    ):
        """Test the equivalence of blocksworld problems."""
        p1 = builder.build(blocksworld_fully_specified)
        p2 = builder.build(blocksworld_missing_clears)
        p3 = builder.build(blocksworld_missing_ontables)
        p4 = builder.build(blocksworld_underspecified)

        p1 = oracle.fully_specify(p1)
        p2 = oracle.fully_specify(p2)
        p3 = oracle.fully_specify(p3)
        p4 = oracle.fully_specify(p4)

        P = (
            ("blocksworld_fully_specified", p1),
            ("blocksworld_missing_clears", p2),
            ("blocksworld_missing_ontables", p3),
            ("blocksworld_underspecified", p4),
        )

        # equivalence to itself
        for name, p in P:
            with subtests.test(f"{name} equals {name}"):
                assert metric.equals(p, p, is_placeholder=True)
                assert metric.equals(p, p, is_placeholder=False)

        # check invalid equivalence

        for idx1, idx2 in (
            (0, 3),
            (1, 3),
            (2, 3),
        ):
            (name1, p1), (name2, p2) = P[idx1], P[idx2]
            with subtests.test(f"{name1} not equals {name2}"):
                assert not metric.equals(p1, p2, is_placeholder=True)
                assert not metric.equals(p1, p2, is_placeholder=False)
                assert not metric.equals(p2, p1, is_placeholder=True)
                assert not metric.equals(p2, p1, is_placeholder=False)

    def test_rover_single_eqquivalence(
        self,
        subtests,
        rover_single_line_fully_specified_4,
        rover_single_line_fully_specified_4a,
    ):
        """Test the equivalence of rover single line problems."""
        p1 = builder.build(rover_single_line_fully_specified_4)
        p2 = builder.build(rover_single_line_fully_specified_4a)

        p1 = oracle.fully_specify(p1)
        p2 = oracle.fully_specify(p2)

        # equivalence to itself
        assert metric.equals(p1, p1, is_placeholder=True)
        assert metric.equals(p2, p2, is_placeholder=False)

        # check invalid equivalence
        assert metric.equals(p1, p2, is_placeholder=True)
        assert metric.equals(p1, p2, is_placeholder=False)
        assert metric.equals(p2, p1, is_placeholder=True)
        assert metric.equals(p2, p1, is_placeholder=False)
