import pytest

from .test_pddl import problem_string

from planetarium import builder


@pytest.fixture
def sgraph(problem_string):
    """
    Fixture providing an SGraph instance built from a PDDL problem string.
    """
    return builder.build(problem_string).decompose()[0]


class TestGraph:
    """
    Test suite for the SGraph instance built from a PDDL problem.
    """

    def test_constant_node_names(self, sgraph):
        """
        Test if the names of constant nodes in the graph match the expected set.
        """
        names = set(["p0", "p1", "f0", "f1", "f2", "f3"])
        assert all([(node.name in names) for node in sgraph.constant_nodes])

    def test_constant_node_size(self, sgraph):
        """
        Test if the number of constant nodes in the graph matches the expected count.
        """
        assert len(sgraph.constant_nodes) == 6

    def test_predicate_names(self, sgraph):
        """
        Test if the names of predicate nodes in the graph match expected patterns.
        """
        for predicate in sgraph.predicate_nodes:
            match predicate.node.split("-"):
                case ["above", _, _]:
                    assert True
                case ["origin", _, _]:
                    assert True
                case ["destin", _, _]:
                    assert True
                case ["lift", "at", _]:
                    assert True
                case _:
                    assert False
