import pytest

from planetarium import builder

from pddl.parser.problem import LenientProblemParser


@pytest.fixture
def problem_string():
    """
    Fixture providing a sample PDDL problem definition as a string.
    """
    return """
        (define (problem mixed-f4-p2-u0-v0-g0-a0-n0-A0-B0-N0-F0-r0)
          (:domain miconic)
          (:objects p0 p1 - passenger
                    f0 f1 f2 f3 - floor)

          (:init
            (above f0 f1)
            (above f0 f2)
            (above f0 f3)
            (above f1 f2)
            (above f1 f3)
            (above f2 f3)
            (origin p0 f3)
            (destin p0 f2)
            (origin p1 f1)
            (destin p1 f3)
            (lift-at f0))

          (:goal (and
            (served p0)
            (served p1)))
        )
    """


@pytest.fixture
def two_initial_problem_string():
    """
    Fixture providing a sample PDDL problem definition as a string.
    """
    return """
        (define (problem mixed-f4-p2-u0-v0-g0-a0-n0-A0-B0-N0-F0-r0)
          (:domain miconic)
          (:objects p0 p1 - passenger
                    f0 f1 f2 f3 - floor)

          (:init
            (above f0 f1)
            (above f0 f2)
            (above f0 f3)
            (above f1 f2)
            (above f1 f3)
            (above f2 f3)
            (origin p0 f3)
            (destin p0 f2)
            (origin p1 f1)
            (destin p1 f3)
            (lift-at f0))

          (:goal (and
            (above f0 f1)
            (above f0 f2)
            (above f0 f3)
            (above f1 f2)
            (above f1 f3)
            (above f2 f3)
            (origin p0 f3)
            (destin p0 f2)
            (origin p1 f1)
            (destin p1 f3)
            (lift-at f0)))
        )
    """


@pytest.fixture
def renamed_problem_string():
    """
    Fixture providing a sample PDDL problem definition as a string.
    """
    return """
        (define (problem mixed-f4-p2-u0-v0-g0-a0-n0-A0-B0-N0-F0-r0)
          (:domain miconic)
          (:objects p1 p0 - passenger
                    f1 f2 f3 f0 - floor)

          (:init
            (above f1 f2)
            (above f1 f3)
            (above f1 f0)
            (above f2 f3)
            (above f2 f0)
            (above f3 f0)
            (origin p1 f0)
            (destin p1 f3)
            (origin p0 f2)
            (destin p0 f0)
            (lift-at f1))

          (:goal (and
            (served p1)
            (served p0)))
        )
    """


@pytest.fixture
def wrong_problem_string():
    """
    Fixture providing a sample PDDL problem definition as a string.
    """
    return """
        (define (problem mixed-f4-p2-u0-v0-g0-a0-n0-A0-B0-N0-F0-r0)
          (:domain miconic)
          (:objects p1 p0 - passenger
                    f1 f2 f3 f0 - floor)

          (:init
            (above f1 f2)
            (above f1 f3)
            (above f1 f0)
            (above f2 f3)
            (above f2 f0)
            (above f3 f0)
            (origin p1 f0)
            (destin p1 f3)
            (origin p0 f2)
            (destin p0 f0)
            (lift-at f1))

          (:goal (and
            (served f3)
            (served p0)))
        )
    """


@pytest.fixture
def wrong_initial_problem_string():
    """
    Fixture providing a sample PDDL problem definition as a string.
    """
    return """
        (define (problem mixed-f4-p2-u0-v0-g0-a0-n0-A0-B0-N0-F0-r0)
          (:domain miconic)
          (:objects p1 p0 - passenger
                    f1 f2 f3 f0 - floor)

          (:init
            (above f1 f2)
            (above f1 f3)
            (above f1 f0)
            (above f2 f3)
            (above f2 f0)
            (above f3 f0)
            (destin p1 f3)
            (origin p0 f2)
            (destin p0 f0)
            (lift-at f1))

          (:goal (and
            (served p1)
            (served p0)))
        )
    """


@pytest.fixture
def swap_problem_string():
    """
    Fixture providing a sample PDDL problem definition as a string.
    """
    return """
        (define (problem swap)
          (:domain swap)
          (:objects a0 a1 - object
                    b0 b1 - room)

          (:init
            (in a0 b0)
            (in a1 b1))

          (:goal (and
            (in a0 b1)
            (in a1 b0)))
        )
    """


@pytest.fixture
def wrong_swap_problem_string():
    """
    Fixture providing a sample PDDL problem definition as a string.
    """
    return """
        (define (problem swap)
          (:domain swap)
          (:objects a0 a1 - object
                    b0 b1 - room)

          (:init
            (in a0 b0)
            (in a1 b1))

          (:goal (and
            (in a0 b0)
            (in a1 b1)))
        )
    """


@pytest.fixture
def move_problem_string():
    """
    Fixture providing a sample PDDL problem definition as a string.
    """
    return """
        (define (problem move)
          (:domain move)
          (:objects a0 a1 - object
                    b0 b1 - room)

          (:init
            (in a0 b0)
            (in a1 b0))

          (:goal (and
            (in a0 b1)
            (in a1 b0)))
        )
    """


@pytest.fixture
def wrong_move_problem_string():
    """
    Fixture providing a sample PDDL problem definition as a string.
    """
    return """
        (define (problem move)
          (:domain move)
          (:objects a0 a1 - object
                    b0 b1 - room)

          (:init
            (in a0 b0)
            (in a1 b1))

          (:goal (and
            (in a0 b0)
            (in a1 b1)))
        )
    """


@pytest.fixture
def single_predicate_goal():
    """
    Fixture providing a sample PDDL problem definition as a string.
    """
    return """
        (define (problem move)
          (:domain move)
          (:objects a0 a1 - object
                    b0 b1 - room)

          (:init
            (in a0 b0)
            (in a1 b1))

          (:goal (in a0 b0))
        )
    """


@pytest.fixture
def not_predicate_goal():
    """
    Fixture providing a sample PDDL problem definition as a string.
    """
    return """
        (define (problem move)
          (:domain move)
          (:objects a0 a1 - object
                    b0 b1 - room)

          (:init
            (in a0 b0)
            (in a1 b1))

          (:goal (not
            (in a0 b0)))
        )
    """


@pytest.fixture
def problem(problem_string):
    """
    Fixture providing a parsed PDDL problem object.
    """
    return LenientProblemParser()(problem_string)


class TestConstantToDict:
    """
    Test suite for the _constant_to_dict function.
    """

    def test_constat_name(self, problem):
        """
        Test the conversion of a PDDL Constant to a dictionary with the correct name.
        """
        constant = list(problem.objects)[0]
        assert builder._constant_to_dict(constant)["name"] == str(constant.name)

    def test_constat_type(self, problem):
        """
        Test the conversion of a PDDL Constant to a dictionary with the correct typing.
        """
        constant = list(problem.objects)[0]
        result_dict = builder._constant_to_dict(constant)
        assert (
            result_dict["typing"] == constant.type_tags
            and type(result_dict["typing"]) == set
        )


class TestPredicateToDict:
    """
    Test suite for the _predicate_to_dict function.
    """

    def test_predicate_name(self, problem):
        """
        Test the conversion of a PDDL Predicate to a dictionary with the correct name.
        """
        predicate = list(problem.init)[0]
        assert builder._predicate_to_dict(predicate)["typing"] == str(predicate.name)

    def test_predicate_parameters(self, problem):
        """
        Test the conversion of a PDDL Predicate to a dictionary with the correct parameters.
        """
        predicate = list(problem.init)[0]
        result_dict = builder._predicate_to_dict(predicate)
        assert (
            result_dict["parameters"] == [term.name for term in predicate.terms]
            and type(result_dict["parameters"]) == list
        )


class TestBuildConstants:
    """
    Test suite for the _build_constants function.
    """

    def test_size(self, problem):
        """
        Test the size of the list of constants built from a PDDL problem.
        """
        assert len(builder._build_constants(problem.objects)) == len(problem.objects)


class TestBuildPredicates:
    """
    Test suite for the _build_predicates function.
    """

    def test_initial_size(self, problem):
        """
        Test the size of the list of initial predicates built from a PDDL problem.
        """
        assert len(builder._build_predicates(problem.init)) == len(problem.init)

    def test_goal_size(self, problem):
        """
        Test the size of the list of goal predicates built from a PDDL problem.
        """
        assert len(builder._build_predicates(problem.goal.operands)) == len(
            problem.goal.operands
        )


class TestBuild:
    """
    Test suite for the build function.
    """

    def test_node_size(self, problem_string):
        """
        Test the size of nodes in the scene graphs built from a PDDL problem.
        """
        graph_1, graph_2 = builder.build(problem_string).decompose()
        assert len(graph_1.nodes) == 17 and len(graph_2.nodes) == 8

    def test_edge_size(self, problem_string):
        """
        Test the size of edges in the scene graphs built from a PDDL problem.
        """
        graph_1, graph_2 = builder.build(problem_string).decompose()
        assert len(graph_1.edges) == 21 and len(graph_2.edges) == 2

    def test_edge_size(self, problem_string):
        """
        Test the size of edges in the scene graphs built from a PDDL problem.
        """
        modified_problem_string = f"Here is an example of a problem string that is not a PDDL problem. ```pddl\n{problem_string}\n```"
        graph_1, graph_2 = builder.build(modified_problem_string).decompose()
        assert len(graph_1.edges) == 21 and len(graph_2.edges) == 2

    def test_single_predicate_goal(self, single_predicate_goal):
        """
        Test the size of nodes in the scene graphs built from a PDDL problem.
        """
        builder.build(single_predicate_goal).decompose()


    def test_to_pddl_str(self, single_predicate_goal):
        """
        Test the size of nodes in the scene graphs built from a PDDL problem.
        """
        builder.build(single_predicate_goal).to_pddl_str()

    def test_not_predicate_goal(self, not_predicate_goal):
        """
        Test the size of nodes in the scene graphs built from a PDDL problem.
        """
        with pytest.raises(ValueError):
            builder.build(not_predicate_goal).decompose()
