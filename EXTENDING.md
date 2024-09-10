# Extending Planetarium

If you're looking to evaluate your own domain, here is a guide to help you get started.

### 1. Add a domain file
Add a domain PDDL file to the `planetarium/domains/` directory, where the filename is the name of your domain.

### 2. Add an Oracle

Every domain in Planetarium requires an [`Oracle`](https://github.com/BatsResearch/planetarium/blob/main/planetarium/oracles/oracle.py#L8) object and file tied to it.
There are three fundamental components to the Oracle:

- [`.reduce()`](https://github.com/BatsResearch/planetarium/blob/main/planetarium/oracles/oracle.py#L11) function, which takes a [`ProblemGraph`](https://github.com/BatsResearch/planetarium/blob/main/planetarium/graph.py#L430) or [`SceneGraph`](https://github.com/BatsResearch/planetarium/blob/main/planetarium/graph.py#L391) object and returns a [`ReducedProblemGraph`](https://github.com/BatsResearch/planetarium/blob/main/planetarium/reduced_graph.py#L80) or [`ReducedSceneGraph`](https://github.com/BatsResearch/planetarium/blob/main/planetarium/reduced_graph.py#L48) object, respectively.
- [`.inflate()`](https://github.com/BatsResearch/planetarium/blob/main/planetarium/oracles/oracle.py#L26) function, which takes a [`ReducedProblemGraph`](https://github.com/BatsResearch/planetarium/blob/main/planetarium/reduced_graph.py#L80) or [`ReducedSceneGraph`](https://github.com/BatsResearch/planetarium/blob/main/planetarium/reduced_graph.py#L48) object and returns a [`ProblemGraph`](https://github.com/BatsResearch/planetarium/blob/main/planetarium/graph.py#L430) or [`SceneGraph`](https://github.com/BatsResearch/planetarium/blob/main/planetarium/graph.py#L391) object, respectively.
- [`.fully_specify()`](https://github.com/BatsResearch/planetarium/blob/main/planetarium/oracles/oracle.py#L40) function, which takes a [`ProblemGraph`](https://github.com/BatsResearch/planetarium/blob/main/planetarium/graph.py#L430) and returns either a [`ProblemGraph`](https://github.com/BatsResearch/planetarium/blob/main/planetarium/graph.py#L430) or a [`ReducedProblemGraph`](https://github.com/BatsResearch/planetarium/blob/main/planetarium/reduced_graph.py#L80) with all possible predicates added to the goal scene without changing the original definition of the problem.
We refer to these predicates as "trivial predicates" in our paper.
The `fully_specify` function is used to ensure that the problem is fully specified before evaluation.

To add your domain, you must create a python script under `planetarium/oracles/` that contains your implementation of an Oracle subclass.
This script should contain a class that inherits from `Oracle` and implements the three functions described above.
While we provide a generic `reduce` and `inflate` function in the base `Oracle` class, you will definitely want to override these functions if your domain has any 0-ary, 1-ary, or (3+)-ary predicates (more info below).

#### 2.1 Remember to add your Oracle script to the `__init__.py` file in the `planetarium/oracles/` directory:
```python
# planetarium/oracles/__init__.py
__all__ = ["blocksworld", "gripper", "rover_single", "floortile", "YOUR_DOMAIN"]

from . import oracle

from . import blocksworld
from . import gripper
from . import rover_single
from . import floortile
from . import YOUR_DOMAIN

ORACLES: dict[str, oracle.Oracle] = {
    "blocksworld": blocksworld.BlocksworldOracle,
    "gripper": gripper.GripperOracle,
    "rover-single": rover_single.RoverSingleOracle,
    "floor-tile": floortile.FloorTileOracle,
    "YOUR_DOMAIN": YOUR_DOMAIN.YourDomainOracle,
}
```

#### 2.2 Working with non-binary predicates (Using `ReducedNode`s)
Some predicates require special care for reducing and inflating.
The key idea behind our reduced representation is to represent our PDDL problem in a domain-specific manner with as few graph nodes and edges as possible to reduce the search space for our equivalence check.
The reduced representation also allows us to perform higher-level graph manipulations and searches more efficiently.

**[`ReducedNode`](https://github.com/BatsResearch/planetarium/blob/main/planetarium/reduced_graph.py#L9)**:

A [`ReducedNode`](https://github.com/BatsResearch/planetarium/blob/main/planetarium/reduced_graph.py#L9) is a domain-specific set of nodes that will be added to every `ReducedSceneGraph` or `ReducedProblemGraph` object on construction. They help hold metadata for specific types of predicates (non-binary predicates, 0-ary predicates, etc.) that are defined by the domain.

Here is an example on how to reduce different -ary predicates using `ReducedNode`s:

**0-ary predicates**:
These predicates can be represented by using a `ReducedNode` with a `name` attribute that matches the predicate name.
If the predicate is true in the scene, one way to handle this is by adding a self-edge on this `ReducedNode` to represent the predicate.

**1-ary predicates**:
These predicates can be represented by using a `ReducedNode` with a `name` attribute that matches the predicate name.
If the predicate is true in the scene, we can add an edge from the `ReducedNode` to the node that represents the object in the scene.

**3+-ary predicates**:
There is no easy way to reduce these predicates, so the best way to keep track of these is to simply add a predicate node to the `ReducedSceneGraph` or `ReducedProblemGraph` object that represents the predicate, and add edges to the nodes that represent the objects in the scene.
Make sure to set the `position=` argument when adding the edge to the reduced graph to ensure you can reverse this action in your `inflate` function.

**To register your `ReducedNode`**:
At the top of your oracle script, you can call the [`ReducedNode.register`](https://github.com/BatsResearch/planetarium/blob/main/planetarium/reduced_graph.py#L12) method, like the following:

```python
ReducedNode.register(
    {
        # ReducedNodeName: corresponding predicate name
        "ROOMS": "room",
        "BALLS": "ball",
        "GRIPPERS": "gripper",
        "ROBBY": "at-robby",
        "FREE": "free",
    },
    "YOUR_DOMAIN", # name of your domain
)
```

This will let you use your `ReducedNode` like any other enum throughout your oracle script (e.g. `ReducedNode.ROOMS`).

#### 2.3 Implementing `.plan()` (Optional)
If you would like to evaluate whether or not a problem is _solvable_, you can implement the [`.plan()`](https://github.com/BatsResearch/planetarium/blob/main/planetarium/oracles/oracle.py#L56) function in your `Oracle`.
You will still be able to evaluate whether or not a problem is solvable without implementing this function, but it will rely on running the FastDownward planner to solve your problem, which may be *significantly* slower than using a domain-specific planner.
(Note that you should try using the `lama-first` alias if possible, as this planner does not look for the optimal plan, just a satisficing plan.)