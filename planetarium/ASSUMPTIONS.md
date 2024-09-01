# Assumptions

For each domain, the following assumptions are made regarding the types of problems evaluated.
We also assume that problems are solvable via the domain's actions.

## Blocksworld

Generally, since Blocks World has reversible actions, essentially all problems are evaluable.

`:init` conditions:
- No two blocks are on top of a single block
- No block is initialized on top of two blocks
- No loops of blocks are made
- Arm can only hold one block

## Grippers

Generally, since Grippers has reversible actions, essentially all problems are evaluable.

`:init` conditions:
- No double "typing" of objects (we don't using `:typing`, we use certain immutable predicates)
- All balls have only one location
- All grippers have only one location
- All grippers hold up to 1 ball

## Rover
Rover has the capability of being a much more complex domain, but for the purposes of this benchmark, we assume all problems will have symmetry of traversal.
This changes our `fullySpecify` function from essentially a multi-agent traveling salesman problem into simple pathfinding between two points.
Our goal is not to be *complete*, but to be *correct* for the problems we do support evaluating.

`:init` conditions:
- All objects have 1 type
- All rovers only have one location
- All landers only have one location
- If `can_traverse ?rover ?x ?y`, then `can_traverse ?rover ?y ?x`

**Note**: For the special case of up to 1 rover and up to 1 sample per analysis (rock, soil, imaging), we do not require symmmetry.