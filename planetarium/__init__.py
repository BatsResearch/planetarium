import os
from importlib import resources

__all__ = [
    "builder",
    "downward",
    "graph",
    "metric",
    "oracle",
    "evaluate",
    "DOMAINS",
]

from . import builder
from . import downward
from . import graph
from . import metric
from . import oracle
from . import domains

DOMAINS = dict()

# load domains
for domain in resources.files(domains).iterdir():
    with domain.open() as f:
        DOMAINS[os.path.basename(domain).split(".")[0]] = f.read()

from .evaluate import evaluate
