![Tests](https://github.com/ghendrickx/ANNESI/actions/workflows/tests.yml/badge.svg)

# GAPool
Genetic algorithm with "best-pool" functionality. One of many `numpy`-based genetic algorithm implementations in 
`python`. However, this implementation includes the option to return not only the _person_ with the best _fitness_ -- as
is standard with genetic algorithms --, but also allows to return a _pool_ of _people_ whose _fitness_ is within a user-
defined deviation from the best _fitness_.

**Why the "best-pool" functionality?** This functionality is useful when the _fitness_-function is a computationally
cheap proxy function of the real optimisation problem. Using this proxy in combination with the "best pool" 
functionality allows to narrow-down the input space to be explored/investigated with the more computationally expensive
function (or model).

## Requirements
This repository only depends on `numpy` (see [`requirements.txt`](requirements.txt)).

## Usage
The basic usage of the genetic algorithm requires the initiation of the `GeneticAlgorithm`-class, which requires some 
definitions of the optimisation problem at hand:
 1. _fitness_-function
 1. dimensionality
 1. variable types (`{'bool', 'int', 'float'}`)
 1. variable boundaries

**Note** that the dimensionality should match the number of input arguments of the _fitness_-function as well as the
variable types and boundaries. In case all variables have the same type, a single definition suffices.

The initiation (including the definition of a _fitness_-function) looks as follows:
```python
from src.ga import GeneticAlgorithm

# define the fitness function


def fitness(x, y, z):
    """Dummy fitness function for a three-dimensional problem."""
    return sum([x, y, z])


# initiate the GA-class
model = GeneticAlgorithm(
    fitness,        # fitness function
    3,              # dimensionality
    'float',        # variable type (all float)
    [
        [0, 10],    # variable boundaries for `x`
        [3, 5],     # variable boundaries for `y`
        [0, 1]      # variable boundaries for `z`
    ] 
)
```

After the initiation, the genetic algorithm must be executed. To do so, the `exec()`-method has be called, which returns a
`tuple` with three items:
 1. Most _fit person_ and its _fitness_ (`dict`);
 1. Evolution of _fitness_ over the _generations_ (`dict`);
 1. Pool of _fittest people_ and their _fitness_ (`numpy.ndarray`, or `None`).

**Note** that the evolution of _fitness_ can contain other data of the _population_ during its _evolution_, based on the
optional argument provided by the user to the `exec()`-method.

**Note** that the pool of _fittest people_ equals `None` if the genetic algorithm is executed without its "best pool" 
functionality enabled (disabled by default).

The subsequent execution of the genetic algorithm looks as follows:
```python
from src.ga import GeneticAlgorithm

# define the fitness function


def fitness(x, y, z):
    """Dummy fitness function for a three-dimensional problem."""
    return sum([x, y, z])


# initiate the GA-class
model = GeneticAlgorithm(fitness, 3, 'float', [[0, 10], [3, 5], [0, 1]])

# execute without "best pool" functionality
output_without_pool = model.exec()

# execute with "best pool" functionality
output_with_pool = model.exec(output_pool=True)
```

The exact output varies per execution due to the random nature of the genetic algorithm. Nevertheless, the results in
`output_without_pool` will contain the following three items:
```
{'person': array([v01, v02, v03]), 'fitness': f0}
{'best_fitness': [..., ..., f0]}
None
```
The results in the `output_with_pool` will look the same, except the addition of the pool with the _fittest people_:
```
{'person': array([v01, v02, v03]), 'fitness': f0}
{'best_fitness': [..., ..., f0]}
[[v11, v12, v13, f1]
 [v21, v22, v23, f2]
 ...
 [vn1, vn2, vn3, fn]]
```
**Note** that the pool may (and probably will) contain duplicates.

**Note** that the shown example only addresses the basic usage. However, the genetic algorithm includes more options to 
fine-tune the results to the user's needs. Most of these are included in the `settings` of the genetic algorithm, which
can be modified by entering the settings' key(s) as optional argument(s):
```python
from src.ga import GeneticAlgorithm

model = GeneticAlgorithm(
    lambda *args: sum(args), 3, 'float', [[0, 10], [3, 5], [0, 1]],
    population_size=200,  # default: 100
    elite_ratio=0,  # default: 0.1
)
```
Other optional arguments can be provided to the `exec()`-method. These concern the "best pool" functionality, and 
_evolution_ progress track-record (i.e., what data to store throughout the _evolution_).

## Author
Gijs G. Hendrickx 
[![alt text](https://camo.githubusercontent.com/e1ec0e2167b22db46b0a5d60525c3e4a4f879590a04c370fef77e6a7e00eb234/68747470733a2f2f696e666f2e6f726369642e6f72672f77702d636f6e74656e742f75706c6f6164732f323031392f31312f6f726369645f31367831362e706e67) 0000-0001-9523-7657](https://orcid.org/0000-0001-9523-7657)
(_Delft University of Technology_).

Contact: [G.G.Hendrickx@tudelft.nl](mailto:G.G.Hendrickx@tudelft.nl?subject=[GitHub]%20ANNESI:%20).

## References
Please refer appropriately when using this repository. This repository is made as part of the research described in
[_Hendrickx et al._ (2023)](https://doi.org/10.1016/j.coastaleng.2023.104289), which may be used as linked reference for 
this repository.
>   [Hendrickx, G.G.](https://orcid.org/0000-0001-9523-7657),
    [Antol&iacute;nez, J.A.A.](https://orcid.org/0000-0002-0694-4817), and
    [Herman, P.M.J.](https://orcid.org/0000-0003-2188-6341)
    (2023).
    Predicting the response of complex systems for coastal management. 
    _Coastal Engineering_, **182**:104289.
    doi:[10.1016/j.coastaleng.2023.104289](https://doi.org/10.1016/j.coastaleng.2023.104289).

## License
This repository is licensed under [`Apache License 2.0`](LICENSE).

## Structure
The genetic algorithm is included in the [`src`](src)-directory:
```
+-- src
|   +-- __init__.py
|   +-- ga.py
+-- tests
|   +-- __init__.py
|   +-- test_ga.py
+-- .gitignore
+-- __init__.py
+-- LICENSE
+-- README.md
+-- requirements.txt
+-- setup.py
```
