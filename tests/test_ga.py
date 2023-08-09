"""
Tests for genetic algorithm.

Author: Gijs G. Hendrickx
"""
import numpy as np
import pytest

from src.ga import GeneticAlgorithm


"""fitness function"""


def fitness(*args) -> float:
    """Dummy fitness-function."""
    return sum(args)


"""pytest.fixtures"""


@pytest.fixture
def ga_pool() -> GeneticAlgorithm:
    dim = 10
    return GeneticAlgorithm(fitness, dim, 'float', [[0, 1]] * dim)


class TestGeneticAlgorithm:
    """Tests for the `GeneticAlgorithm`-object (from `src.ga`)."""

    """Initiation"""

    def test_init_all_bool(self):
        ga = GeneticAlgorithm(fitness, 2, 'bool', [[0, 1], [0, 1]])
        assert all(vt == 'bool' for vt in ga.var_types)
        for vb in ga.var_bounds:
            assert all(isinstance(v, int) for v in vb)

    def test_init_all_int(self):
        ga = GeneticAlgorithm(fitness, 2, 'int', [[0, 1], [1, 2]])
        assert all(vt == 'int' for vt in ga.var_types)
        for vb in ga.var_bounds:
            assert all(isinstance(v, int) for v in vb)

    def test_init_all_float(self):
        ga = GeneticAlgorithm(fitness, 2, 'float', [[0, 1], [1e-10, 1]])
        assert all(vt == 'float' for vt in ga.var_types)
        for vb in ga.var_bounds:
            assert all(isinstance(v, float) for v in vb)

    def test_init_mixed_bi(self):
        ga = GeneticAlgorithm(fitness, 2, ('bool', 'int'), [[0, 1], [1, 2]])
        assert ga.var_types == ['bool', 'int']
        assert all(isinstance(v, int) for v in ga.var_bounds[0])
        assert all(isinstance(v, int) for v in ga.var_bounds[1])

    def test_init_mixed_bf(self):
        ga = GeneticAlgorithm(fitness, 2, ('bool', 'float'), [[0, 1], [1, 2]])
        assert ga.var_types == ['bool', 'float']
        assert all(isinstance(v, int) for v in ga.var_bounds[0])
        assert all(isinstance(v, float) for v in ga.var_bounds[1])

    def test_init_mixed_if(self):
        ga = GeneticAlgorithm(fitness, 2, ('int', 'float'), [[0, 1], [1, 2]])
        assert ga.var_types == ['int', 'float']
        assert all(isinstance(v, int) for v in ga.var_bounds[0])
        assert all(isinstance(v, float) for v in ga.var_bounds[1])

    """Genetic operations: Crossover"""

    def test_crossover_index(self, ga_pool):
        # set crossover-type
        ga_pool.c_type = 'index'
        # generate parents
        p1 = np.zeros(ga_pool.dim)
        p2 = np.ones(ga_pool.dim)
        # generate children
        c1, c2 = ga_pool.apply_crossover(p1, p2)
        # equal crossover-index
        i1 = next((i for i, v in enumerate(c1) if v == 1), -1)
        i2 = next((i for i, v in enumerate(c2) if v == 0), -1)
        assert i1 == i2

    def test_crossover_slice(self, ga_pool):
        # set crossover-type
        ga_pool.c_type = 'slice'
        # generate parents
        p1 = np.zeros(ga_pool.dim)
        p2 = np.ones(ga_pool.dim)
        # generate children
        c1, c2 = ga_pool.apply_crossover(p1, p2)
        # equal crossover-index: start of slice
        i1 = next((i for i, v in enumerate(c1) if v == 1), -1)
        i2 = next((i for i, v in enumerate(c2) if v == 0), -1)
        assert i1 == i2
        # equal crossover-index: end of slice
        j1 = next((j for j, v in enumerate(reversed(c1)) if v == 1), -1)
        j2 = next((j for j, v in enumerate(reversed(c2)) if v == 0), -1)
        assert j1 == j2

    def test_crossover_uniform(self, ga_pool):
        # set crossover-type
        ga_pool.c_type = 'uniform'
        # generate parents
        p1 = np.zeros(ga_pool.dim)
        p2 = np.ones(ga_pool.dim)
        # generate children
        c1, c2 = ga_pool.apply_crossover(p1, p2)
        # equal crossover-indices: switching indices
        b1 = (c1 == p1)
        b2 = (c2 == p2)
        assert np.all(b1 == b2)

    def test_crossover_unknown(self, ga_pool):
        # set crossover-type
        ga_pool.c_type = 'unknown'
        # generate parents
        p1 = np.zeros(ga_pool.dim)
        p2 = np.ones(ga_pool.dim)
        # generate children: ValueError
        with pytest.raises(ValueError):
            ga_pool.apply_crossover(p1, p2)

    """Genetic operations: Mutation"""

    def test_mutation_bool(self):
        mut = GeneticAlgorithm.apply_mutation(np.array([0, 1]), 'bool')
        assert 0 <= mut <= 1
        assert isinstance(mut, int)

    def test_mutation_int(self):
        mut = GeneticAlgorithm.apply_mutation(np.array([0, 10]), 'int')
        assert 0 <= mut <= 10
        assert isinstance(mut, int)

    def test_mutation_float(self):
        mut = GeneticAlgorithm.apply_mutation(np.array([0, 10]), 'float')
        assert 0 <= mut <= 10
        assert isinstance(mut, float)
