"""
Tests for genetic algorithm.

Author: Gijs G. Hendrickx
"""
import numpy as np
# noinspection PyPackageRequirements
import pytest

from src.ga import GeneticAlgorithm

"""fitness function"""


def fitness(*args) -> float:
    """Dummy fitness-function."""
    return sum(args)


"""pytest.fixtures"""


@pytest.fixture
def ga_pool() -> GeneticAlgorithm:
    return GeneticAlgorithm(fitness, 3, 'float', [[0, 10], [3, 5], [0, 1]])


"""TestObjects"""


class TestGeneticAlgorithm:
    """Tests for the `GeneticAlgorithm`-object (from `src.ga`)."""

    @staticmethod
    def teardown_method():
        """Reset default settings for `GeneticAlgorithm`."""
        GeneticAlgorithm._settings = {
            'crossover_probability': .5,
            'crossover_type': 'uniform',
            'elite_ratio': .1,
            'exploration_ratio': .5,
            'max_no_improve': None,
            'mutation_probability': .1,
            'population_size': 100,
            'replicate_ratio': .1,
        }

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

    """Errors: Initiation"""

    def test_error_non_callable_function(self):
        f = 1
        with pytest.raises(TypeError):
            GeneticAlgorithm(f, 2, 'int', [[0, 1], [1, 2]])

    def test_error_dim_var_types(self):
        with pytest.raises(ValueError):
            GeneticAlgorithm(fitness, 2, ('int', 'int', 'float'), [[0, 1], [1, 2]])

    def test_error_dim_var_bounds(self):
        with pytest.raises(ValueError):
            GeneticAlgorithm(fitness, 2, 'int', [[0, 1], [1, 2], [2, 3]])

    def test_error_invalid_var_type(self):
        with pytest.raises(ValueError):
            GeneticAlgorithm(fitness, 2, 'dict', [[0, 1], [1, 2]])

    """Errors: Settings"""

    def test_error_invalid_crossover_probability(self):
        # invalid probabilities
        invalid_probabilities = -.1, 1.1
        for p in invalid_probabilities:
            with pytest.raises(ValueError):
                GeneticAlgorithm(fitness, 2, 'int', [[0, 1], [1, 2]], crossover_probability=p)

        # valid probabilities (edge-cases)
        valid_probabilities = 0, 1
        for p in valid_probabilities:
            GeneticAlgorithm(fitness, 2, 'int', [[0, 1], [1, 2]], crossover_probability=p)

    def test_error_invalid_mutation_probability(self):
        # invalid probabilities
        invalid_probabilities = -.1, 1.1
        for p in invalid_probabilities:
            with pytest.raises(ValueError):
                GeneticAlgorithm(fitness, 2, 'int', [[0, 1], [1, 2]], mutation_probability=p)

        # valid probabilities (edge-cases)
        valid_probabilities = 0, 1
        for p in valid_probabilities:
            GeneticAlgorithm(fitness, 2, 'int', [[0, 1], [1, 2]], mutation_probability=p)

    def test_error_invalid_elite_ratio(self):
        # invalid ratios
        invalid_ratios = -.1, 1.1
        for r in invalid_ratios:
            with pytest.raises(ValueError):
                GeneticAlgorithm(fitness, 2, 'int', [[0, 1], [1, 2]], elite_ratio=r)

        # valid ratios (edge-cases)
        valid_ratios = 0, .4
        for r in valid_ratios:
            GeneticAlgorithm(fitness, 2, 'int', [[0, 1], [1, 2]], elite_ratio=r)

    def test_error_invalid_replicate_ratio(self):
        # invalid ratios
        invalid_ratios = -.1, 1.1
        for r in invalid_ratios:
            with pytest.raises(ValueError):
                GeneticAlgorithm(fitness, 2, 'int', [[0, 1], [1, 2]], replicate_ratio=r)

        # valid ratios (edge-cases)
        valid_ratios = 0, .4
        for r in valid_ratios:
            GeneticAlgorithm(fitness, 2, 'int', [[0, 1], [1, 2]], replicate_ratio=r)

    def test_error_invalid_exploration_ratio(self):
        # invalid ratios
        invalid_ratios = -.1, 1.1
        for r in invalid_ratios:
            with pytest.raises(ValueError):
                GeneticAlgorithm(fitness, 2, 'int', [[0, 1], [1, 2]], exploration_ratio=r)

        # valid ratios (edge-cases)
        valid_ratios = 0, .8
        for r in valid_ratios:
            GeneticAlgorithm(fitness, 2, 'int', [[0, 1], [1, 2]], exploration_ratio=r)

    def test_error_too_large_ratios(self):
        # invalid ratio-combination
        invalid_ratios = dict(elite_ratio=.4, replicate_ratio=.4, exploration_ratio=.4)
        with pytest.raises(ValueError):
            GeneticAlgorithm(fitness, 2, 'int', [[0, 1], [1, 2]], **invalid_ratios)

        # valid ratio-combinations
        valid_ratios = dict(elite_ratio=.3, replicate_ratio=.3, exploration_ratio=.4), \
            dict(elite_ratio=0, replicate_ratio=.1, exploration_ratio=.8)
        for rc in valid_ratios:
            GeneticAlgorithm(fitness, 2, 'int', [[0, 1], [1, 2]], **rc)

    def test_error_invalid_crossover_type(self):
        # invalid types
        invalid_types = None, 'unknown', 'mistery', 'probabilistic', 'etc.'
        for t in invalid_types:
            with pytest.raises(ValueError):
                GeneticAlgorithm(fitness, 2, 'int', [[0, 1], [1, 2]], crossover_type=t)

        # valid types
        valid_types = 'index', 'slice', 'uniform'
        for t in valid_types:
            GeneticAlgorithm(fitness, 2, 'int', [[0, 1], [1, 2]], crossover_type=t)

    """Errors: Execution"""

    def test_error_invalid_output_pool_deficit(self, ga_pool):
        # invalid deficits
        invalid_deficits = -.1, 0, 1, 1.1
        for d in invalid_deficits:
            with pytest.raises(ValueError):
                ga_pool.exec(output_pool_deficit=d)

        # valid deficits
        valid_deficits = 1e-9, .1, .9, (1 - 1e-9)
        for d in valid_deficits:
            ga_pool.exec(output_pool_deficit=d)

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
