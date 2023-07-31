"""
Genetic algorithm with best-pool functionality.

Author: Gijs G. Hendrickx
"""
import logging
import os
import sys
import time

import numpy as np
import typing

_LOG = logging.getLogger(__name__)


class GeneticAlgorithm:
    """A genetic algorithm that searches for the minimum of a fitness function."""
    _settings: dict = {
        'population_size': 100,
        'crossover_probability': .5,
        'mutation_probability': .1,
        'elite_ratio': .1,
        'replicate_ratio': .1,
        'parent_ratio': .4,
        'exploration_ratio': .5,
        'crossover_type': 'uniform',
        'max_no_improve': None,
    }

    def __init__(
            self, function: callable, dimension: int,
            var_types: typing.Union[str, typing.Collection] = 'bool',
            var_boundaries: typing.Collection = None,
            n_iterations: int = None,
            **kwargs
    ) -> None:
        """
        :param function: fitness function
        :param dimension: input dimension
        :param var_types: variable type(s), defaults to bool
        :param var_boundaries: variable boundaries, defaults to None
        :param n_iterations: maximum number of iterations, defaults to None
        :param kwargs: genetic algorithm settings:
            :param population_size: population size, defaults to 100
            :param crossover_probability: probability of crossover, defaults to 0.5
            :param mutation_probability: probability of mutation, defaults to 0.1
            :param elite_ratio: ratio of population subjected to elitism, defaults to 0.1
            :param replicate_ratio: ratio of population subjected to replication, defaults to 0.1
            :param parent_ratio: ratio of population used as parents for the new generation, defaults to 0.4
            :param exploration_ratio: ratio of children focused on exploration (i.e. mutation) over exploitation
                (i.e. crossover), defaults to 0.5
            :param crossover_type: type of crossover method, defaults to 'uniform'
            :param max_no_improve: maximum number of iterations without improvement, defaults to None
            :param function_timeout: maximum execution time of fitness function (in seconds), defaults to 10

        :type function: callable
        :type dimension: int
        :type var_types: type, iterable, optional
        :type var_boundaries: iterable, optional
        :type n_iterations: int, optional
        :type kwargs: optional
            :type population_size: int
            :type crossover_probability: float
            :type mutation_probability: float
            :type elite_ratio: float
            :type replicate_ratio: float
            :type parent_ratio: float
            :type exploration_ratio: float
            :type crossover_type: str
            :type max_no_improve: int
            :type function_timeout: float
        """
        assert callable(function)
        self.func = function
        self.dim = dimension

        self.var_types = self._set_variable_types(var_types)
        self.var_bounds = self._set_variable_bounds(var_boundaries, var_types)

        self._set_settings(kwargs)

        # set sizes of the population and its subsets
        self.pop_size: int = int(self._settings['population_size'])
        self.n_elites: int = int(self.pop_size * self._settings['elite_ratio'])
        if self.n_elites < 1 and self._settings['elite_ratio'] > 0:
            self.n_elites = 1
        self.n_replicates = int(self.pop_size * self._settings['replicate_ratio'])
        if self.n_replicates < 1 and self._settings['replicate_ratio'] > 0:
            self.n_replicates = 1
        self.n_crossovers = int(
            (self.pop_size - self.n_elites - self.n_replicates) * (1 - self._settings['exploration_ratio'])
        )
        if self.n_crossovers % 2:
            self.n_crossovers += 1
        self.n_mutations: int = int(self.pop_size - self.n_elites - self.n_replicates - self.n_crossovers)
        self.n_parents: int = int(self.pop_size * self._settings['parent_ratio'])

        # set genetic operations probabilities
        self.p_mutation: float = self._settings['mutation_probability']
        self.p_crossover: float = self._settings['crossover_probability']

        self.n_iterations: int = self._set_iterations(n_iterations)
        self.c_type: str = self._settings['crossover_type']

        if self._settings['max_no_improve'] is None:
            self.n_no_improve: int = self.n_iterations + 1
        else:
            self.n_no_improve: int = int(self._settings['max_no_improve'])

    """Set variables"""

    def _set_variable_types(self, var_types: typing.Union[str, typing.Collection]) -> typing.Collection:
        """Set variable types.

        :param var_types: variable type(s)
        :type var_types: type, iterable

        :return: array of variable types
        :rtype: numpy.array
        """
        # array of variable types
        if not isinstance(var_types, str):
            assert len(var_types) == self.dim
            return var_types

        # single variable type
        assert var_types in ('bool', 'int', 'float')
        if var_types == 'bool':
            var_types = 'int'
        return [var_types] * self.dim

    def _set_variable_bounds(
            self, var_bounds: typing.Collection, var_types: typing.Union[str, typing.Collection]
    ) -> list:
        """Set variable boundaries. In case the variable is a boolean, no variable boundaries are required as these will
        default to [0, 1].

        :param var_bounds: variable boundaries
        :param var_types: variable type(s)

        :type var_bounds: iterable
        :type var_types: type, iterable

        :return: list of variable boundaries
        :rtype: list
        """
        if not isinstance(var_types, str):
            assert len(var_types) == len(var_bounds) == self.dim
            return [[0, 1] if vt == 'bool' else [min(vb), max(vb)] for vt, vb in zip(var_types, var_bounds)]

        if var_types == 'bool':
            return [[0, 1]] * self.dim

        assert len(var_bounds) == self.dim
        return [[min(vb), max(vb)] for vb in var_bounds]

    def _set_iterations(self, iterations: typing.Union[int, None]) -> int:
        """Set maximum number of iterations. If no value is provided (i.e. `None`), the maximum number of iterations is
        determined by means of the variable type(s) and their boundaries.

        :param iterations: maximum number of iterations
        :type iterations: int, None

        :return: maximum number of iterations
        :rtype: int
        """
        if iterations is None:
            iterations = 0
            for type_, bounds_ in zip(self.var_types, self.var_bounds):
                dim = self.dim if type_ == int else 50
                iterations += int((max(bounds_) - min(bounds_)) * dim * 100 / self.pop_size)

            if iterations * self.pop_size > 1e7:
                iterations = 1e7 / self.pop_size

        return int(iterations)

    """Genetic algorithm settings"""

    @property
    def settings(self) -> dict:
        """
        :return: genetic algorithm settings
        :rtype: dict
        """
        return self._settings

    def _set_settings(self, settings: dict) -> None:
        """Set and check custom genetic algorithm settings.

        :param settings: custom-defined settings
        :type settings: dict
        """
        for k, v in settings.items():
            if k in self._settings.keys():
                self._settings[k] = v
            else:
                _LOG.warning(f'Unknown settings key: {k}')

        self._check_settings(self._settings)

    @staticmethod
    def _check_settings(settings: dict) -> None:
        """Check genetic algorithm settings.

        :param settings: algorithm settings
        :type settings: dict
        """
        probabilities = ('crossover_probability', 'mutation_probability')
        assert all(0 <= settings[k] <= 1 for k in probabilities)

        ratios = ('elite_ratio', 'replicate_ratio', 'exploration_ratio', 'parent_ratio')
        assert all(0 <= settings[k] <= 1 for k in ratios)

        crossover_types = ('index', 'slice', 'uniform')
        assert settings['crossover_type'] in crossover_types

    """Genetic operations: Crossover"""

    def apply_crossover(self, parent_1: np.ndarray, parent_2: np.ndarray) -> tuple:
        """Apply crossover operation. The following crossover methods are included:
         1. 'index'     :   switch the "genes" of the parents until a randomly selected index of the array.
         2. 'slice'     :   switch the "genes" of the parents between two randomly selected indices of the array.
         3. 'uniform'   :   switch the "genes" of the parents at random throughout the array.

        :param parent_1: parent (1)
        :param parent_2: parent (2)

        :type parent_1: numpy.array
        :type parent_2: numpy.array

        :return: children
        :rtype: tuple
        """
        child_1 = parent_1[:self.dim].copy()
        child_2 = parent_2[:self.dim].copy()

        if self.c_type == 'index':
            r = np.random.randint(self.dim)
            child_1[:r] = parent_2[:r]
            child_2[:r] = parent_1[:r]

        elif self.c_type == 'slice':
            r1 = np.random.randint(0, self.dim)
            r2 = np.random.randint(r1, self.dim)
            child_1[r1:r2] = parent_2[r1:r2]
            child_2[r1:r2] = parent_1[r1:r2]

        elif self.c_type == 'uniform':
            r = np.random.random(self.dim)
            child_1[r < .5] = parent_2[r < .5]
            child_2[r < .5] = parent_1[r < .5]

        else:
            msg = f'Unknown crossover-type: {self.c_type} (see documentation)'
            raise NotImplementedError(msg)

        return child_1, child_2

    def crossover(self, parents: np.ndarray, repeats: int) -> list:
        """Crossover operations.

        :param parents: selection of parents
        :param repeats: number of repetitions, output is twice the size of this value

        :type parents: numpy.array
        :type repeats: int

        :return: children
        :rtype: list
        """
        children = []
        for _ in range(int(repeats)):
            couple = parents[np.random.randint(len(parents), size=2), :self.dim].copy()
            children.extend([
                np.append(c, self.func(c)) for c in self.apply_crossover(*couple)
            ])
        return children.copy()

    """Genetic operations: Mutation"""

    @staticmethod
    def apply_mutation(bounds: typing.Collection, var_type: str) -> typing.Union[int, float]:
        """Apply mutation operation.

        :param bounds: variable bounds
        :param var_type: variable type

        :type bounds: iterable
        :type var_type: type

        :return: variable mutation
        :rtype: int, float
        """
        if var_type == 'int':
            return int(np.random.randint(min(bounds), max(bounds) + 1))
        elif var_type == 'float':
            return float(min(bounds) + np.random.random() * (max(bounds) - min(bounds)))
        else:
            msg = f'Unknown variable-type: {var_type}.'
            raise TypeError(msg)

    def mutate(self, parent: np.ndarray, *bounds: typing.Collection) -> np.ndarray:
        """Mutation operation. If no variable boundaries (i.e. `bounds`) are defined, the global variable boundaries are
        used, defined by `self.var_bounds`.

        :param parent: parent
        :param bounds: variable boundaries

        :type parent: numpy.array
        :type bounds: iterable

        :return: child
        :rtype: numpy.array
        """
        assert len(bounds) in (0, 2)

        bounds = np.transpose(bounds) if bounds else self.var_bounds
        child = parent[:self.dim].copy()

        for i in range(self.dim):
            if np.random.random() < self.p_mutation:
                child[i] = self.apply_mutation(bounds[i], self.var_types[i])

        return np.append(child, self.func(child))

    """Selection procedure"""

    def sort_population(self, population: np.ndarray) -> np.ndarray:
        """Sort population based on fitness: Lowest value (i.e. highest fitness) on top.

        :param population: population
        :type population: numpy.array

        :return: sorted population
        :rtype: numpy.array
        """
        return np.array(sorted(population, key=lambda p: p[self.dim]))

    @staticmethod
    def calc_probability(fitness: np.ndarray) -> np.ndarray:
        """Calculate probability of selection based on the fitness: The highest fitness (i.e. lowest fitness value) has
        the highest change of selection.

        :param fitness: population fitness
        :type fitness: numpy.array

        :return: probability of selection
        :rtype: numpy.array
        """
        fitness_ = fitness.copy()
        if fitness_[0] < 0:
            fitness_ += abs(fitness_[0])

        norm_fitness = max(fitness_) - fitness_ + 1
        return np.cumsum(norm_fitness / sum(norm_fitness))

    @staticmethod
    def selection(population: np.ndarray, size: int, probability: np.ndarray = None):
        """Select a subset of the population. When no probability of selection (i.e. `probability`) is provided, the top
        of the population is selected, i.e. the people with the highest fitness. Otherwise, the selection includes a
        random factor.

        :param population: population
        :param size: size of selection
        :param probability: probability of selection, defaults to None

        :type population: numpy.array
        :type size: int
        :type probability: float, optional

        :return: subset of population
        :rtype: numpy.array
        """
        if probability is None:
            return population[:size].copy()

        return population[np.searchsorted(probability, np.random.random(size))].copy()

    """Output data"""

    def output_update(self, population: np.ndarray, best_pool: typing.Collection, deficit: float) -> tuple:
        """Update output data, consisting of the best person and optionally the pool of best people based on this best
        person and the pool criteria.

        :param population: population
        :param best_pool: pool of best people
        :param deficit: relative difference between person and best person to be allowed in the best pool

        :type population: numpy.array
        :type best_pool: iterable
        :type deficit: float

        :return: best person, and pool of best people (optional)
        :rtype: tuple
        """
        person = population[0].copy()
        if len(best_pool) > 0:
            assert 0 < deficit < 1
            pool = self.output_pool(best_pool, person[self.dim], deficit)
            pool = np.append(
                pool, self.output_pool(population, person[self.dim], deficit)
            ).reshape((-1, self.dim + 1))
        else:
            pool = best_pool.copy()

        return person, pool

    def output_pool(self, population: np.ndarray, best_fitness: float, deficit: float) -> np.ndarray:
        """Determine which people are allowed in the pool of best people based on the best person's fitness.

        :param population: population
        :param best_fitness: best person's fitness
        :param deficit: relative difference between person and the best person to be allowed in the best pool

        :type population: numpy.array
        :type best_fitness: float
        :type deficit: float

        :return: selection of best people
        :rtype: numpy.array
        """
        assert 0 < deficit < 1
        fitness = population[:, self.dim]
        return population[np.where(abs(fitness - best_fitness) < deficit * abs(best_fitness))].copy()

    """Progress data"""

    def progress_update(
            self, population: np.ndarray, progress_details: dict, progress_data: dict = None, **kwargs
    ) -> dict:
        """Initiate and update progress data. The data included is defined by `progress_details`:
         -  None        :   store the best fitness [default]
         -  'range'     :   store the best fitness, and the worst fitness
         -  'stats'     :   store the best fitness, the worst fitness, the mean fitness, and the standard deviation in
                            fitness
         -  'pool'      :   store the fitness of the pool with best people
         -  'pool-max'  :   store maximum fitness of the pool with best people
         -  'full'      :   store the fitness of the whole population
         -  'all'       :   store the best fitness, the worst fitness, the mean fitness, the standard deviation in
                            fitness, and the fitness of the pool with best people, the fitness of the whole population
                            (i.e. all the above)

        The returned dictionary includes the following (optional) keys:
         -  'best_fitness'      :   value of best fitness
         -  'worst_fitness'     :   value of worst fitness (optional)
         -  'mean_fitness'      :   mean value of the fitness of the whole population (optional)
         -  'std_fitness'       :   standard deviation of the fitness of the whole population (optional)
         -  'pool_fitness'      :   array of fitness values of the pool of best people (optional)
         -  'max_pool_fitness   :   value of worst fitness in the pool of best people (optional)
         -  'pop_fitness'       :   array of fitness values of the whole population (optional)

        :param population: population
        :param progress_details: progress details to include
        :param progress_data: previous progress data, defaults to None
        :param kwargs: data to accelerate the execution, or required to store the requested data
            best_fitness: fitness of best person, defaults to its calculation
            best_pool: pool with best performing people, defaults to raising error when missing

        :type population: numpy.array
        :type progress_details: dict
        :type progress_data: dict, optional
        :type kwargs: optional
            best_fitness: float
            best_pool: numpy.array

        :return: updated progress data
        :rtype: dict
        """
        # include best person and fitness
        data = {
            'best_fitness': kwargs.get('best_fitness', min(population[:, self.dim]))
        }

        # include worst fitness
        if progress_details in ('range', 'stats', 'all'):
            data['worst_fitness'] = max(population[:, self.dim])

        # include fitness statistics
        if progress_details in ('stats', 'all'):
            data['mean_fitness'] = np.mean(population[:, self.dim])
            data['std_fitness'] = np.std(population[:, self.dim])

        # include best pool's fitness
        if progress_details in ('pool', 'all'):
            best_pool = kwargs.get('best_pool')
            if best_pool is None:
                msg = f'No best pool is provided: Cannot store its data.'
                raise ValueError(msg)
            elif len(best_pool) == 0:
                msg = f'No best pool is determined: Cannot store its data.'
                raise ValueError(msg)
            data['pool_fitness'] = list(best_pool[:, self.dim])

        if progress_details in ('pool-max', 'all'):
            best_pool = kwargs.get('best_pool')
            if best_pool is None:
                msg = f'No best pool is provided: Cannot use its data.'
                raise ValueError(msg)
            elif len(best_pool) == 0:
                msg = f'No best pool is determined: Cannot use its data.'
                raise ValueError(msg)
            data['max_pool_fitness'] = max(best_pool[:, self.dim])

        # include whole population's fitness
        if progress_details in ('full', 'all'):
            data['pop_fitness'] = list(population[:, self.dim])

        # initiate progress data
        if progress_data is None:
            progress_data = {k: [] for k in data.keys()}

        # append progress data
        for k in progress_data.keys():
            progress_data[k].append(data[k])

        # return progress data
        return progress_data

    """Execution"""

    def exec(self, **kwargs) -> tuple:
        """Execute genetic algorithm.

        :param kwargs: execution settings
            max_iterations: overwrite the maximum number of iterations, defaults to None
            n_no_improve: overwrite the maximum number of iterations without improving performance, defaults to
                None

            output_pool: include a pool of similarly performing people (or samples), defaults to False
            output_pool_deficit: relative difference in fitness with the "best person" to be allowed in the
                output pool, defaults to 0.1

            progress_bar: print a progress bar, defaults to False
            progress_bar_length: print-length of progress bar, defaults to 50
            progress_export: export progress details, if a directory is provided, the progress details are exported
                accordingly, defaults to False
            progress_details: progress details to be stored:
                None        :   store the best fitness [default]
                'range'     :   store the best fitness, and the worst fitness
                'stats'     :   store the best fitness, the worst fitness, the mean fitness, and the standard deviation
                                in fitness
                'pool'      :   store the fitness of the pool with best people
                'pool-max'  :   store maximum fitness of the pool with best people
                'full'      :   store the fitness of the whole population
                'all'       :   store the best fitness, the worst fitness, the mean fitness, the standard deviation in
                                fitness, and the fitness of the pool with best people (i.e. all the above)

        :type kwargs: optional
            max_iterations: int
            n_no_improve: int

            output_pool: bool
            output_pool_deficit: float

            progress_bar: bool
            progress_bar_length: int
            progress_export: bool, DirConfig, str, iterable
            progress_details: str

        :return: output data, progress data, and pool of best fits (optional)
        :rtype: tuple
        """
        # execution settings
        # > iterations
        self.n_iterations = self._set_iterations(kwargs.get('max_iterations', self.n_iterations))
        self.n_no_improve = int(kwargs.get('n_no_improve', self.n_no_improve))
        # > output
        output_pool: bool = kwargs.get('output_pool', False)
        output_pool_deficit: float = kwargs.get('output_pool_deficit', .1)
        # > progress
        progress_bar: bool = kwargs.get('progress_bar', False)
        progress_bar_length: int = kwargs.get('progress_bar_length', 50)
        progress_export: bool = kwargs.get('progress_export', False)
        progress_details: dict = kwargs.get('progress_details')

        # initial population
        population = np.zeros((self.pop_size, self.dim + 1))
        for pi in range(self.pop_size):
            person = np.array([self.apply_mutation(vb, vt) for vb, vt in zip(self.var_bounds, self.var_types)])
            population[pi] = person + [self.func(person)]

        # initiate best performance variables
        best_person = min(population, key=lambda p: p[self.dim])
        best_pool = self.output_pool(population, best_person[self.dim], output_pool_deficit) if output_pool else []

        # initiate progress variables
        n_no_improve = 0
        progress_data = self.progress_update(
            population, progress_details,
            best_fitness=best_person[self.dim], best_pool=best_pool
        )

        # evolution
        for t in range(self.n_iterations):
            if progress_bar:
                _progress_bar(t, self.n_iterations, bar_length=progress_bar_length)

            # sort population
            population = self.sort_population(population)

            if population[0, self.dim] < best_person[self.dim]:
                best_person, best_pool = self.output_update(population, best_pool, output_pool_deficit)
                n_no_improve = 0
            else:
                n_no_improve += 1

            # update progress data
            progress_data = self.progress_update(
                population, progress_details, progress_data,
                best_fitness=best_person[-1], best_pool=best_pool
            )

            # probability of selection
            probability = self.calc_probability(population[:, self.dim])

            # new generation: elitism
            elites = self.selection(population, self.n_elites)

            # new generation: replication
            replicates = self.selection(population, self.n_replicates, probability=probability)

            # new generation: crossover
            parents = self.selection(population, self.n_parents, probability=probability)
            crossovers = self.crossover(parents, int(self.n_crossovers / 2))

            # new generation: mutation
            parents = self.selection(population, self.n_mutations, probability=probability)
            mutations = [self.mutate(p) for p in parents]

            # new generation
            assert self.pop_size == len(elites) + len(replicates) + len(crossovers) + len(mutations)
            population = np.concatenate([
                elites, replicates, crossovers, mutations
            ]).reshape((self.pop_size, self.dim + 1))

            # no improvement
            if n_no_improve >= self.n_no_improve:
                if progress_bar:
                    _progress_bar(self.n_iterations, self.n_iterations, bar_length=progress_bar_length)
                    time.sleep(.1)
                _LOG.warning(f'{n_no_improve} iterations without improvement: search halted after {t + 1} iterations.')
                break

        if progress_bar and n_no_improve < self.n_no_improve:
            _progress_bar(self.n_iterations, self.n_iterations, bar_length=progress_bar_length)

        # sort population
        population = self.sort_population(population)
        if population[0, self.dim] < best_person[self.dim]:
            best_person, best_pool = self.output_update(population, best_pool, output_pool_deficit)

        # update progress data
        progress_data = self.progress_update(
            population, progress_details, progress_data,
            best_fitness=best_person[-1], best_pool=best_pool
        )

        # export progress details
        if progress_export:
            export = filing.Export(None if isinstance(progress_export, bool) else progress_export)
            export.to_csv(data=progress_data, file_name='ga_progress', index=False)

        output_dict = {
            'variable': best_person[:self.dim],
            'function': best_person[self.dim],
        }

        # return output data, progress data, and pool of best fits
        return output_dict, progress_data, best_pool


def _progress_bar(step: int, n_iterations: int, **kwargs) -> None:
    """Print the progress of an execution.

    :param step: execution step
    :param n_iterations: total number of iterations
    :param kwargs:
        bar_length: printed length of progress bar, default to 50
        msg: message displayed while running, defaults to 'running...'

    :type step: int
    :type n_iterations: int
    :type kwargs: optional
        bar_length: int
        msg: str
    """
    # optional settings
    length: int = kwargs.get('bar_length', 50)
    msg: str = kwargs.get('msg', 'running...') if step < n_iterations else 'completed\n'

    # progress
    completed = step / n_iterations
    filled = int(round(completed * length))

    # create bar
    bar = '|' + '.' * filled + ' ' * (length - filled) + '|'

    # write bar to screen
    sys.stdout.write(f'\r{bar} {f"{completed * 100:.1f}":>5}% | {msg}')
    sys.stdout.flush()


def _export2csv(data: dict, file_name: str = 'ga_process.csv', wd: str = None) -> None:
    """Export `dict`-data

    :param data:
    :param file_name:
    :param wd:
    :return:
    """
    # optional arguments: output file
    wd = wd or os.getcwd()
    if not file_name.endswith('.csv'):
        file_name += '.csv'
    file = os.path.join(wd, file_name)

    # data check: values of equal length
    assert len(set([len(v) for v in data.values()])) == 1, \
        'All lists in dictionary must have the same length'

    # prepare data
    rows = zip(*data.values())

    # write to `*.csv`-file
    with open(file, mode='w') as f:
        # header
        f.write(','.join(data.keys()) + '\n')
        # content
        f.write('\n'.join(
            ','.join(map(str, row)) for row in rows
        ))
