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
    """A genetic algorithm that searches for the minimum of a fitness function with the option to search for a selection
    that has a fitness within a prescribed deficit of the best performance.
    """
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
            var_types: typing.Union[str, typing.Collection],
            var_boundaries: typing.Collection,
            n_iterations: int = None,
            **kwargs
    ) -> None:
        """
        :param function: fitness function
        :param dimension: input dimension
        :param var_types: variable type(s)
            {'bool', 'float', 'int'}
        :param var_boundaries: variable boundaries
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

        :type function: callable
        :type dimension: int
        :type var_types: str, typing.Collection
        :type var_boundaries: typing.Collection
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

        :raises TypeError: if `function` is not callable
        """
        # function must be callable
        if not callable(function):
            msg = f'Fitness\' function must be callable: `function` of type {type(function)}'
            raise TypeError(msg)

        # initiate object
        self.func = function
        self.dim = dimension

        self.var_types = self._set_variable_types(var_types)
        self.var_bounds = self._set_variable_bounds(var_boundaries, self.var_types)

        self._set_settings(kwargs)

        # set sizes of the population and its subsets
        self.pop_size: int = int(self._settings['population_size'])
        self.n_elites: int = int(self.pop_size * self._settings['elite_ratio'])
        if self.n_elites < 1 and self._settings['elite_ratio'] > 0:
            self.n_elites = 1
        self.n_replicates: int = int(self.pop_size * self._settings['replicate_ratio'])
        if self.n_replicates < 1 and self._settings['replicate_ratio'] > 0:
            self.n_replicates = 1
        self.n_crossovers: int = int(
            (self.pop_size - self.n_elites - self.n_replicates) * (1 - self._settings['exploration_ratio'])
        )
        if self.n_crossovers % 2:
            self.n_crossovers += 1
        self.n_mutations: int = int(self.pop_size - self.n_elites - self.n_replicates - self.n_crossovers)
        self.n_parents: int = int(self.pop_size * self._settings['parent_ratio'])

        # set genetic operations probabilities
        self.p_mutation: float = self._settings['mutation_probability']
        self.p_crossover: float = self._settings['crossover_probability']

        # set other settings
        self.n_iterations: int = self._set_iterations(n_iterations)
        self.c_type: str = self._settings['crossover_type']
        self.max_no_improve: int = int(self._settings['max_no_improve'] or self.n_iterations + 1)

    """Set variables"""

    def _set_variable_types(self, var_types: typing.Union[str, typing.Collection]) -> list:
        """Set variable types.

        :param var_types: variable type(s)
        :type var_types: type, iterable

        :return: array of variable types
        :rtype: list

        :raises ValueError: if length of `var_types` mismatches dimensionality
        :raises ValueError: if any invalid variable types are provided
        """
        _var_types = ('bool', 'int', 'float')

        def set_single_var_type(var_type: str) -> str:
            """Check, modify, and return per variable type.

            :param var_type: variable type
            :type var_type: str

            :return: valid variable type
            :rtype: str

            :raises ValueError: if `var_type` is invalid
            """
            # check validity variable type
            if var_type not in _var_types:
                msg = f'Invalid variable type: {var_types} not in {_var_types}.'
                raise ValueError(msg)

            # convert 'bool' to 'int'
            if var_type == 'bool':
                var_type = 'int'

            # return valid variable type
            return var_type

        # single variable type
        if isinstance(var_types, str):
            # return list of variable types
            return [set_single_var_type(var_types)] * self.dim

        # list of variable types
        if not len(var_types) == self.dim:
            # assert correct dimensionality
            if not len(var_types) == self.dim:
                msg = f'Length of `var_types` mismatches dimensionality: {len(var_types)} =/= {self.dim}.'
                raise ValueError(msg)
            # return list of variable types
            return [set_single_var_type(vt) for vt in var_types]

    def _set_variable_bounds(
            self, var_bounds: typing.Collection, var_types: typing.Collection[str]
    ) -> list:
        """Set variable boundaries. In case the variable is a boolean, no variable boundaries are required as these will
        default to [0, 1].

        :param var_bounds: variable boundaries
        :param var_types: variable type(s)

        :type var_bounds: iterable
        :type var_types: type, iterable

        :return: list of variable boundaries
        :rtype: list

        :raises ValueError: if length of `var_bounds` and/or `var_types` mismatches dimensionality
        """
        # assert correct dimensionality
        var_bounds = np.reshape(var_bounds, (-1, 2))
        if not len(var_bounds) == self.dim:
            msg = f'Length of `var_bounds` mismatches dimensionality: {len(var_bounds)} =/= {self.dim}'
            raise ValueError(msg)

        # return variable boundaries
        return [[0, 1] if vt == 'bool' else [min(vb), max(vb)] for vb, vt in zip(var_bounds, var_types)]

    def _set_iterations(self, iterations: typing.Union[int, None]) -> int:
        """Set maximum number of iterations. If no value is provided (i.e. `None`), the maximum number of iterations is
        determined by means of the variable type(s) and their boundaries.

        :param iterations: maximum number of iterations
        :type iterations: int, None

        :return: maximum number of iterations
        :rtype: int
        """
        # determine number of iterations
        if iterations is None:
            iterations = 0
            for vt, vb in zip(self.var_types, self.var_bounds):
                dim = self.dim if vt == int else 50
                iterations += int((max(vb) - min(vb)) * dim * 100 / self.pop_size)

            if iterations * self.pop_size > 1e7:
                iterations = 1e7 / self.pop_size

        # return number of iterations
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
        # update GA-settings
        for k, v in settings.items():
            if k in self._settings:
                self._settings[k] = v
            else:
                _LOG.warning(f'Unknown setting\'s key: {k} [skipped]')

        # check GA-settings
        self._check_settings(self._settings)

    @staticmethod
    def _check_settings(settings: dict) -> None:
        """Check genetic algorithm settings.

        :param settings: algorithm settings
        :type settings: dict

        :raises ValueError: if not all probabilities are in [0, 1]
        :raises ValueError: if not all ratios are in [0, 1]
        :raises ValueError: if `crossover_type` is unknown
        """
        probabilities = ('crossover_probability', 'mutation_probability')
        if not all(0 <= settings[k] <= 1 for k in probabilities):
            msg = f'Not all probabilities are in [0, 1]: {settings}'
            raise ValueError(msg)

        ratios = ('elite_ratio', 'replicate_ratio', 'exploration_ratio', 'parent_ratio')
        if not all(0 <= settings[k] <= 1 for k in ratios):
            msg = f'Not all ratios are in [0, 1]: {settings}'
            raise ValueError(msg)

        crossover_types = ('index', 'slice', 'uniform')
        if settings['crossover_type'] not in crossover_types:
            msg = f'Unknown `crossover_type`: {settings["crossover_type"]} not in {crossover_types}'
            raise ValueError(msg)

    """Genetic operations: Crossover"""

    def apply_crossover(self, parent_1: np.ndarray, parent_2: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
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

        :raises ValueError: unknown crossover-type definition
        """
        # initiate children: copies of parents
        child_1: np.ndarray = parent_1[:self.dim].copy()
        child_2: np.ndarray = parent_2[:self.dim].copy()

        # apply crossover: index
        if self.c_type == 'index':
            r = np.random.randint(self.dim)
            child_1[:r] = parent_2[:r]
            child_2[:r] = parent_1[:r]

        # apply crossover: slice
        elif self.c_type == 'slice':
            r1 = np.random.randint(0, self.dim)
            r2 = np.random.randint(r1, self.dim)
            child_1[r1:r2] = parent_2[r1:r2]
            child_2[r1:r2] = parent_1[r1:r2]

        # apply crossover: uniform
        elif self.c_type == 'uniform':
            r = np.random.random(self.dim)
            child_1[r < .5] = parent_2[r < .5]
            child_2[r < .5] = parent_1[r < .5]

        # unknown crossover type
        else:
            msg = f'Unknown crossover-type: {self.c_type} (see documentation)'
            raise ValueError(msg)

        # return children
        return child_1, child_2

    def crossover(self, parents: np.ndarray, repeats: int) -> np.ndarray:
        """Crossover operations.

        :param parents: selection of parents
        :param repeats: number of repetitions, output is twice the size of this value

        :type parents: numpy.array
        :type repeats: int

        :return: children
        :rtype: numpy.ndarray
        """
        children = []
        for _ in range(int(repeats)):
            couple = parents[np.random.randint(len(parents), size=2), :self.dim].copy()
            children.extend([
                np.append(c, self.func(*c)) for c in self.apply_crossover(*couple)
            ])
        return np.array(children)

    """Genetic operations: Mutation"""

    @staticmethod
    def apply_mutation(bounds: np.ndarray, var_type: str) -> typing.Union[int, float]:
        """Apply mutation operation.

        :param bounds: variable bounds
        :param var_type: variable type

        :type bounds: numpy.ndarray
        :type var_type: type

        :return: variable mutation
        :rtype: int, float

        :raises ValueError: if `var_type` is unknown
        """
        # apply mutation: int
        if var_type == 'int':
            return int(np.random.randint(min(bounds), max(bounds) + 1))

        # apply mutation: float
        elif var_type == 'float':
            return float(min(bounds) + np.random.random() * (max(bounds) - min(bounds)))

        # unknown variable type
        else:
            msg = f'Unknown variable-type: {var_type}.'
            raise ValueError(msg)

    def mutate(self, parent: np.ndarray) -> np.ndarray:
        """Mutation operation. If no variable boundaries (i.e. `bounds`) are defined, the global variable boundaries are
        used, defined by `self.var_bounds`.

        :param parent: parent
        :type parent: numpy.array

        :return: child
        :rtype: numpy.array
        """
        # initiate child
        child = parent[:self.dim].copy()

        # mutate child
        for i in range(self.dim):
            if np.random.random() < self.p_mutation:
                child[i] = self.apply_mutation(self.var_bounds[i], self.var_types[i])

        # return child
        return np.append(child, self.func(*child))

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
        # minimum fitness of zero
        fitness_ = fitness.copy()
        if fitness_[0] < 0:
            fitness_ += abs(fitness_[0])

        # normalise fitness
        norm_fitness = max(fitness_) - fitness_ + 1

        # return probabilities based on fitness
        return np.cumsum(norm_fitness / sum(norm_fitness))

    @staticmethod
    def selection(population: np.ndarray, size: int, probability: np.ndarray = None) -> np.ndarray:
        """Select a subset of the population. When no probability of selection (i.e. `probability`) is provided, the top
        of the population is selected, i.e. the people with the highest fitness. Otherwise, the selection includes a
        random factor.

        :param population: population
        :param size: size of selection
        :param probability: probability of selection, defaults to None

        :type population: numpy.ndarray
        :type size: int
        :type probability: numpy.ndarray, optional

        :return: subset of population
        :rtype: numpy.ndarray
        """
        # select top-ranked population
        if probability is None:
            return population[:size].copy()

        # select based on selection-probabilities
        return population[np.searchsorted(probability, np.random.random(size))].copy()

    """Output data"""

    def output_update(
            self, population: np.ndarray, best_pool: np.ndarray, deficit: float
    ) -> typing.Tuple[np.ndarray, typing.Union[None, np.ndarray]]:
        """Update output data, consisting of the best person and optionally the pool of best people based on this best
        person and the pool criteria.

        :param population: population
        :param best_pool: pool of best people
        :param deficit: relative difference between person and best person to be allowed in the best pool

        :type population: numpy.ndarray
        :type best_pool: numpy.ndarray
        :type deficit: float

        :return: best person, and pool of best people (optional)
        :rtype: tuple

        :raises AssertionError: if best pool's `deficit` is not in (0, 1)
        """
        # best person
        person = population[0].copy()

        # pool of best people
        if best_pool is None:
            # best pool is disabled
            pool = None
        else:
            # update best pool's people
            assert 0 < deficit < 1, \
                f'Best pool\'s `deficit` must be in (0, 1), {deficit} given.'
            # remove unfit persons
            pool = self.output_pool(best_pool, person[self.dim], deficit)
            # add fit persons
            pool = np.append(
                pool, self.output_pool(population, person[self.dim], deficit)
            ).reshape((-1, self.dim + 1))

        # return best person and pool
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

        :raises AssertionError: if best pool's `deficit` is not in (0, 1)
        """
        assert 0 < deficit < 1, \
            f'Best pool\'s fitness must be in (0, 1), {deficit} given.'
        fitness = population[:, self.dim]
        return population[np.where(abs(fitness - best_fitness) < deficit * abs(best_fitness))].copy()

    """Progress data"""

    def _collect_progress_data(self, population: np.ndarray, progress_details: str, **kwargs) -> dict:
        """Collect data on evolutionary progress. See documentation of `.progress_update()` on the possible keywords of
        `progress_details`, and what data is collected based on every keyword.

        :param population: population
        :param progress_details: progress details to include
        :param kwargs: data to accelerate the execution, or required to store the requested data
            best_fitness: fitness of best person, defaults to `min(population[:, self.dim])`
            best_pool: pool with best performing people, defaults to None

        :type population: numpy.ndarray
        :type progress_details: str
        :type kwargs: optional
            best_fitness: float
            best_pool: numpy.ndarray

        :return: collected progress data
        :rtype: dict

        :raises ValueError: if `best_pool` has a length of zero.
        """
        # optional arguments
        best_fitness: float = kwargs.get('best_fitness', min(population[:, self.dim]))
        best_pool: typing.Union[np.ndarray, None] = kwargs.get('best_pool')

        # include best person's fitness
        data = {
            'best_fitness': best_fitness,
        }

        # include worst fitness
        if progress_details in ('range', 'stats', 'all'):
            data['worst_fitness'] = max(population[:, self.dim])

        # include fitness statistics
        if progress_details in ('stats', 'all'):
            data['mean_fitness'] = np.mean(population[:, self.dim])
            data['std_fitness'] = np.std(population[:, self.dim])

        # include best pool's fitness
        if progress_details in ('pool', 'pool-max', 'all'):
            # best pool option disabled: skip
            if best_pool is None:
                _LOG.debug(f'No best pool included but cannot store its data: `progress_details={progress_details}`')
                pass
            # best pool is empty: raise error
            if len(best_pool) == 0:
                msg = f'No best pool is determined: Cannot store its data.'
                raise ValueError(msg)
            # update best pool's fitness
            else:
                if progress_details in ('pool', 'all'):
                    data['pool_fitness'] = list(best_pool[:, self.dim])
                if progress_details in ('pool-max', 'all'):
                    data['max_pool_fitness'] = max(best_pool[:, self.dim])

        # include whole population's fitness
        if progress_details in ('full', 'all'):
            data['pop_fitness'] = list(population[:, self.dim])

        # return progress data
        return data

    def progress_update(
            self, population: np.ndarray, progress_details: str, progress_data: dict = None, **kwargs
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
        :type progress_details: str
        :type progress_data: dict, optional
        :type kwargs: optional
            best_fitness: float
            best_pool: numpy.ndarray

        :return: updated progress data
        :rtype: dict
        """
        # collect progress data
        data = self._collect_progress_data(population, progress_details, **kwargs)

        # initiate progress data
        if progress_data is None:
            progress_data = {k: [] for k in data.keys()}

        # append progress data
        for k in progress_data.keys():
            progress_data[k].append(data[k])

        # return progress data
        return progress_data

    """Execution"""

    def exec(self, **kwargs) -> typing.Tuple[dict, dict, typing.Union[None, np.ndarray]]:
        """Execute genetic algorithm.

        :param kwargs: execution settings
            max_iterations: overwrite the maximum number of iterations, defaults to None
            max_no_improve: overwrite the maximum number of iterations without improving performance, defaults to None

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
            max_no_improve: int

            output_pool: bool
            output_pool_deficit: float

            progress_bar: bool
            progress_bar_length: int
            progress_export: bool, DirConfig, str, iterable
            progress_details: str

        :return: output data, progress data, and pool of best fits (optional)
        :rtype: tuple

        :raises ValueError: if `output_pool_deficit` is not in (0, 1)
        :raises AssertionError: if initiated population size and sub-population sizes do not match up
        """
        # execution settings
        # > iterations
        n_iterations: int = kwargs.get('max_iterations', self.n_iterations)
        max_no_improve: int = kwargs.get('max_no_improve', self.max_no_improve)
        # > output
        output_pool: bool = kwargs.get('output_pool', False)
        output_pool_deficit: float = kwargs.get('output_pool_deficit', .1)
        if not (0 < output_pool_deficit < 1):
            msg = f'Best pool\'s `deficit` must be in (0, 1), {output_pool_deficit} given.'
            raise ValueError(msg)
        # > progress
        progress_bar: bool = kwargs.get('progress_bar', False)
        progress_bar_length: int = kwargs.get('progress_bar_length', 50)
        progress_export: typing.Union[bool, str] = kwargs.get('progress_export', False)
        progress_details: str = kwargs.get('progress_details')

        # initial population
        population = np.zeros((self.pop_size, self.dim + 1))
        for pi in range(self.pop_size):
            person = [self.apply_mutation(vb, vt) for vb, vt in zip(self.var_bounds, self.var_types)]
            population[pi] = person + [self.func(*person)]

        # initiate best performance variables
        best_person = min(population, key=lambda p: p[self.dim])
        best_pool = self.output_pool(population, best_person[self.dim], output_pool_deficit) if output_pool else None

        # initiate progress variables
        n_no_improve = 0
        progress_data = self.progress_update(
            population, progress_details,
            best_fitness=best_person[self.dim], best_pool=best_pool
        )

        # evolution
        for t in range(n_iterations):
            if progress_bar:
                _progress_bar(t, n_iterations, bar_length=progress_bar_length)

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
                best_fitness=best_person[self.dim], best_pool=best_pool
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
            pop_size = len(elites) + len(replicates) + len(crossovers) + len(mutations)
            assert self.pop_size == pop_size, \
                f'Population size is not ensured: {self.pop_size} =/= {pop_size}'
            population = np.concatenate([
                elites, replicates, crossovers, mutations
            ]).reshape((self.pop_size, self.dim + 1))

            # no improvement
            if n_no_improve >= max_no_improve:
                if progress_bar:
                    _progress_bar(n_iterations, n_iterations, bar_length=progress_bar_length)
                    time.sleep(.1)
                _LOG.warning(f'{n_no_improve} iterations without improvement: search halted after {t + 1} iterations.')
                break

        if progress_bar and n_no_improve < max_no_improve:
            _progress_bar(n_iterations, n_iterations, bar_length=progress_bar_length)

        # sort population
        population = self.sort_population(population)
        if population[0, self.dim] < best_person[self.dim]:
            best_person, best_pool = self.output_update(population, best_pool, output_pool_deficit)

        # update progress data
        progress_data = self.progress_update(
            population, progress_details, progress_data,
            best_fitness=best_person[self.dim], best_pool=best_pool
        )

        # export progress details
        if progress_export:
            wd = progress_export if isinstance(progress_export, str) else None
            _export2csv(progress_data, wd=wd)

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


def _export2csv(data: dict, file_name: str = None, wd: str = None) -> None:
    """Export `dict`-data as `*.csv`-file.

    :param data: data
    :param file_name: file name, defaults to None
    :param wd: working directory, defaults to None

    :type data: dict
    :type file_name: str, optional
    :type wd: str, optional

    :raises AssertionError: if data-values are not all of same length
    """
    # optional arguments: output file
    wd = wd or os.getcwd()
    file_name = file_name or 'ga_progress.csv'
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

    _LOG.info(f'Data exported to {file}')
