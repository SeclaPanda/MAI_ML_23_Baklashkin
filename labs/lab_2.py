import pygmo as pg
import numpy as np

class SphereFunction:
    def fitness(self, x):
        return [sum(x**2)]

    def get_bounds(self):
        return ([-10, -10], [10, 10])

class BoothFunction:
    def fitness(self, x):
        return [(x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2]

    def get_bounds(self):
        return ([-10, -10], [10, 10])
     

def optimize_and_print(algorithm, problem):
    prob = pg.problem(problem)
    algo = pg.algorithm(algorithm(gen=100))

    pop = pg.population(prob, size=10)
    result = algo.evolve(pop)

    best_solution = result.champion_x

    print(f"Алгоритм: {algorithm.__name__}, Функция: {problem.__class__.__name__}")
    print(f"Лучшее решение: {best_solution}")
    print(f"Значение функции: {problem.fitness(best_solution)[0]}")
    print("\n")

# Сравнение для функции сферы
sphere_problem = SphereFunction()
optimize_and_print(pg.de, sphere_problem)
optimize_and_print(pg.pso, sphere_problem)
optimize_and_print(pg.sga, sphere_problem)

# Сравнение для функции Бута
booth_problem = BoothFunction()
optimize_and_print(pg.de, booth_problem)
optimize_and_print(pg.pso, booth_problem)
optimize_and_print(pg.sga, booth_problem)