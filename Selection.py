import copy
from Individual import Individual


class Selection:

    '''
    Selection operation for differential evolution algorithm
    '''

    def __init__(self, population):
        '''
        population: individual实体
        '''
        self.population = population

    def evaluate(self, x):
        '''
        evaluation of the population fitnesses
        '''
        x.calculateFitness()



    def selectionOperation(self, i, ui):
        '''
        selection operation for differential evolution algorithm
        '''
        xi_next = copy.deepcopy(self.population[i])
        xi_next.chrom = ui
        self.evaluate(xi_next)
        if xi_next.fitness > self.population[i].fitness:
            return xi_next
        else:
            return self.population[i]
