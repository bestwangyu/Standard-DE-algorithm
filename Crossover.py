import numpy as np
import random

class Crossover:

    '''
    crossover operation for differential evolution algorithm
    '''

    def __init__(self,  vardim, population ,params):
        '''
        vardim: dimension of variables
        bound: boundaries of variables
        '''
        self.vardim = vardim
        self.population = population
        self.params = params

    def Standard_Crossover(self, i, vi):
        '''
        Standard crossover operation for differential evolution algorithm
        '''
        k = np.random.random_integers(0, self.vardim - 1)
        ui = np.zeros(self.vardim)
        for j in range(0, self.vardim):
            pick = random.random()
            if pick < self.params[0] or j == k:
                ui[j] = vi[j]
            else:
                ui[j] = self.population[i].chrom[j]
        return ui
