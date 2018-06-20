import numpy as np


class Mutation:
    '''
    Mutation operation for differential evolution algorithm
    '''

    def __init__(self, vardim, bound ,sizepop ,population ,params):
        '''
        vardim: dimension of variables
        bound: boundaries of variables
        '''
        self.vardim = vardim
        self.bound = bound
        self.sizepop = sizepop
        self.population = population
        self.params = params

    def Random_3_different_current_mutation(self,i):
        '''
        A random selection of three individuals
        different from the current individual

        The three individuals Random evolution
        '''
        a = np.random.random_integers(0, self.sizepop - 1)
        while a == i:
            a = np.random.random_integers(0, self.sizepop - 1)
        b = np.random.random_integers(0, self.sizepop - 1)
        while b == i or b == a:
            b = np.random.random_integers(0, self.sizepop - 1)
        c = np.random.random_integers(0, self.sizepop - 1)
        while c == i or c == b or c == a:
            c = np.random.random_integers(0, self.sizepop - 1)
        vi = self.population[c].chrom + self.params[1] * \
             (self.population[a].chrom - self.population[b].chrom)
        for j in range(0, self.vardim):
            if vi[j] < self.bound[0, j]:
                vi[j] = self.bound[0, j]
            if vi[j] > self.bound[1, j]:
                vi[j] = self.bound[1, j]
        return vi
