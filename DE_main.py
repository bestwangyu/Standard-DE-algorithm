import numpy as np
from Individual import Individual
from Mutation import Mutation
from Crossover import Crossover
from Selection import Selection
import copy
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']


class DifferentialEvolutionAlgorithm:

    '''
    主函数目录
    '''

    def __init__(self, sizepop, vardim, bound, MAXGEN, params):
        '''
        sizepop: population sizepop ：NP 种群大小
        vardim: dimension of variables ：DV:个体变量数
        bound: boundaries of variables :BO:边界上下限
        MAXGEN: termination condition ：G：最大迭代
        param: algorithm required parameters, it is a list which is consisting of [crossover rate CR, scaling factor F]  参数
        '''
        self.sizepop = sizepop
        self.MAXGEN = MAXGEN
        self.vardim = vardim
        self.bound = bound
        self.population = []  #note:里面放的是individual实体
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))
        self.params = params

    def initialize(self):
        '''
        initialize the population
        '''
        for i in range(0, self.sizepop):
            ind = Individual(self.vardim, self.bound)
            ind.generate()
            self.population.append(ind)

    def evaluate(self, x):
        '''
        evaluation of the population fitnesses
        '''
        x.calculateFitness()

    def solve(self):
        '''
        evolution process of differential evolution algorithm
        '''
        self.t = 0
        self.initialize()
        for i in range(0, self.sizepop):
            self.evaluate(self.population[i])
            self.fitness[i] = self.population[i].fitness
        best = np.max(self.fitness)
        bestIndex = np.argmax(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])
        self.avefitness = np.mean(self.fitness)
        self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
        self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
        print("当前代数 %d: 最优个体适应度值: %f; 当代平均适应度值 %f; " % (
            self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        # print("最有个体:")
        # print(self.best.chrom)
        while (self.t < self.MAXGEN - 1):
            self.t += 1
            for i in range(0, self.sizepop):
                # 遗传操作
                mutation = Mutation(self.vardim, self.bound ,self.sizepop ,self.population ,self.params)
                vi = mutation.Random_3_different_current_mutation(i)
                crossover = Crossover(self.vardim, self.population, self.params)
                ui = crossover.Standard_Crossover(i, vi)
                selection = Selection(self.population)
                xi_next = selection.selectionOperation(i, ui)
                self.population[i] = xi_next
            for i in range(0, self.sizepop):
                self.evaluate(self.population[i])
                self.fitness[i] = self.population[i].fitness
            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)
            if best > self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.fitness)
            self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
            self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
            print("当前代数 %d: 最优个体适应度值: %f; 当代平均适应度值 %f; " % (
                self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
            # print("最优个体:")
            # print(self.best.chrom)

        print("Optimal function value is: %f; " %
              self.trace[self.t, 0])
        print ("Optimal solution is:")
        print (self.best.chrom)
        self.printResult()


    def printResult(self):
        '''
        plot the result of the differential evolution algorithm
        '''
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        plt.plot(x, y1, 'r', label=u'最优值')
        plt.plot(x, y2, 'g', label=u'平均值')
        plt.xlabel(u"最大迭代",fontsize=15)
        plt.ylabel(u"目标函数",fontsize=15)
        plt.title(u"单目标标准差分进化算法",fontsize=15)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    bound = np.tile([[-600], [600]], 25)
    dea = DifferentialEvolutionAlgorithm(60, 25, bound, 1000, [0.8, 0.6])
    dea.solve()