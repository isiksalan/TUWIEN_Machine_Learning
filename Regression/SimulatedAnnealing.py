import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import math
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor


class SimulatedAny():
    def __init__(self, objective_function, parameter_ranges):
        self.objective_function = objective_function
        self.parameter_ranges = parameter_ranges
        self.candidate = {}
        self.candidate_score = 0
        self.candidate_index = {}
        self.current_candidate = {}
        self.current_candidate_score = 0
        self.temretature_list = []
        self.metropolis_list = []
        self.best_score = []
        self.best_candidate = []

        #inicialize solution
        for param in list(parameter_ranges.keys()):
            self.candidate_index[param] = np.random.randint(0,len(self.parameter_ranges[param]))
            self.candidate[param] = self.parameter_ranges[param][self.candidate_index[param]]


    def get_temperature(self,temp,n_iter):
        # n_iter - accual number of iteration
        # temp - initial temperature
        t = temp / float(n_iter + 1)
        return t

    def get_metropolis(self, diff):
        #diff - difference between candidate score and current score 
        
        metropol = math.exp(-10*diff / self.temretature_list[-1])
        self.metropolis_list.append(metropol)
        return metropol


    def run(self,n_of_iteration,step,temperature,pd_X,pd_Y):
        #set cuurent candidate as the best one
        self.best_candidate.append(self.candidate.copy())
        self.best_score.append(self.objective_function(self.candidate,pd_X,pd_Y).copy())
        self.candidate_score = self.best_score[-1].copy()


        for iteration in range(n_of_iteration):
            
            #create current candidate for solution
            self.current_candidate = self.candidate.copy()
            key_to_change = np.random.choice(list(self.parameter_ranges.keys()))
            value = np.random.choice(self.parameter_ranges[key_to_change])
            self.current_candidate[key_to_change] = value

            #evaluate the current candidate
            self.current_candidate_score = self.objective_function(self.current_candidate,pd_X,pd_Y).copy()

            #check if current candidate is the best now
            if self.current_candidate_score > self.best_score[-1]:
                self.best_score.append(self.current_candidate_score.copy())
                self.best_candidate.append(self.current_candidate.copy())

            else:
                self.best_score.append(self.best_score[-1].copy())
                self.best_candidate.append(self.best_candidate[-1].copy())

            #calculationg temperature
            current_temper = self.get_temperature(temperature,iteration)
            self.temretature_list.append(current_temper)


            diff = self.current_candidate_score - self.candidate_score
            
            current_metropolis = self.get_metropolis(diff)
            
            #check if the current candidate is worth to keep or not (based on metropolis function)
            if diff > 0 or float(np.random.rand()) < current_metropolis:
                self.candidate = self.current_candidate.copy()
                self.candidate_score = self.current_candidate_score
                self.candidate_index[key_to_change] = value

            print('Iteration {}/{} Current candi = {}, BEST SOLU is {} with {}'.format(iteration,n_of_iteration,self.current_candidate,self.best_candidate[-1],self.best_score[-1]),end='\r')

        return self.best_score, self.best_candidate, self.metropolis_list
