# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:59:25 2018

@author: Matthew

Class used to create neural network. Stores the values of the wieghts and
biases and can take any number of hidden layers.
"""

import numpy as np
import random

class Network:    
    
    def __init__(self, N_input, N_hidden_layers, N_output, functions = -1):
                
        
        self.nodes = [N_input] + N_hidden_layers + [N_output] 
        self.N = len(self.nodes)
        if functions == -1:
            self.functions = ["sigmoid" for i in range(self.N - 1)]
        else:
            self.functions = [0] + functions
        
  
        
        self.input = np.zeros(N_input)
        self.output = np.zeros(N_output)
        
        self.HL = [np.zeros(N_hidden_layers[i]) for i in range(len(N_hidden_layers))]
        
        self.layers = []
        self.layers.append(self.input)
        
        for layer in self.HL:            
            self.layers.append(layer)
            
        self.layers.append(self.output)
        
                
        self.act_func = {
        "sigmoid" : self.sigmoid,
        "softmax" : self.softmax,
        "relu" : self.relu
        }   
        self.back_func = {
        "sigmoid" : self.sigmoid_prime,
        "softmax" : self.softmax_prime,
        "relu" : self.relu_prime
        } 
              
        
        self.weights = [0]
        
        for i in range(len(self.layers) - 1):            
            #matrix for each layer
            w = np.zeros((self.nodes[i+1], self.nodes[i]))
            self.weights.append(w)
        
        self.biases = [0] + [np.zeros(N) for N in self.nodes[1:]]
        
        self.deltas= [0] + [0 * layer for layer in self.layers[1:]]
        self.Z = list(self.deltas)
        
        self.nue = 0.5
        
        
    def initialise_weights(self):
        
        for l in range(1, len(self.weights)):
            
            for j in range(len(self.weights[l])):
                
                self.biases[l][j] = random.uniform(-1, 1)
                
                for k in range(len(self.weights[l][j])):
                    
                    self.weights[l][j][k] = random.uniform(-1, 1)        
                
        
        
    def feed_forward(self, inp):
        
        try:
            length = len(inp)            
        except:
            print("ERROR\tFeed-Forward: expected list input")
        else:
            
            if length == self.nodes[0]:
                
                self.layers[0] = np.array(inp)
                
                for l in range(1, self.N):
                    
                    self.Z[l] = np.matmul(self.weights[l], self.layers[l-1]) \
                                                            + self.biases[l]
                                                            
                    self.layers[l] = self.act_func[self.functions[l]](self.Z[l])
                    
                    
                self.output = np.array(self.layers[-1])
            else:
                print("ERROR\tFeed-Forward: input list is not the same size as initialised")
            
            
            
    def __calc_deltas(self, inp):
        
        
        self.deltas[-1] = (self.layers[-1] - inp) * self.back_func[self.functions[-1]](self.layers[-1])
        
        
        for l in range(1, self.N - 1)[::-1]:
            #weights and zs are in layer l+1 but do not exist for layer 0
            #therefore are saved as l-1
            
            w = np.transpose(self.weights[l+1])
            sigma_prime = self.back_func[self.functions[l]](self.layers[l])
             
            self.deltas[l] = np.matmul(w, self.deltas[l+1]) * sigma_prime    
            
        self.__update_weights()
        
        
        
    def __calc_deltas_softmax(self, inp):
        
        
        self.deltas[-1] = self.softmax_prime(inp)
        
        
        for l in range(1, self.N - 1)[::-1]:
            #weights and zs are in layer l+1 but do not exist for layer 0
            #therefore are saved as l-1            
            w = np.transpose(self.weights[l+1])
            sigma_prime = self.back_func[self.functions[l]](self.layers[l])
             
            self.deltas[l] = np.matmul(w, self.deltas[l+1]) * sigma_prime    
            
        self.__update_weights()
        
    def __update_weights(self):
        
        for l in range(1, len(self.weights)):
            
            del_w = np.outer(self.deltas[l], self.layers[l-1])
            self.weights[l] = self.weights[l] - self.nue * del_w
            
            del_b = self.deltas[l]
            self.biases[l] = self.biases[l] - del_b
                      
        
    def backpropogate(self, inp):   
        
        try:
            length = len(inp)            
        except:
            print("ERROR\tBackpropogate: expected list input")
        else:
            
            if length == self.nodes[-1]:
                
                #backpropogate_algorithm
                if self.functions[-1] == "softmax":
                    self.__calc_deltas_softmax(inp)
                else:
                    self.__calc_deltas(inp)
                
            else:
                print("ERROR\tBackpropogate: input list is not the same size as initialised")
    
        
    def sigmoid(self, z):
        
        return 1 / (1 + np.exp(-z))
    
    def softmax(self, z):
        base = sum(np.exp(z))
        return np.exp(z) / base
    
    def relu(self, z):
        
        return 0
    
    def sigmoid_prime(self, a):
        
        return a * (1 - a)
    
    def softmax_prime(self, inp):
        
        delta = []
        
        for j in range(self.nodes[-1]):
            
            delta_j = 0
            
            for k in range(self.nodes[-1]):
                
                kron = 0
                
                if j == k:
                    kron = 1
                    
                add = (self.layers[-1][k] - inp[k]) * self.layers[-1][k] * (kron - self.layers[-1][j])
                delta_j = delta_j +add
            
            delta.append(delta_j)
            
        return np.array(delta)
        
    
    def relu_prime(self, z):
        
        return 0
        
        
if __name__ == "__main__" :
    
    function = ["sigmoid", "softmax"]
    brain = Network(3, [3], 2, function)
    brain.initialise_weights()
    
    count = 0
    
   
    
    while count < 50000:
        
        col = [random.random(), random.random(), random.random()]
        
        brain.feed_forward(col)
        
        if sum(col) > 1.5:
            brain.backpropogate([1,0])
        else:
            brain.backpropogate([0, 1])
        
        
        count = count + 1
        
    count = 0
    correct = 0
    while count < 500:
        
        col = [random.random(), random.random(), random.random()]
        
        brain.feed_forward(col)
        
        out = brain.layers[-1]
    
 
        if sum(col) > 1.5:
            if out[0] > out[1]:
                correct = correct + 1
            brain.backpropogate([1,0])
        else:
            if out[0] < out[1]:
                correct = correct + 1
            brain.backpropogate([0, 1])
            
        count = count + 1
        
            
    print(count, correct)
    print("Program End!")        
        
        
        