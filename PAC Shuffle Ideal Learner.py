# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 20:32:33 2019

@author: Mike Silber
"""
import numpy as np
import pickle
from scipy.special import binom

# Alphabet (Sigma) size
s = 5



def P(L,n,s):
    return binom(n-1,L-1) * ((1/s)**(L-1)) * ((1-(1/s))**(n-L))


def check_sample_correctness_probability(epsilon, L):
    
    prob = 0.0
    
    with open(data_set_prob_name, 'rb') as data:
        prob = pickle.load(data)
    
    
    if prob <= 2*epsilon/3:
        # Returns hypothesis which classifies all examples negative
        print("All examples negative with current epsilon. Please call again with a lower value.")
        return False
    elif prob >= 1 - 2*epsilon/3:
        # Returns hypothesis which classifies all examples positive
        print("All examples positive with current epsilon. Please call again with a lower value.")
        return False
    return True
    
    
    
def in_span(u,x):
    N = len(x)
    L = len(u)
    curr_index = 0
    
    if L == 0:
        return True
    
    for i in range(N):
        if x[i] == u[curr_index]:
            curr_index+=1
        if curr_index == L:
            return True
            
    return False
    

def left_most_embedding(u,x):
    N = len(x)
    L = len(u)
    curr_index = 0
    
    if L == 0:
        return -1
    
    for i in range(N):
        if x[i] == u[curr_index]:
            curr_index+=1
        if curr_index == L:
            return i
            
    return 0





#########################################################################
###################### Statistical Query Algorithm ######################
#########################################################################
    
# Computes a single predicate value in a statistical query. The value for this
# predicate was defined in the paper.
def SQ(u, a, x, y):
    if in_span(u,x):
        index = left_most_embedding(u,x)+1
        
        if index >= len(x):
            return 0
        
        if a == int(x[index]):
            return y
        else:
            return -1*y/(s-1)
    else:
        return 0
    
    
# Computes a statistical query, that is computes the sample mean
# for the binary predicate Chi defined over 'u' and the symbol 'a'. We use this
# mean to understand in which case of the two expected values we are, thereby
# allowing us to know if a is the next symbol in u.
def compute_query(u, a):
    sample_set, dataset_size = get_sample_set()
    
    sample_mean = 0;
    for (x,y) in sample_set:
        sample_mean += SQ(u, a, x, y)
    sample_mean /= dataset_size
    
    return sample_mean
    

# This function is the one that actually infers our sample ideal generator.
# In every round of the outer for-loop a symbol of our generator 'u' is
# inferred. Runtime: O(Ls*(data-set size))
def SQ_algorithm(n, L, epsilon, tau):
    u = ""
    
    # In every round we infer the next symbol in u
    for round in range(L):
        if round > len(u):
            print("Error: no symbol found for last round.")
            return
        
        
        # We must check the expected value over all possible symbols in the alphabet
        for a in range(s):
            # Computes the sample expectation over our data set on the SQ over u and a
            mean = compute_query(u, a)
            if mean <= (2/s)*P(L,n,s)+tau and mean >= (2/s)*P(L,n,s)-tau:
                # a is the next symbol in u
                u = u + str(a)
                break
            elif mean <= -1*(2/(s*(s-1)))*P(L,n,s)+tau and mean <= -1*(2/(s*(s-1)))*P(L,n,s)+tau:
                # a is not the next symbol in u
                continue
            else:
                print("Error: Query mean wasn't in either possibilities of the expectation. Could be because of a small or incompitent data-set.")
                return
    print("u = %s" % u)
     
    
    
    
    
#########################################################################
############################## PAC Learner ##############################
#########################################################################       
            
# The main function from which the PAC learner begins its process
# It checks the sample probability to ensure correctness and then calls the
# SQ oracle which will determine shuffle ideal in O(Ls) runtime. 
def PAC_learn(n, L, epsilon, delta):
    
    # The tolerance, tau, for the statistical queries.
    tau = (2*epsilon)/(9*(s-1)*n)
    
    # Check that there is a likeliness of getting both a positive and a
    # negative sample in our data (Otherwise the algorithm will trivially 
    # classify all samples as positive or negative accordingly
    check_sample_correctness_probability(epsilon,L)
        
    # The core of the algorithm: the SQ oracle
    SQ_algorithm(n, L, epsilon, tau)
    
    
    
    
    
#########################################################################
###### Sampler ##### Here we assume we know 'u' in order to sample ######
#########################################################################
    
dataset_name = "5alphabet.pickle"
data_set_prob_name = "5alphabetprob.pickle"
n = 10
    
# Returns a uniformly distributed sample from Sigma**N
# along with a tag of whether it is a positive sample
# or a negative one. 
def sample(n):
    
    # Toggle the 'u' here in order to change the current target shuffle ideal.
    u = "012031"
    x = ""
    for i in range(n):
        # We sample x uniformly on each symbol in the alphabet.
        curr_symbol = np.random.randint(s)
        x = x + str(curr_symbol)
    
    # If x is spanned by u then it is a positive sample, otherwise we
    # return that it is a negative sample.
    if in_span(u,x):
        return (x,1)
    else:
        return (x,-1)
    
def get_sample_set():
    dataset = []
    
    with open(dataset_name, 'rb') as data:
        dataset = pickle.load(data)
        
    return (dataset, len(dataset))
    
def create_sample_set(size):
    dataset = []
    
    count = 0;
    for i in range(size):
        sample_tuple = sample(n)
        dataset.append(sample_tuple)
        if sample_tuple[1] == 1:
            count += 1
            
    prob_pos_sample = count / size
    
    with open(data_set_prob_name, 'wb') as output:
        pickle.dump(prob_pos_sample, output)
        
    
    with open(dataset_name, 'wb') as output:
        pickle.dump(dataset, output)

create_sample_set(100000)       
PAC_learn(10,6,0.5,0.5)