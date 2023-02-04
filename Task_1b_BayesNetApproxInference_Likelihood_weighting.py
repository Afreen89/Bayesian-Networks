#############################################################################
# BayesNetApproxInference.py
#
# This program implements the algorithm "Rejection Sampling", which
# imports functionalities to facilitate reading data of a Bayes net via
# the object self.bn created by the inherited class BayesNetReader.
# Its purpose is to answer probabilistic queries such as P(Y|X=true,Z=false).
# This implementation is agnostic of the data and provides a general
# implementation that can ne used across datasets by providing a config file.
#
# WARNINGS:
#    (1) This code has been revised but not thoroughly tested.
#    (2) The execution time depends on the number of random samples.
#
# Version: 1.0, Date: 07 October 2022, first version
# Version: 1.5, Date: 20 October 2022, revised version (made simpler)
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
import random
import BayesNetUtil as bnu
from BayesNetReader import BayesNetReader


class BayesNetApproxInference(BayesNetReader):
    query = {}
    prob_dist = {}
    seeds = {}
    num_samples = None

    def __init__(self):
        if len(sys.argv) != 4:
            print("USAGE> BayesNetApproxInference.py [your_config_file.txt] [query] [num_samples]")
            print("EXAMPLE> BayesNetApproxInference.py config-alarm.txt \"P(B|J=true,M=true)\" 10000")
        else:
            file_name = sys.argv[1]
            prob_query = sys.argv[2]
            self.num_samples = int(sys.argv[3])
            super().__init__(file_name)
            self.query = bnu.tokenise_query(prob_query)
            self.prob_dist = self.likelihood_weighting()
            print("probability_distribution="+str(self.prob_dist))


##########################################################3


    def likelihood_weighting(self):
        print("\nSTARTING Likelihood Weighting...")
        query_variable = self.query["query_var"]
        evidence = self.query["evidence"]
        W = {}

        print("random key values for our query variables:",self.bn["rv_key_values"][query_variable])

        # initialise vector of counts
        for value in self.bn["rv_key_values"][query_variable]:
            value = value.split("|")[0]
            W[value] = 0
            print("initial W vector with ",value,"is :", W)

        # loop to increase counts when the sampled vector consistent w/evidence
        for j in self.bn["rv_key_values"][query_variable]: #range(0, self.num_samples):
            print("++++++++++++++++++++++++++++++")
            print("for random variable: ", j)
            x, w = self.weighted_sample() #brings the list of values (true/false) and new adjusted weights
            j = j.split("|")[0]
            W[j] = W[j] + w #weight list being updated
            print("now W : ",W)

        print("____________________")
        print("W: ",W)
        return bnu.normalise(W)
        

    def weighted_sample(self):
        w = 1 #default weight value
        X = {} 
        sampled_var_value = {}      # counter
        evidence_variables = list(self.query["evidence"].keys())    #from query
        for variable in self.bn["random_variables"]:           # testing all the variables from config file
            print("solving for variable: ", variable)
            
            if self.is_in_evidance(evidence_variables, variable):  # if the variable is provided in the evidence...
                v = self.query["evidence"][variable]      # we get its value (true/false) from the evidence
                w = w*(bnu.get_probability_given_parents(variable, v, sampled_var_value, self.bn))
                # just multiplied weight from P(variable = value | parents)
                sampled_var_value[variable] = v
            else:
                #we get the random sampling done and add to X dictionary
                X[variable] = self.get_sampled_value(variable, sampled_var_value)
                sampled_var_value[variable] = X[variable]
        
        print("sampled_var_values dict: ", sampled_var_value)
        print("X: ",X)
        print("w = ", w)
        return X, w


    def is_in_evidance(self, k, var):
        for i in k:
            if i == var:
                return True
        return False

###########################################################


    def get_sampled_value(self, V, sampled):
        # get the conditional probability distribution (cpt) of variable V
        parents = bnu.get_parents(V, self.bn)
        cpt = {}
        prob_mass = 0

        # generate a cumulative distribution for random variable V
        if parents is None:
            for value, probability in self.bn["CPT("+V+")"].items():
                prob_mass += probability
                cpt[value] = prob_mass

        else:
            for v in bnu.get_domain_values(V, self.bn):
                p = bnu.get_probability_given_parents(V, v, sampled, self.bn)
                prob_mass += p
                cpt[v] = prob_mass

        # check that the cpt sums to 1 (or almost)
        if prob_mass < 0.999 and prob_mass > 1:
            print("ERROR: CPT=%s does not sum to 1" % (cpt))
            exit(0)

        return self.sampling_from_cumulative_distribution(cpt)

    def sampling_from_cumulative_distribution(self, cumulative):
        random_number = random.random()
        for value, probability in cumulative.items():
            if random_number <= probability:
                random_number = random.random()
                return value.split("|")[0]

        print("ERROR couldn't do sampling from:")
        print("cumulative_dist="+str(cumulative))
        exit(0)

    def is_compatible_with_evidence(self, X, evidence):
        for variable, value in evidence.items():
            if X[variable] != value:
                return False
        return True


BayesNetApproxInference()
