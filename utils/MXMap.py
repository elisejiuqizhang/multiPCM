import os

import numpy as np

import graphviz # need to be installed with conda not pip??

from utils.data_utils import time_delay_embed
from utils.causal_simplex import CM_simplex, PCM_simplex

# model structure inspired by RESIT: https://lingam.readthedocs.io/en/latest/_modules/lingam/resit.html#RESIT
# Phase 1: Causal order determination with bivariate convergent cross mapping (CM)
# Phase 2: Elimination of redundant edges with multivariate partial convergent cross mapping (multiPCM)

class MXMap:
    """ The class for the Multivar cross (X) MAPping model.
    Two-phase framework:
    1. Initial causal graph determination with cross mapping (CM) - do bivariate CM exhaustively on all the variables.
    2. Elimination of redundant edges with partial CM.
    
    Parameters:
    df: pandas DataFrame
        the data with each column as a variable.
    score_type: str, default 'err', determine which score to use for causal order determination and edge elimination.
        'err' for error, 'corr' for correlation. 
    tau: int, default 2
        time delay for time delay embedding.
    emd: int, default 8
        embedding dimension for time delay embedding.
    pcm_thres: float, default 0.4
        threshold for PCM, 
        If using correlation score: if threshold is smaller than the value, do not remove the edge;
                            else remove the edge.
        If using error score: if threshold is greater than the value, do not remove the edge;
                            else remove the edge.
    **kwargs: dict for additional CM and PCM parameters, whether to use PCA on embedding first 
    before doing kNN during cross mapping."""

    def __init__(self, df, score_type='corr', tau=2, emd=8, pcm_thres=0.5, **kwargs):
        
        self.df = df # the dataframe, extract the column names or indices, determine the causal graph
        self.kwargs = kwargs # dictionary of other parameters, including the CM and PCM parameters
        
        if score_type not in ['corr', 'err', 'r2']:
            raise ValueError('score_type must be either "corr" or "err" or "r2"')
        self.score_type = score_type

        self.tau = tau # time delay for time delay embedding
        self.emd = emd

        self.pcm_thres = pcm_thres # threshold for PCM edge removal

        self.n = df.shape[1] # number of variables
        self.var_names = df.columns # variable names
        self.var_indices = np.arange(self.n) # pool of variable indices

        self.adj_matrix = None # adjacency matrix of the causal graph

        # to save the score stats of the two phases for information
        self.phase1_stats = {}
        self.phase2_stats = {}

    
    def fit(self):
        """ Fit the multivarCM model.
        Returns:
        ch: dict
            the children of each variable.
"""
        ch=self._initial_causal_graph()
        ch=self._eliminate_edges(ch)
        self.ch=ch


        return ch
    
    def get_adj_matrix(self):
        """ Get the adjacency matrix of the causal graph. (to be called after fitting)
        Returns:
        adj_matrix: numpy array
            the adjacency matrix of the causal graph."""
        return self.adj_matrix

    def draw_graph(self, save_path=None):
        """ Draw the causal graph.
        Args:
        ch: dict
            the children of each variable."""
        ch=self.ch
        dot = graphviz.Digraph()
        for k in ch:
            for c in ch[k]:
                cause_name = self.var_names[k]
                effect_name = self.var_names[c]
                dot.edge(cause_name, effect_name)

        # view and save the graph
        if save_path is not None:
            dot.render(os.path.join(save_path, 'causal_graph'), format='png', view=True)

        return dot

    def _initial_causal_graph(self):
        # exhaustive bivariate search for initial causal graph (doesn't distinguish between direct and indirect)
        S=self.var_indices.copy() # pool of variable indices, start with all variables

        # initialize adjacency matrix
        self.adj_matrix=np.zeros((self.n,self.n))

        ch={} # dictionary to store the children of each variable
        
        sc_ratio_stats={}
        for i in range(self.n): # cause

            # initialize children of current var
            ch[i]=[]
            #  store the score stats
            sc_ratio_stats[i]={}

            # do not test redundant pairs
            for j in S[S>i]: # effect
                # cross map between the current variable i and the candidate child j
                # determine whether the edge between i and j is redundant
                cause_ind=[i]
                effect_ind=[j]
                cause_list=self.var_names[cause_ind].tolist()
                effect_list=self.var_names[effect_ind].tolist()

                # key in dictionary to store the stats ("cause -> effect | conds")
                phase1_stats_key=f'causes_{cause_ind} -> effect_{effect_ind}'

                # create the CM_simplex object
                cm=CM_simplex(self.df,cause_list,effect_list,self.tau,self.emd,**self.kwargs)
                output=cm.causality() # order: sc1_err, sc2_err, sc1_corr, sc2_corr, sc1_r2, sc2_r2


                del cm

                # store the stats
                self.phase1_stats[phase1_stats_key] = output

                if self.score_type=='err': 
                    ratio=output[0]/output[1]
                if self.score_type=='corr':
                    ratio=output[2]/output[3]
                if self.score_type=='r2':
                    ratio=output[4]/output[5]

                # store the ratio
                sc_ratio_stats[i][j]=ratio

        # link all the variable pairs if the ratio is greater than 1 (corr, r1) or smaller than 1 (err)
        for i in range(self.n):
            if self.score_type=='err':
                for j in S[S>i]:
                    if sc_ratio_stats[i][j]<1:
                        ch[i].append(j)
                        self.adj_matrix[i,j]=1
                    else:
                        ch[j].append(i)
                        self.adj_matrix[j,i]=1
            else: # if corr or r2
                for j in S[S>i]:
                    # access output from phase1 stats
                    output=self.phase1_stats[f'causes_{[i]} -> effect_{[j]}']
                    if sc_ratio_stats[i][j]>1:
                        # condition for not establishing the edge, if the score (corr or r2) is too small
                        if self.score_type=='corr':
                            if output[2]<0.5 and output[3]<0.5:
                                continue
                        if self.score_type=='r2':
                            if output[4]<0.5 and output[5]<0.5:
                                continue
                        ch[i].append(j)
                        self.adj_matrix[i,j]=1
                    else:
                        ch[j].append(i)
                        if self.score_type=='corr':
                            if output[2]<0.5 and output[3]<0.5:
                                continue
                        if self.score_type=='r2':
                            if output[4]<0.5 and output[5]<0.5:
                                continue
                        self.adj_matrix[j,i]=1

        return ch     
                
        
    def _eliminate_edges(self, ch):
        """ The second step: eliminate redundant edges.
        Use the score_type to determine which returned scores from PCM model to use.
        * Note my definition of ordering is from sink to top.
        
        For each pairwise edge that has other variables in between, this might be indirecto causality"""

        # Multi PCM
        list_to_remove=[] # will be used to store tuples of edges (cause, effect) to remove

        for i in range(self.n):
            for j in ch[i]: # children of i
                # check if a causal path (from adjacency matrix) can be established between i and j
                # if there is a path, do PCM to determine if it is a indirect causation
                    # if it is, remove the edge between i and j
                    # if it is not, keep the edge between i and j
                # if there is no path, keep the edge between i and j
                
                # bool, list_var_on_path = has_path(self.adj_matrix, i, j)
                bool, list_var_on_path = find_longest_path(self.adj_matrix, i, j)

                if bool:
                    # create the PCM object
                    # cause_list=self.var_names[[i]].tolist()
                    # effect_list=self.var_names[[j]].tolist()
                    # conds_list=[self.var_names[k] for k in list_var_on_path]

                    cause_ind=[i]
                    effect_ind=[j]
                    # conditions: list_var_on_path - i - j
                    conds_ind=[k for k in list_var_on_path if k!=i and k!=j]

                    # skip if there are no conditions
                    if len(conds_ind)==0:
                        continue

                    cause_list=self.var_names[cause_ind].tolist()
                    effect_list=self.var_names[effect_ind].tolist()
                    conds_list=self.var_names[conds_ind].tolist() 

                    # key in dictionary to store the stats ("cause -> effect | conds")
                    phase2_stats_key=f'causes_{cause_ind} -> effect_{effect_ind} | conds_{conds_ind}'

                    pcm=PCM_simplex(self.df,cause_list,effect_list,conds_list,self.tau,self.emd,**self.kwargs)
                    output=pcm.causality() # sc1_error, sc2_error, ratio_error, sc1_corr, sc2_corr, ratio_corr, sc1_r2, sc2_r2, ratio_r2
                    del pcm

                    # store the stats
                    self.phase2_stats[phase2_stats_key] = output


                    if self.score_type=='err':
                        if output[2]>self.pcm_thres:
                            list_to_remove.append((i,j))
                    if self.score_type=='corr':
                        if output[5]<self.pcm_thres:
                            list_to_remove.append((i,j))
                    if self.score_type=='r2':
                        if output[8]<self.pcm_thres:
                            list_to_remove.append((i,j))
                else:
                    continue

        # remove the edges
        for edge in list_to_remove:
            i,j=edge
            ch[i].remove(j)
            self.adj_matrix[i,j]=0

        return ch

def find_longest_path(adj_matrix, i, j):
    """Find the longest path of length >= 3 between two variables i and j.

    Args:
        adj_matrix: numpy array
            the adjacency matrix of the causal graph.
        i: int
            the index of the cause variable.
        j: int
            the index of the effect variable.
    
    Returns:
        bool: True if there is a path of length >= 3 between i and j, False otherwise.
        list: The longest path of nodes if a valid path exists, otherwise None.
    """
    n = adj_matrix.shape[0]
    stack = [(i, [i])]  # Stack stores tuples of (current node, path)
    longest_path = []  # Track the longest valid path
    
    while stack:
        node, path = stack.pop()

        if node == j and len(path) >= 3:  # Check if the current path is valid
            if len(path) > len(longest_path):  # Update if it's the longest valid path
                longest_path = path
        
        for k in range(n):
            if adj_matrix[node, k] == 1 and k not in path:  # Avoid revisiting nodes in the same path
                stack.append((k, path + [k]))  # Append the new path

    if longest_path:
        return True, longest_path  # Return the longest valid path
    else:
        return False, None  # No valid path found


