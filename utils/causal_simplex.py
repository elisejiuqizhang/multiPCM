import numpy as np

from sklearn.decomposition import PCA

from utils.data_utils import time_delay_embed, corr, partial_corr

# Cross Mapping (CM) and Partial CM for Causal Inference
# based on simplex projection between two embeddings 
# (inputs are embeddings not necessarily of the same dimensions)
# Insprired by: https://github.com/PrinceJavier/causal_ccm/blob/main/causal_ccm/causal_ccm.py


# 1. Cross Mapping
class CM_simplex:
    """ Cross mapping based on simplex projection (kNN) between two embeddings (inputs are embeddings not necessarily of the same dimensions)
        
    Inputs: 
        df: dataframe containing multivariate time series data, each column as a variable to be accessed by column name
        causes: list of namestrings of cause variables;
        effects: list of namestrings of effect variables;
        tau: time delay for time delay embedding;
        emd: embedding dimension for time delay embedding;

        knn (int): number of nearest neighbors to use for the simplex projection (default: 10

        L (int): the length of the time series to use (default: 1000)


        the first dimensions of M_cause and M_effect should be the same, representing the time indices;
        second dimensions can be different, representing the embedding dimensions.

    """

    def __init__(self, df, causes, effects, tau=2, emd=8, knn=10, L=3000, method='vanilla',**kwargs):
        self.df = df
        self.causes = causes
        self.effects = effects

        self.tau = tau
        self.emd = emd
        
        self.M_cause=CM_simplex._time_delay_embed(df[causes], tau, emd, L)
        self.M_effect=CM_simplex._time_delay_embed(df[effects], tau, emd, L)

        self.knn = knn

        self.method = method # 'vanilla' or 'PCA'

        self.kwargs = kwargs # dictionary of other parameters (PCA dims)

        assert self.M_cause.shape[0] == self.M_effect.shape[0], "The first dimensions of M_cause and M_effect should be the same, representing the time indices."

        self.model=CM_rep_simplex(cause_reps=self.M_cause, effect_reps=self.M_effect, knn=knn, L=L, method=method, **kwargs)

    def predict_manifolds(self):
        """ Cross Mapping Prediction:
        Reconstruct the manifolds of cause and effect.
        use kNN weighted average to get the reconstruction of the two manifolds, 
        return the two reconstructions.
        """
        return self.model.predict_manifolds()



    def causality(self):
        """ Causality score (error and averaged pearson correlation)"""
        return self.model.causality()

    @staticmethod
    def _time_delay_embed(df, tau, emd, L):
        """ Process the input dataframe to time delay embedding.
        Need to process each univariate time series one by one, then stack together.
        """
        embed = []
        for col in df.columns:
            ts = df[col].values
            embed.append(time_delay_embed(ts, tau, emd, L))
        embed = np.concatenate(embed, axis=1)
        return embed

    


    
# 2. Partial Cross Mapping
class PCM_simplex(CM_simplex):
    """ Partial Cross Mapping based on simplex projection (kNN) between two embeddings (inputs are embeddings not necessarily of the same dimensions)
        
    Inputs: 
        df: dataframe containing multivariate time series data, each column as a variable to be accessed by column name
        causes: list of namestrings of cause variables;
        effects: list of namestrings of effect variables;
        cond: list of namestrings of conditioning variables;

        tau: time delay for time delay embedding;
        emd: embedding dimension for time delay embedding;

        knn (int): number of nearest neighbors to use for the simplex projection (default: 10

        L (int): the length of the time series to use (default: 1000)


        the first dimensions of M_cause and M_effect should be the same, representing the time indices;
        second dimensions can be different, representing the embedding dimensions.

    
    -----------------------------

        The prediction procedure is modified: 
        1. Need 3 inputs: 
            cause, effect, condition

        2. According to the PCM paper, suppose X1->X_cond->X2,
            we don't know if there is direct causation between X1 and X2 only by CCM.
            Suppose cause = X1, effect = X2, condition = Xcond.
            
            To obtain: 
            "M_cause_reconst1" - the CM estimate of cause from effect;
            (first, "M_cond_reconst" - the CM estimate of M_cond from effect);
            then, "M_cause_reconst2" - the CM estimate of cause from "M_cond_reconst.

        3. Compute the partial correlation: ParCorr(X1, X1_reconst1 | X1_reconst2):
            Intuition is that now the information flow through the intermediate Xcond is eliminated,
            so if there is still a strong correlation between X1 and X1_reconst1,
            then X1 and X2 are directly causally related.
            The larger the ParCorr, the stronger the direct causation.


    """

    def __init__(self, df, causes, effects, cond, tau=2, emd=8, knn=10, L=3000, method='vanilla', **kwargs):
        super().__init__(df, causes, effects, tau, emd, knn, L, method, **kwargs)
        self.cond = cond
        self.M_cond = super()._time_delay_embed(df[cond], tau, emd, L)

        assert self.M_cause.shape[0] == self.M_effect.shape[0] == self.M_cond.shape[0], "The first dimensions of M_cause, M_effect, and M_cond should be the same, representing the time indices."

        self.model=PCM_rep_simplex(cause_reps=self.M_cause, effect_reps=self.M_effect, cond_reps=self.M_cond, knn=knn, L=L, method=method, **kwargs)

    def predict_manifolds(self):
        """ Partial Cross Mapping Prediction:
        Overriding the predict_manifolds() method in CM_simplex class.

        use kNN weighted average for reconstruction, 
        return the two reconstructions.
        """
        return self.model.predict_manifolds()
    
    def causality(self):
        """ Causality scores:
        Correlation based:
            1. direct correlation between M_cause and M_cause_reconst1;
            2. partial correlation between M_cause and M_cause_reconst1 given M_cause_reconst2.
            3. the ratio of ParCorr over DirectCorr.
            
        Error based:
            1. direct error between M_cause and M_cause_reconst1;
            2. indirect error between M_cause and M_cause_reconst2.
            3. the ratio of IndirectError over DirectError.
        """

        return self.model.causality()
    


# Utility 1: CM mapping between representations - either the delay embeddings or latent representations
class CM_rep_simplex:
    """ Cross mapping based on simplex projection (kNN) between two representations (inputs are not necessarily of the same dimensions)
        
    Inputs: 
        cause_reps: representation of cause variable;
        effect_reps: representation of effect variable;
        knn (int): number of nearest neighbors to use for the simplex projection (default: 10)

        method (str): the method to use for kNN search, either 'PCA' or 'vanilla' 

    """
    # mise a jour mardi 5 mars 2024 (reimplemented with PCA) - allow another input to determine whether to use PCA or vanilla kNN
    def __init__(self, cause_reps, effect_reps, knn=10, L=None, method='vanilla', **kwargs):
        self.M_cause = cause_reps[:]
        self.M_effect = effect_reps
        self.knn = knn
        self.L = L
        if L is not None:
            self.M_cause = self.M_cause[:L]
            self.M_effect = self.M_effect[:L]
        
        self.method = method
        self.kwargs = kwargs # dictionary of other parameters (PCA dims)
        
        assert self.method=='PCA' or self.method=='vanilla', "The method should be either 'PCA' or 'vanilla'."


        assert self.M_cause.shape[0] == self.M_effect.shape[0], "The first dimensions of cause_reps and effect_reps should be the same, representing the time indices."

    def predict_manifolds(self):
        """ Cross Mapping Prediction:  (mise a jour mardi 5 mars 2024 avec PCA)
        Reconstruct the manifolds of cause and effect.
        use kNN weighted average to get the reconstruction of the two manifolds, 
        return the two reconstructions.
        """
        self.M_cause_reconst=np.zeros(self.M_cause.shape)
        self.M_effect_reconst=np.zeros(self.M_effect.shape)
        
        if self.method=='vanilla':
            self.dists_cause=self.get_distance_vanilla(self.M_cause)
            self.dists_effect=self.get_distance_vanilla(self.M_effect)

            for t_tar in range(self.M_cause.shape[0]):
                # -------The cause manifold reconstruction from the effect -------
                # get the nearest distances of the target point t_tar on the effect manifold
                nearest_time_indices, nearest_distances=self.get_nearest_distances(self.dists_effect, t_tar, self.knn)
                # get the weights of the nearest neighbors on the cause manifold
                v=np.exp(-nearest_distances/np.max([1e-10, nearest_distances[0]]))
                w=v/np.sum(v)
                # get the reconstruction of the target point t_tar with corresponding points on the effect manifold
                self.M_cause_reconst[t_tar]=np.sum(w[:,np.newaxis]*self.M_cause[nearest_time_indices], axis=0)


                # -------The effect manifold reconstruction from the cause -------
                # get the nearest distances of the target point t_tar on the cause manifold
                nearest_time_indices, nearest_distances=self.get_nearest_distances(self.dists_cause, t_tar, self.knn)
                # get the weights of the nearest neighbors on the effect manifold
                v=np.exp(-nearest_distances/np.max([1e-10, nearest_distances[0]]))
                w=v/np.sum(v)
                # get the reconstruction of the target point t_tar with corresponding points on the cause manifold
                self.M_effect_reconst[t_tar]=np.sum(w[:,np.newaxis]*self.M_effect[nearest_time_indices], axis=0)

            return self.M_cause_reconst, self.M_effect_reconst

        elif self.method=='PCA':
            # use PCA to reduce the dimension of the representations
            n_comp=self.kwargs['pca_dim'] # PCA component

            # use PCA to reduce the dimension of the representations
            pca_cause=PCA(n_components=n_comp)
            pca_effect=PCA(n_components=n_comp)
            M_cause_pca=pca_cause.fit_transform(self.M_cause)
            M_effect_pca=pca_effect.fit_transform(self.M_effect)

            self.dists_cause=self.get_distance_vanilla(M_cause_pca)
            self.dists_effect=self.get_distance_vanilla(M_effect_pca)

            for t_tar in range(self.M_cause.shape[0]):
                # -------The cause manifold reconstruction from the effect -------
                # get the nearest distances of the target point t_tar on the effect manifold
                nearest_time_indices, nearest_distances=self.get_nearest_distances(self.dists_effect, t_tar, self.knn)
                # get the weights of the nearest neighbors on the cause manifold
                v=np.exp(-nearest_distances/np.max([1e-10, nearest_distances[0]]))
                w=v/np.sum(v)
                # get the reconstruction of the target point t_tar with corresponding points on the effect manifold
                self.M_cause_reconst[t_tar]=np.sum(w[:,np.newaxis]*self.M_cause[nearest_time_indices], axis=0)

                # -------The effect manifold reconstruction from the cause -------
                # get the nearest distances of the target point t_tar on the cause manifold
                nearest_time_indices, nearest_distances=self.get_nearest_distances(self.dists_cause, t_tar, self.knn)
                # get the weights of the nearest neighbors on the effect manifold
                v=np.exp(-nearest_distances/np.max([1e-10, nearest_distances[0]]))
                w=v/np.sum(v)
                # get the reconstruction of the target point t_tar with corresponding points on the cause manifold
                self.M_effect_reconst[t_tar]=np.sum(w[:,np.newaxis]*self.M_effect[nearest_time_indices], axis=0)

            return self.M_cause_reconst, self.M_effect_reconst





    


    def causality(self):
        """ vanilla kNN! 
        Causality score (error and averaged pearson correlation)"""
        # if the cause manifold reconstruction from effect is good, the cause->effect relationship is strong;
        # if the effect manifold reconstruction from cause is good, the effect->cause relationship is strong.
        

        M_cause_reconst, M_effect_reconst=self.predict_manifolds()

        # get the causality score (error)
        sc1_error=np.mean(np.sqrt(np.sum((self.M_cause-M_cause_reconst)**2, axis=1)))
        sc2_error=np.mean(np.sqrt(np.sum((self.M_effect-M_effect_reconst)**2, axis=1)))
        
        # get the causality score (pearson correlation) - average over each emd dimension
        sc1_corr=np.nanmean(np.abs(corr(self.M_cause, M_cause_reconst)))
        sc2_corr=np.nanmean(np.abs(corr(self.M_effect, M_effect_reconst)))

        # get the causality score (R2)
        sc1_r2=1-np.sum((self.M_cause-M_cause_reconst)**2)/np.sum((self.M_cause-np.mean(self.M_cause))**2)
        sc2_r2=1-np.sum((self.M_effect-M_effect_reconst)**2)/np.sum((self.M_effect-np.mean(self.M_effect))**2)

        return sc1_error, sc2_error, sc1_corr, sc2_corr, sc1_r2, sc2_r2
    
        


    @staticmethod
    def get_nearest_distances(distM, t_tar, knn=10):
        """ used for vanilla kNN!
        Get the nearest distances of the target point t_tar in distM.

        Input: (2D array)
            distM: Matrix of distances between each pair of points in M, (T_indices x T_indices) array
            t_tar: target time index
            knn: number of nearest neighbors to use for the simplex projection (default: 10)

        Output: (1D array)
            nearest_time_indices: time indices of the nearest neighbors
            nearest_distances: distances of the nearest neighbors in the same order
        """

        # get the distances of the target point t_tar to all other points
        dists=distM[t_tar]

        # get the nearest distances of the target point t_tar
        nearest_time_indices=np.argsort(dists)[1:knn+1]
        nearest_distances=dists[nearest_time_indices]

        return nearest_time_indices, nearest_distances

    

    @staticmethod
    def get_distance_vanilla(M):
        """ used for vanilla kNN!
        Calculate the distances between each pair of points in M.
        
        Input: (2D array)
            M: embedding of a variable, 2D array of shape (T_indices, embedding_dim)
            
        Output: (2D array)
            t_steps: time indices
            dists: distances between each pair of points in M, (T_indices x T_indices) array
        """

        # extract the temporal indices
        T_max=M.shape[0]

        # get the distances between each pair of points in M
        dists=np.zeros((T_max,T_max))

        # for i in range(T_max):
        #     for j in range(T_max):
        #         dists[i,j]=np.linalg.norm(M[i]-M[j])

        # more efficient loop - only compute half of the matrix, all the diagonal elements are 0
        for i in range(T_max):
            for j in range(i+1,T_max):
                dists[i,j]=np.linalg.norm(M[i]-M[j])
                dists[j,i]=dists[i,j]       

        return dists
        

    


# Utility 2: PCM mapping between representations - either the delay embeddings or latent representations
class PCM_rep_simplex(CM_rep_simplex):
    """ Partial Cross Mapping based on simplex projection (kNN) between two representations (inputs are not necessarily of the same dimensions)
        
    Inputs: 
        cause_reps: representation of cause variable;
        effect_reps: representation of effect variable;
        cond_reps: representation of conditioning variable;

        knn (int): number of nearest neighbors to use for the simplex projection (default: 10

    """
    def __init__(self, cause_reps, effect_reps, cond_reps, knn=10, L=None, method='vanilla',**kwargs):
        super().__init__(cause_reps, effect_reps, knn, L, method, **kwargs)
        self.M_cond = cond_reps
        if L is not None:
            self.M_cond = self.M_cond[:L]

        assert self.M_cause.shape[0] == self.M_effect.shape[0] == self.M_cond.shape[0], "The first dimensions of M_cause, M_effect, and M_cond should be the same, representing the time indices."

    def predict_manifolds(self):
        """ Partial Cross Mapping Prediction:
        Overriding the predict_manifolds() method in CM_simplex class.

        use kNN weighted average for reconstruction, 
        return the two reconstructions.
        """
        self.M_cause_reconst1=np.zeros(self.M_cause.shape) # direct reconstruction of cause from effect
        self.M_cause_reconst2=np.zeros(self.M_cause.shape) # indirect reconstruction of cause from M_cond_reconst
        self.M_cond_reconst=np.zeros(self.M_cond.shape)
        
        if self.method=='vanilla':
            self.dists_cause=self.get_distance_vanilla(self.M_cause)
            self.dists_effect=self.get_distance_vanilla(self.M_effect)
            self.dists_cond=self.get_distance_vanilla(self.M_cond)
            
            # starting from the effect, first map to reconstruct the condition and directly the cause
            for t_tar in range(self.M_effect.shape[0]):
                # -------The condition manifold reconstruction from the effect -------
                # get the nearest distances of the target point t_tar on the effect manifold
                nearest_time_indices, nearest_distances=self.get_nearest_distances(self.dists_effect, t_tar, self.knn)
                # get the weights of the nearest neighbors on the condition manifold
                v=np.exp(-nearest_distances/np.max([1e-10, nearest_distances[0]]))
                w=v/np.sum(v)
                # get the reconstruction of the target point t_tar with corresponding points on the effect manifold
                self.M_cond_reconst[t_tar]=np.sum(w[:,np.newaxis]*self.M_cond[nearest_time_indices], axis=0)

                # -------The cause manifold reconstruction from the effect -------
                # get the nearest distances of the target point t_tar on the effect manifold
                nearest_time_indices, nearest_distances=self.get_nearest_distances(self.dists_effect, t_tar, self.knn)
                # get the weights of the nearest neighbors on the cause manifold
                v=np.exp(-nearest_distances/np.max([1e-10, nearest_distances[0]]))
                w=v/np.sum(v)
                # get the reconstruction of the target point t_tar with corresponding points on the effect manifold
                self.M_cause_reconst1[t_tar]=np.sum(w[:,np.newaxis]*self.M_cause[nearest_time_indices], axis=0)    

            self.dists_cond_reconst=self.get_distance_vanilla(self.M_cond_reconst)

            # starting from the reconstructed condition, map to reconstruct the cause
            for t_tar in range(self.M_cond.shape[0]):
                # -------The cause manifold reconstruction from the reconstructed condition -------
                # get the nearest distances of the target point t_tar on the reconstructed condition manifold
                nearest_time_indices, nearest_distances=self.get_nearest_distances(self.dists_cond_reconst, t_tar, self.knn)
                # get the weights of the nearest neighbors on the cause manifold
                v=np.exp(-nearest_distances/np.max([1e-10, nearest_distances[0]]))
                w=v/np.sum(v)
                # get the reconstruction of the target point t_tar with corresponding points on the reconstructed condition manifold
                self.M_cause_reconst2[t_tar]=np.sum(w[:,np.newaxis]*self.M_cause[nearest_time_indices], axis=0)

            return self.M_cause_reconst1, self.M_cause_reconst2    

        elif self.method=='PCA':
            # use PCA to reduce the dimension of the representations
            n_comp=self.kwargs['pca_dim']
            
            # use PCA to reduce the dimension of the representations
            pca_cause=PCA(n_components=n_comp)
            pca_effect=PCA(n_components=n_comp)
            pca_cond=PCA(n_components=n_comp)
            M_cause_pca=pca_cause.fit_transform(self.M_cause)
            M_effect_pca=pca_effect.fit_transform(self.M_effect)
            M_cond_pca=pca_cond.fit_transform(self.M_cond)
            
            self.dists_cause=self.get_distance_vanilla(M_cause_pca)
            self.dists_effect=self.get_distance_vanilla(M_effect_pca)
            self.dists_cond=self.get_distance_vanilla(M_cond_pca)

            # starting from the effect, first map to reconstruct the condition and directly the cause
            for t_tar in range(self.M_effect.shape[0]):
                # -------The condition manifold reconstruction from the effect -------
                # get the nearest distances of the target point t_tar on the effect manifold
                nearest_time_indices, nearest_distances=self.get_nearest_distances(self.dists_effect, t_tar, self.knn)
                # get the weights of the nearest neighbors on the condition manifold
                v=np.exp(-nearest_distances/np.max([1e-10, nearest_distances[0]]))
                w=v/np.sum(v)
                # get the reconstruction of the target point t_tar with corresponding points on the effect manifold
                self.M_cond_reconst[t_tar]=np.sum(w[:,np.newaxis]*self.M_cond[nearest_time_indices], axis=0)

                # -------The cause manifold reconstruction from the effect -------
                # get the nearest distances of the target point t_tar on the effect manifold
                nearest_time_indices, nearest_distances=self.get_nearest_distances(self.dists_effect, t_tar, self.knn)
                # get the weights of the nearest neighbors on the cause manifold
                v=np.exp(-nearest_distances/np.max([1e-10, nearest_distances[0]]))
                w=v/np.sum(v)
                # get the reconstruction of the target point t_tar with corresponding points on the effect manifold
                self.M_cause_reconst1[t_tar]=np.sum(w[:,np.newaxis]*self.M_cause[nearest_time_indices], axis=0)

            self.dists_cond_reconst=self.get_distance_vanilla(self.M_cond_reconst)

            # starting from the reconstructed condition, map to reconstruct the cause
            for t_tar in range(self.M_cond.shape[0]):
                # -------The cause manifold reconstruction from the reconstructed condition -------
                # get the nearest distances of the target point t_tar on the reconstructed condition manifold
                nearest_time_indices, nearest_distances=self.get_nearest_distances(self.dists_cond_reconst, t_tar, self.knn)
                # get the weights of the nearest neighbors on the cause manifold
                v=np.exp(-nearest_distances/np.max([1e-10, nearest_distances[0]]))
                w=v/np.sum(v)
                # get the reconstruction of the target point t_tar with corresponding points on the reconstructed condition manifold
                self.M_cause_reconst2[t_tar]=np.sum(w[:,np.newaxis]*self.M_cause[nearest_time_indices], axis=0)

            return self.M_cause_reconst1, self.M_cause_reconst2

    
    def causality(self):
        """ Causality scores:
        Correlation based:
            1. direct correlation between M_cause and M_cause_reconst1;
            2. partial correlation between M_cause and M_cause_reconst1 given M_cause_reconst2.
            3. the ratio of ParCorr over DirectCorr.
            
        Error based:
            1. direct error between M_cause and M_cause_reconst1;
            2. indirect error between M_cause and M_cause_reconst2.
            3. the ratio of IndirectError over DirectError.
        """

        # get the reconstructions of the two manifolds
        M_cause_reconst1, M_cause_reconst2=self.predict_manifolds()

        # get the causality score (error)
        sc1_error=np.mean(np.sqrt(np.sum((self.M_cause-M_cause_reconst1)**2, axis=1)))
        sc2_error=np.mean(np.sqrt(np.sum((self.M_cause-M_cause_reconst2)**2, axis=1)))
        ratio_error=sc2_error/sc1_error

        # get the causality score (pearson correlation) - average over each emd dimension
        # direct correlation
        sc1_corr=corr(self.M_cause, M_cause_reconst1)
        sc1_corr=np.mean(np.abs(sc1_corr))
        # partial correlation conditioned on M_cause_reconst2
        sc2_corr=partial_corr(self.M_cause, M_cause_reconst1, M_cause_reconst2)
        sc2_corr=np.nanmean(np.abs(sc2_corr))
        ratio_corr=sc2_corr/sc1_corr

        # the causality score of r2
        sc1_r2=1-np.sum((self.M_cause-M_cause_reconst1)**2)/np.sum((self.M_cause-np.mean(self.M_cause))**2)
        sc2_r2=1-np.sum((self.M_cause-M_cause_reconst2)**2)/np.sum((self.M_cause-np.mean(self.M_cause))**2)
        ratio_r2=sc2_r2/sc1_r2

        return sc1_error, sc2_error, ratio_error, sc1_corr, sc2_corr, ratio_corr, sc1_r2, sc2_r2, ratio_r2
    



