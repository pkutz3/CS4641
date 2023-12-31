import numpy as np
from tqdm import tqdm
from kmeans import KMeans


SIGMA_CONST = 1e-6
LOG_CONST = 1e-32

FULL_MATRIX = False # Set False if the covariance matrix is a diagonal matrix

class GMM(object):
    def __init__(self, X, K, max_iters=100):  # No need to change
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters

        self.N = self.points.shape[0]  # number of observations
        self.D = self.points.shape[1]  # number of features
        self.K = K  # number of components/clusters

    # Helper function for you to implement
    def softmax(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        Hint:
            Add keepdims=True in your np.sum() function to avoid broadcast error. 
        """
        #find max values and subtract them from each row
        max_array = np.max(logit, axis=1).reshape(logit.shape[0],1)
        logit -= max_array
        #follow equation
        return np.exp(logit)/np.sum(np.exp(logit),axis=1,keepdims=True)


        #raise NotImplementedError

    def logsumexp(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        Hint:
            The keepdims parameter could be handy
        """
        #find max values and subtract them from each row
        max_array = np.max(logit, axis=1).reshape(logit.shape[0],1)
        logit -= max_array
        #follow formula and add maximums back
        s = (np.log(np.sum(np.exp(logit),axis=1,keepdims=True)))+max_array

        return s
        #raise NotImplementedError

    # for undergraduate student
    def normalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """
        pdf = np.ones(points.shape[0])
        d = len(mu_i)
        for i in range(d):
            var = sigma_i.diagonal()[i]
            part1 = 1/((2*np.pi*var)**.5)
            x_mu = points[:,i]-mu_i[i]
            exp = -((x_mu)**2)/(2*var)
            part2 = np.e**exp
            full = part1*part2
            pdf *= full
        
        return pdf
        #raise NotImplementedError

    # for grad students
    def multinormalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            1. np.linalg.det() and np.linalg.inv() should be handy.
            2. The value in self.D may be outdated and not correspond to the current dataset,
            try using another method involving the current arguments to get the value of D
        """

        raise NotImplementedError

    def _init_components(self, **kwargs):  # [5pts]
        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case
        """
        np.random.seed(5) #Do not remove this line!
        k = self.K
        d = self.D
        n = self.points.shape[0]
        #want every pi value to be 1/k
        pi = np.full(k,1/k)
        #pick k center indices from n possible indices and use these to select from points[]
        mu = self.points[np.random.choice(n,k,replace=True),:]
        #want k dxd identity matrices since diagonal
        sigma = np.array([np.eye(d) for _ in range(k)])
        return pi, mu, sigma
        #raise NotImplementedError

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):  # [10 pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.

        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """
        # === graduate implementation
        #if full_matrix is True:
            #...

        # === undergraduate implementation
        n = self.points.shape[0]
        k = self.K
        ll = np.zeros((n,k))
        if full_matrix is False:
            for curr_k in range(k):
                #get pdf for kth gaussian
                pdf = self.normalPDF(self.points, mu[curr_k], sigma[curr_k])
                #follow equation 
                ll[:,curr_k] = np.log(pi[curr_k]+1e-32) + np.log(pdf+1e-32)
        return ll

    def _E_step(self, pi, mu, sigma, full_matrix = FULL_MATRIX , **kwargs):  # [5pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        # === graduate implementation
        #if full_matrix is True:
            # ...

        # === undergraduate implementation
        gamma = None
        if full_matrix is False:
            #using helper function to make this simple
            gamma = self.softmax(self._ll_joint(pi, mu, sigma))
        return gamma

    def _M_step(self, gamma, full_matrix=FULL_MATRIX, **kwargs):  # [10pts]
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
            Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
        """
        # === graduate implementation
        #if full_matrix is True:
            # ...

        # === undergraduate implementation
        n = self.points.shape[0]
        k = self.K
        pi = np.zeros(k)
        mu = np.zeros((k,self.D))
        sigma = np.zeros((k,self.D,self.D))
        if full_matrix is False:
            #pretty self explanatory: follow equation for each
            for curr_k in range(k):
                pi[curr_k] = np.sum(gamma[:,curr_k])/np.sum(gamma)
                mu[curr_k] = np.sum(gamma[:,curr_k].reshape(n,1)*self.points,axis=0)/np.sum(gamma[:,curr_k])
                x_mu = self.points - mu[curr_k]
                gamma_x_mu = gamma[:,curr_k].T*x_mu.T
                sigma[curr_k] = np.dot(gamma_x_mu,x_mu) / np.sum(gamma[:,curr_k])
        #multiply by identity to get diagonal matrix
        sigma = sigma*np.eye(self.D)
        return pi, mu, sigma
        #raise NotImplementedError

    def __call__(self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs):  # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma, full_matrix)

            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)