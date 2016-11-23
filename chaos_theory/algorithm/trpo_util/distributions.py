from collections import OrderedDict

import numpy as np
import scipy
import scipy.optimize
import theano.tensor as T
import theano
concat = np.concatenate


from chaos_theory.algorithm.trpo_util.keras_theano_setup import floatX, FNOPTS
from chaos_theory.algorithm.trpo_util.trpo_util import unflatten, flatten, EzPickle, EzFlat, flatgrad


def categorical_sample(prob_nk):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_nk = np.asarray(prob_nk)
    assert prob_nk.ndim == 2
    N = prob_nk.shape[0]
    csprob_nk = np.cumsum(prob_nk, axis=1)
    return np.argmax(csprob_nk > np.random.rand(N,1), axis=1)

TINY = np.finfo(np.float64).tiny

def categorical_kl(p_nk, q_nk):
    p_nk = np.asarray(p_nk,dtype=np.float64)
    q_nk = np.asarray(q_nk,dtype=np.float64)
    ratio_nk = p_nk / (q_nk+TINY) # so we don't get warnings
    # next two lines give us zero when p_nk==q_nk==0 but inf when q_nk==0
    ratio_nk[p_nk==0] = 1
    ratio_nk[(q_nk==0) & (p_nk!=0)] = np.inf
    return (p_nk * np.log(ratio_nk)).sum(axis=1)

def categorical_entropy(p_nk):
    p_nk = np.asarray(p_nk,dtype=np.float64)
    p_nk = p_nk.copy()
    p_nk[p_nk == 0] = 1
    return (-p_nk * np.log(p_nk)).sum(axis=1)


class ProbType(object):
    def sampled_variable(self):
        raise NotImplementedError
    def prob_variable(self):
        raise NotImplementedError
    def likelihood(self, a, prob):
        raise NotImplementedError
    def loglikelihood(self, a, prob):
        raise NotImplementedError
    def kl(self, prob0, prob1):
        raise NotImplementedError
    def entropy(self, prob):
        raise NotImplementedError
    def maxprob(self, prob):
        raise NotImplementedError

class StochPolicy(object):
    @property
    def probtype(self):
        raise NotImplementedError
    @property
    def trainable_variables(self):
        raise NotImplementedError
    @property
    def input(self):
        raise NotImplementedError
    def get_output(self):
        raise NotImplementedError
    def act(self, ob, stochastic=True):
        prob = self._act_prob(ob[None])
        if stochastic:
            return self.probtype.sample(prob)[0], {"prob" : prob[0]}
        else:
            return self.probtype.maxprob(prob)[0], {"prob" : prob[0]}

    def probs(self, ob):
        prob_params = self._act_prob(ob[None])
        return prob_params[0]

    def finalize(self):
        self._act_prob = theano.function([self.input], self.get_output(), **FNOPTS)

class StochPolicyKeras(StochPolicy, EzPickle):
    def __init__(self, net, probtype):
        EzPickle.__init__(self, net, probtype)
        self._net = net
        self._probtype = probtype
        self.finalize()
    @property
    def probtype(self):
        return self._probtype
    @property
    def net(self):
        return self._net
    @property
    def trainable_variables(self):
        return self._net.trainable_weights
    @property
    def variables(self):
        return self._net.get_params()[0]
    @property
    def input(self):
        return self._net.input
    def get_output(self):
        return self._net.output
    def get_updates(self):
        self._net.output #pylint: disable=W0104
        return self._net.updates
    def get_flat(self):
        return flatten(self.net.get_weights())
    def set_from_flat(self, th):
        weights = self.net.get_weights()
        self._weight_shapes = [weight.shape for weight in weights]
        self.net.set_weights(unflatten(th, self._weight_shapes))

class Categorical(ProbType):
    def __init__(self, n):
        self.n = n
    def sampled_variable(self):
        return T.ivector('a')
    def prob_variable(self):
        return T.matrix('prob')
    def likelihood(self, a, prob):
        return prob[T.arange(prob.shape[0]), a]
    def loglikelihood(self, a, prob):
        return T.log(self.likelihood(a, prob))
    def kl(self, prob0, prob1):
        return (prob0 * T.log(prob0/prob1)).sum(axis=1)
    def entropy(self, prob0):
        return - (prob0 * T.log(prob0)).sum(axis=1)
    def sample(self, prob):
        return categorical_sample(prob)
    def maxprob(self, prob):
        return prob.argmax(axis=1)

class CategoricalOneHot(ProbType):
    def __init__(self, n):
        self.n = n
    def sampled_variable(self):
        return T.matrix('a')
    def prob_variable(self):
        return T.matrix('prob')
    def likelihood(self, a, prob):
        return (a * prob).sum(axis=1)
    def loglikelihood(self, a, prob):
        return T.log(self.likelihood(a, prob))
    def kl(self, prob0, prob1):
        return (prob0 * T.log(prob0/prob1)).sum(axis=1)
    def entropy(self, prob0):
        return - (prob0 * T.log(prob0)).sum(axis=1)
    def sample(self, prob):
        assert prob.ndim == 2
        inds = categorical_sample(prob)
        out = np.zeros_like(prob)
        out[np.arange(prob.shape[0]), inds] = 1
        return out
    def maxprob(self, prob):
        out = np.zeros_like(prob)
        out[prob.argmax(axis=1)] = 1

class DiagGauss(ProbType):
    def __init__(self, d):
        self.d = d
    def sampled_variable(self):
        return T.matrix('a')
    def prob_variable(self):
        return T.matrix('prob')
    def loglikelihood(self, a, prob):
        mean0 = prob[:,:self.d]
        std0 = prob[:, self.d:]
        # exp[ -(a - mu)^2/(2*sigma^2) ] / sqrt(2*pi*sigma^2)
        return - 0.5 * T.square((a - mean0) / std0).sum(axis=1) - 0.5 * T.log(2.0 * np.pi) * self.d - T.log(std0).sum(axis=1)
    def likelihood(self, a, prob):
        return T.exp(self.loglikelihood(a, prob))
    def kl(self, prob0, prob1):
        mean0 = prob0[:, :self.d]
        std0 = prob0[:, self.d:]
        mean1 = prob1[:, :self.d]
        std1 = prob1[:, self.d:]
        return T.log(std1 / std0).sum(axis=1) + ((T.square(std0) + T.square(mean0 - mean1)) / (2.0 * T.square(std1))).sum(axis=1) - 0.5 * self.d
    def entropy(self, prob):
        std_nd = prob[:, self.d:]
        return T.log(std_nd).sum(axis=1) + .5 * np.log(2 * np.pi * np.e) * self.d
    def sample(self, prob):
        mean_nd = prob[:, :self.d]
        std_nd = prob[:, self.d:]
        return np.random.randn(prob.shape[0], self.d).astype(floatX) * std_nd + mean_nd
    def maxprob(self, prob):
        return prob[:, :self.d]



class LbfgsOptimizer(EzFlat):
    def __init__(self, loss,  params, symb_args, extra_losses=None, maxiter=25):
        EzFlat.__init__(self, params)
        self.all_losses = OrderedDict()
        self.all_losses["loss"] = loss
        if extra_losses is not None:
            self.all_losses.update(extra_losses)
        self.f_lossgrad = theano.function(list(symb_args), [loss, flatgrad(loss, params)],**FNOPTS)
        self.f_losses = theano.function(symb_args, self.all_losses.values(),**FNOPTS)
        self.maxiter=maxiter

    def update(self, *args):
        thprev = self.get_params_flat()
        def lossandgrad(th):
            self.set_params_flat(th)
            l,g = self.f_lossgrad(*args)
            g = g.astype('float64')
            return (l,g)
        losses_before = self.f_losses(*args)
        theta, _, opt_info = scipy.optimize.fmin_l_bfgs_b(lossandgrad, thprev, maxiter=self.maxiter)
        del opt_info['grad']
        print opt_info
        self.set_params_flat(theta)
        losses_after = self.f_losses(*args)
        info = OrderedDict()
        for (name,lossbefore, lossafter) in zip(self.all_losses.keys(), losses_before, losses_after):
            info[name+"_before"] = lossbefore
            info[name+"_after"] = lossafter
        return info

class NnRegression(EzPickle):
    def __init__(self, net, mixfrac=1.0, maxiter=25):
        EzPickle.__init__(self, net, mixfrac, maxiter)
        self.net = net
        self.mixfrac = mixfrac

        x_nx = net.input
        self.predict = theano.function([x_nx], net.output, **FNOPTS)

        ypred_ny = net.output
        ytarg_ny = T.matrix("ytarg")
        var_list = net.trainable_weights
        l2 = 1e-3 * T.add(*[T.square(v).sum() for v in var_list])
        N = x_nx.shape[0]
        mse = T.sum(T.square(ytarg_ny - ypred_ny))/N
        symb_args = [x_nx, ytarg_ny]
        loss = mse + l2
        self.opt = LbfgsOptimizer(loss, var_list, symb_args, maxiter=maxiter, extra_losses={"mse":mse, "l2":l2})

    def fit(self, x_nx, ytarg_ny):
        nY = ytarg_ny.shape[1]
        ypredold_ny = self.predict(x_nx)
        out = self.opt.update(x_nx, ytarg_ny*self.mixfrac + ypredold_ny*(1-self.mixfrac))
        yprednew_ny = self.predict(x_nx)
        out["PredStdevBefore"] = ypredold_ny.std()
        out["PredStdevAfter"] = yprednew_ny.std()
        out["TargStdev"] = ytarg_ny.std()
        #if nY==1:
        #    out["EV_before"] =  explained_variance_2d(ypredold_ny, ytarg_ny)[0]
        #    out["EV_after"] =  explained_variance_2d(yprednew_ny, ytarg_ny)[0]
        #else:
        #    out["EV_avg"] = explained_variance(yprednew_ny.ravel(), ytarg_ny.ravel())
        return out


class NnVf(object):
    def __init__(self, net, timestep_limit, regression_params):
        self.reg = NnRegression(net, **regression_params)
        self.timestep_limit = timestep_limit
    def predict(self, path):
        ob_no = self.preproc(path["observation"])
        return self.reg.predict(ob_no)[:,0]
    def fit(self, paths):
        ob_no = concat([self.preproc(path["observation"]) for path in paths], axis=0)
        vtarg_n1 = concat([path["return"] for path in paths]).reshape(-1,1)
        return self.reg.fit(ob_no, vtarg_n1)
    def preproc(self, ob_no):
        return concat([ob_no, np.arange(len(ob_no)).reshape(-1,1) / float(self.timestep_limit)], axis=1)
