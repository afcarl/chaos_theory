import tensorflow as tf
import numpy as np

class TFGraphDeps(object):
    def __init__(self, g=None):
        if g is None:
            self.g = tf.get_default_graph()
        else:
            self.g = g
        self.deps = self._calc_dependency_graph()

    def _calc_dependency_graph(self):
        """Build a dependency graph.

        Returns:
            a dict. Each key is the name of a node (Tensor or Operation) and each value is a set of
            dependencies (other node names)
        """
        deps = defaultdict(set)
        for op in self.g.get_operations():
            # the op depends on its input tensors
            for input_tensor in op.inputs:
                deps[op].add(input_tensor)
            # the op depends on the output tensors of its control_dependency ops
            for control_op in op.control_inputs:
                for output_tensor in control_op.outputs:
                    deps[op].add(output_tensor)
            # the op's output tensors depend on the op
            for output_tensor in op.outputs:
                deps[output_tensor].add(op)
        return deps

    def ancestors(self, op):
        """Get all nodes upstream of the current node."""
        explored = set()
        queue = deque([op])
        while len(queue) != 0:
            current = queue.popleft()
            for parent in self.deps[current]:
                if parent in explored: continue
                explored.add(parent)
                queue.append(parent)
        return explored

def grad_params(out, param_list):
    return [tf.gradients(out, param)[0] for param in param_list]

def _total_size(shape):
    tot = 1
    for dim in shape:
        tot = tot*int(dim)
    return tot

def assert_shape(shape1, shape2):
    assert shape1.is_compatible_with(shape2)

class ParamFlatten(object):
    def __init__(self, param_list):
        """
        :param param_list: A list of tensorflow variables
        """
        self.param_list = param_list
        self.shapes = [tensor.get_shape() for tensor in param_list]
        self.sizes = [_total_size(shape) for shape in self.shapes]
        self.total_size = sum(self.sizes)

    def pack(self, param_vals):
        vec = []
        for i, val in enumerate(param_vals):
            val = np.array(val)
            assert_shape(self.shapes[i], val.shape)
            flat = np.reshape(val, [-1])
            vec.append(flat)
        vec = np.concatenate(vec)
        return vec

    def unpack(self, param_vec):
        assert param_vec.shape == (self.total_size,)
        cur_idx = 0
        unflat = []
        for i, size in enumerate(self.sizes):
            param = param_vec[cur_idx:cur_idx+size]
            cur_idx += size
            unflat.append(np.reshape(param, self.shapes[i]))
        return unflat

def entropy(p):
    return -np.sum(p*np.log(p+1e-10))

def pol_entropy(n, acts):
    histo = np.bincount(acts, minlength=n)
    n_act = float(len(acts))
    histo = histo/n_act
    assert len(histo) == n
    return entropy(histo)

def print_stats(env, samples):
    print '--'*10
    N = len(samples)
    R = np.array([samp.tot_rew for samp in samples])
    print 'Avg Rew:', np.mean(R), '+/-', np.sqrt(np.var(R))

    T = np.array([samp.T for samp in samples]).astype(np.float)
    avg_len = np.mean(T)
    stdev = np.sqrt(np.var(T))
    print 'Avg Len:', avg_len, '+/-', stdev
    print 'Num sam:', len(samples)
    
    #all_acts = np.concatenate([samp.act for samp in samples])
    #ent = pol_entropy(env.action_space.n, all_acts)
    #print 'Pol Entropy:', ent
    

def discount_rew(rew, gamma=0.99):
    """
    >>> r = np.ones(5)
    >>> discount_rew(r, gamma=0.5).tolist()
    [1.0, 0.5, 0.25, 0.125, 0.0625]
    """
    T = rew.shape[0]
    new_rew = np.zeros_like(rew)
    for i in range(T):
        new_rew[i] = rew[i]*gamma**i
    return new_rew


def discount_value(rew, gamma=0.99):
    values = np.zeros_like(rew)
    for t in range(len(rew)):
        values[t] = np.sum(discount_rew(rew[t:], gamma=gamma))
    return values





        
