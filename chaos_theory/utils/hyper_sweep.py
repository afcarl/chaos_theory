"""
Usage

{
'param1': [1e-3, 1e-2, 1e-2],
'param2': [1,5,10,20],
}

"""
import itertools
import time
import inspect

class Sweeper(object):
    def __init__(self, hyper_config):
        self.hyper_config = hyper_config

    def __iter__(self):
        for config in itertools.product(*[val for val in self.hyper_config.values()]):
            yield {key:config[i] for i, key in enumerate(self.hyper_config.keys())}


def create_name(hyper_dict, use_time=True):
    ts = ''
    if use_time:
        ts = '_ts'+str(time.time())

    return ('_'.join([str(k)+str(v) for k,v in hyper_dict.iteritems()])) + ts


def example_run_method(param1=1., param2=2, param3=3, param4=4):
    print param1, param2, param3, param4


def run_sweep(run_method, params):
    sweeper = Sweeper(params)
    args, varargs, varkw, defaults = inspect.getargspec(run_method)
    for config in sweeper:
        if 'hyperparam_string' in args:
            config['hyperparam_string'] = create_name(config)
        run_method(**config)

if __name__ == "__main__":
    sweep_op = {
        'param1': [1e-3, 1e-2, 1e-1],
        'param2': [1,5,10,20],
        'param3': [True, False]
    }
    run_sweep(example_run_method, sweep_op)
