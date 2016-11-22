"""
Usage

{
'param1': [1e-3, 1e-2, 1e-2],
'param2': [1,5,10,20],
}

"""
import itertools

class Sweeper(object):
    def __init__(self, hyper_config):
        self.hyper_config = hyper_config

    def __iter__(self):
        for config in itertools.product(*[val for val in self.hyper_config.values()]):
            yield {key:config[i] for i, key in enumerate(self.hyper_config.keys())}


def example_run_method(param1=1., param2=2, param3=3, param4=4):
    print param1, param2, param3, param4


def run_sweep(run_method, params):
    sweeper = Sweeper(params)
    for config in sweeper:
        run_method(**config)

if __name__ == "__main__":
    sweep_op = {
        'param1': [1e-3, 1e-2, 1e-1],
        'param2': [1,5,10,20],
        'param3': [True, False]
    }
    run_sweep(example_run_method, sweep_op)
