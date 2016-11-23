import os
try:
    import imageio
except ImportError:
    print 'ImageIO not found - saving raw arrays for images instead'
    import imageio_proxy as imageio


from chaos_theory.run.sample import online_rollout, rollout, sample_seq
from chaos_theory.utils import print_stats, TBLogger, mkdir_p
from chaos_theory.utils.progressbar import progress_itr


def make_gif(expr_name, itr, images, fps=40):
    mkdir_p(os.path.join(expr_name, 'gifs'))
    imageio.mimwrite(os.path.join(expr_name, 'gifs', 'itr_%d.gif' % itr),
                     images, fps=fps)


def run_batch_algorithm(env, algorithm, max_length=500, alg_itrs=10000, samples_per_itr=10,
                        log_name=None, verbose_trial=-1):
    pol = algorithm.get_policy()

    logger = None
    if log_name:
        logger = TBLogger(log_name, {'rew', 'len'})

    n = 0
    for itr in range(alg_itrs):
        print '--' * 10, 'itr:', itr
        samps = sample_seq(env, pol, max_length=max_length, max_samples=samples_per_itr)

        for sample in samps:
            if log_name:
                logger.log(n, rew=sample.tot_rew, len=sample.T)
        algorithm.update(samps)

        print_stats(itr, pol, env, samps)
        if verbose_trial > 0:
            if itr % verbose_trial == 0 and itr > 0:
                traj, images = rollout(env, pol, max_length=max_length)
                if log_name:
                    make_gif(log_name, itr, images)


def run_online_algorithm(env, algorithm, samples_per_update=5, alg_itrs=10000,
                         max_length=500, log_name=None, verbose_trial=-1):
    pol = algorithm.get_policy()

    logger = None
    if log_name:
        logger = TBLogger(log_name, {'rew', 'len'})

    n = 0
    T = 0
    for itr in range(alg_itrs):
        print '--' * 10, 'itr:', itr

        samples = []
        for _ in progress_itr(range(samples_per_update)):
            pol.reset()
            sample = online_rollout(env, pol, algorithm, max_length=max_length)
            if logger:
                logger.log(n, rew=sample.tot_rew, len=sample.T)
            T += sample.T
            n += 1
            samples.append(sample)

        print_stats(itr, pol, env, samples)
        if verbose_trial > 0:
            if itr % verbose_trial == 0 and itr > 0:
                traj, images = rollout(env, pol, max_length=max_length)
                if log_name:
                    make_gif(log_name, itr, images)
