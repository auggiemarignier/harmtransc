from functools import partial
from pathlib import Path
from typing import Callable
import numpy as np
from numpy.typing import NDArray
import emcee

NCHAINS = 200
NDIM = 5
SAMPLES_PER_CHAIN = 5000
NBURN = 2000

OUTDIR = Path(__file__).parent.parent / "data" / "sampling"


def perform_sampling(
    ln_posterior: Callable[[NDArray[np.float_]], float],
    ndim=NDIM,
) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    sampler = emcee.EnsembleSampler(NCHAINS, ndim, ln_posterior)

    # initialise random seed
    np.random.seed(1)

    # Set initial random position and state
    pos = np.random.rand(ndim * NCHAINS).reshape((NCHAINS, ndim))
    rstate = np.random.get_state()
    (pos, prob, state) = sampler.run_mcmc(pos, SAMPLES_PER_CHAIN, rstate0=rstate)

    samples = np.ascontiguousarray(sampler.chain[:, NBURN:, :])
    lnprob = np.ascontiguousarray(sampler.lnprobability[:, NBURN:])
    return samples, lnprob


def main():
    def ln_mvgaussian_posterior(x, inv_cov):
        """Compute log_e of posterior of n dimensional multivariate Gaussian.

        Args:

            x: Position at which to evaluate posterior.

        Returns:

            double: Value of posterior at x.

        """

        return -np.dot(x, np.dot(inv_cov, x)) / 2.0

    inv_cov = np.eye(NDIM)

    # Instantiate and execute sampler
    ln_posterior = partial(ln_mvgaussian_posterior, inv_cov=inv_cov)

    samples, lnprob = perform_sampling(ln_posterior)

    # Save samples and lnprob
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTDIR / "samples.npy", samples)
    np.save(OUTDIR / "lnprob.npy", lnprob)


if __name__ == "__main__":
    main()
