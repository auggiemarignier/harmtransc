from functools import partial
import jax.numpy as jnp
import numpy as np

from harmvtransc.sampling import perform_sampling
from harmvtransc.harmonic_estimator import (
    split_data,
    train_harmonic_model,
    compute_harmonic_evidence,
)


def ln_mvgaussian_posterior(x, inv_cov):
    """Compute log_e of posterior of n dimensional multivariate Gaussian.

    Args:

        x: Position at which to evaluate posterior.

    Returns:

        double: Value of posterior at x.

    """

    return -jnp.dot(x, jnp.dot(inv_cov, x)) / 2.0


def ln_analytic_evidence(ndim, cov):
    """Compute analytic evidence for nD Gaussian.

    Args:

        ndim: Dimension of Gaussian.

        cov: Covariance matrix.

    Returns:

        double: Analytic evidence.

    """

    ln_norm_lik = 0.5 * ndim * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(cov))
    return ln_norm_lik


COVARIANCES = [  # a bunch of nD Gaussians with unit covariance
    jnp.eye(5),
    jnp.eye(2),
    jnp.eye(3),
]


def main():
    for covariance in COVARIANCES:
        ln_posterior = partial(ln_mvgaussian_posterior, inv_cov=covariance)
        samples, lnprob = perform_sampling(ln_posterior)
        training_samples, training_lnprob, inference_samples, inference_lnprob = split_data(
            samples, lnprob
        )
        model = train_harmonic_model(training_samples, training_lnprob)
        ev = compute_harmonic_evidence(inference_samples, inference_lnprob, model)

        print(
            f"ln inverse evidence (harmonic) = {ev.ln_evidence_inv} +/- {ev.compute_ln_inv_evidence_errors()}"
        )
        print(
            f"ln analytic evidence = {ln_analytic_evidence(covariance.shape[0], covariance)}"
        )


if __name__ == "__main__":
    main()