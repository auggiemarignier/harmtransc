import harmonic as hm
from harmonic.model_abstract import Model
import numpy as np
from numpy.typing import NDArray

from learning import train_harmonic_model


TRAINING_PROPORTION = 0.5


def compute_harmonic_evidence(
    inference_samples: NDArray[np.float_],
    inference_lnprob: NDArray[np.float_],
    model: Model,
) -> hm.Evidence:
    ndim = inference_samples.shape[2]
    chains_infer = hm.Chains(ndim)
    chains_infer.add_chains_3d(inference_samples, inference_lnprob)

    ev = hm.Evidence(chains_infer.nchains, model)
    ev.add_chains(chains_infer)
    return ev


def split_data(
    samples: NDArray[np.float_],
    lnprob: NDArray[np.float_],
    training_proportion: float = 0.5,
) -> tuple[
    NDArray[np.float_], NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]
]:

    nchains, nsamples, ndim = samples.shape
    ntraining = int(training_proportion * nsamples)
    training_samples = samples[:, :ntraining, :]
    training_lnprob = lnprob[:, :ntraining]
    inference_samples = samples[:, ntraining:, :]
    inference_lnprob = lnprob[:, ntraining:]
    return training_samples, training_lnprob, inference_samples, inference_lnprob


def main():
    # load samples and lnprob
    ...

    training_samples, training_lnprob, inference_samples, inference_lnprob = split_data(
        samples, lnprob, training_proportion=TRAINING_PROPORTION
    )
    model = train_harmonic_model(training_samples, training_lnprob)
    ev = compute_harmonic_evidence(inference_samples, inference_lnprob, model)

    print(
        f"ln inverse evidence (harmonic) = {ev.ln_evidence_inv} +/- {ev.compute_ln_inv_evidence_errors()}"
    )
