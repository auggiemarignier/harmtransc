import harmonic as hm
from harmonic.model_abstract import Model
import numpy as np
from numpy.typing import NDArray


def get_harmonic_chains(
    samples: NDArray[np.float_],
    lnprob: NDArray[np.float_],
) -> hm.Chains:
    ndim = samples.shape[1]
    chains = hm.Chains(ndim)
    chains.add_chains_3d(samples, lnprob)
    return chains


def train_harmonic_model(
    chains_train: hm.Chains,
) -> Model:
    ndim = chains_train.ndim

    # Using the default model for now
    model = hm.model.RealNVPModel(ndim, standardize=True, temperature=0.8)
    epochs_num = 20
    model.fit(chains_train.samples, epochs=epochs_num, verbose=True)

    return model


def compute_harmonic_evidence(
    chains_infer: hm.Chains,
    model: Model,
) -> hm.Evidence:

    ev = hm.Evidence(chains_infer.nchains, model)
    ev.add_chains(chains_infer)
    return ev


def main():
    # load samples and lnprob
    ...

    chains = get_harmonic_chains(samples, lnprob)
    chains_train, chains_infer = hm.utils.split_data(chains, training_proportion=0.5)
    model = train_harmonic_model(chains_train)
    ev = compute_harmonic_evidence(chains_infer, model)

    print(
        f"ln inverse evidence (harmonic) = {ev.ln_evidence_inv} +/- {ev.compute_ln_inv_evidence_errors()}"
        )
