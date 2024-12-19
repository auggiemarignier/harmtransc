import harmonic as hm
import numpy as np
from numpy.typing import NDArray
from harmonic.model_abstract import Model


def train_harmonic_model(
    train_samples: NDArray[np.float_],
    train_lnprob: NDArray[np.float_],
) -> Model:
    """
    Trains a model from harmonic
    """

    ndim = train_samples.shape[2]
    chains = hm.Chains(ndim)
    chains.add_chains_3d(train_samples, train_lnprob)

    # Using the default model for now
    model = hm.model.RealNVPModel(ndim, standardize=True, temperature=0.8)
    epochs_num = 20
    model.fit(chains.samples, epochs=epochs_num, verbose=True)

    return model
