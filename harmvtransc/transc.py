import numpy as np
from numpy.typing import NDArray

from harmonic.model_abstract import Model


def uniform_proposal_probability(k_max: int) -> float:
    """
    q(k'|k) is uniform and independent of k
    """
    return 1 / k_max


class State:
    def __init__(
        self,
        samples: NDArray[np.float_],
        unnormed_lnprob: NDArray[np.float_],
        pusedo_prior: Model,
        prior_prob: float = 1.0,
    ):
        self.samples = samples
        self.unnormed_lnprob = unnormed_lnprob
        self.psuedo_prior = pusedo_prior
        if not 0 < prior_prob <= 1:
            raise ValueError("Prior probability must be between 0 and 1")
        self.ln_prior_prob = np.log(prior_prob)

        self.n, self.ndim = samples.shape

    def get_sample(self, idx: int) -> NDArray[np.float_]:
        return self.samples[idx]

    def get_unnormed_lnprob(self, idx: int) -> float:
        return self.unnormed_lnprob[idx]

    def __getitem__(self, idx: int) -> tuple[NDArray[np.float_], float]:
        return self.get_sample(idx), self.get_unnormed_lnprob(idx)

    def evaluate_psuedo_prior(self, m: NDArray[np.float_]) -> float:
        return self.psuedo_prior.predict(m)[0]


def propose_k(k_max: int) -> int:
    """
    Propose a new k' given k
    """
    return np.random.randint(0, k_max)


def draw_random_sample_from_state(state: State) -> tuple[NDArray[np.float_], float]:
    """
    Select a sample and associated unormalised logposterior from the state
    """
    idx = np.random.randint(0, state.n)
    return state[idx]


def transition_probability(
    state: State, m: NDArray[np.float_], lnp: float, kmax: int
) -> float:
    phi_m = state.evaluate_psuedo_prior(m)
    return lnp + state.ln_prior_prob - phi_m - uniform_proposal_probability(kmax)


def acceptance_ratio(
    state: State,
    statep: State,
    m: NDArray[np.float_],
    mp: NDArray[np.float_],
    lnp: float,
    lnpp: float,
    kmax: int,
) -> float:
    return transition_probability(statep, mp, lnpp, kmax) - transition_probability(
        state, m, lnp, kmax
    )


def metropolis_hastings(states: list[State]) -> NDArray[np.int_]:
    kmax = len(states)

    k_samples: list[int] = []

    k = propose_k(kmax)
    state = states[k]
    m, lnp = draw_random_sample_from_state(state)

    for _ in range(1000):
        kp = propose_k(kmax)
        statep = states[kp]
        mp, lnpp = draw_random_sample_from_state(statep)
        alpha = acceptance_ratio(state, statep, m, mp, lnp, lnpp, kmax)
        if np.log(np.random.rand()) < alpha:
            k_samples.append(kp)
            k = kp
            state = statep
            m = mp
            lnp = lnpp

    return np.array(k_samples)
