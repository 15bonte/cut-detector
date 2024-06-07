"""Taken from https://github.com/TreB1eN/HiddenMarkovModel_Pytorch"""

import torch


class HiddenMarkovModel(object):
    """
    Hidden Markov self Class
    Parameters:
    -----------

    - N: Number of states.
    - T: numpy.array Transition matrix of size N by N
         stores probability from state i to state j.
    - E: numpy.array Emission matrix of size N by T (number of observations)
         stores the probability of observing  O_j  from state  S_i.
    - T0: numpy.array Initial state probabilities of size N.
    """

    def __init__(self, A, B, pi, epsilon=0.001, max_step=10):
        # Max number of iteration
        self.max_step = max_step
        # convergence criteria
        self.epsilon = epsilon
        # Number of possible states
        self.N = A.shape[0]
        # Number of possible observations
        self.M = B.shape[0]
        # Emission probability
        self.B = torch.tensor(B)
        # Transition matrix
        self.A = torch.tensor(A)
        # Initial state vector
        self.pi = torch.tensor(pi)
        # Used for later display
        self.prob_state_1 = []

    def belief_propagation(self, scores):
        return scores.view(-1, 1) + torch.log(self.A)

    def viterbi_inference(self, x):  # x: observing sequence
        """
        Be careful, all comments below are according to the paper and ignore the log operation
        """
        T = len(x)
        shape = [T, self.N]

        # Init_viterbi_variables
        psi = torch.zeros(shape, dtype=torch.float64)
        delta = torch.zeros_like(psi)
        q_star = torch.zeros([shape[0]], dtype=torch.int64)

        # log probability of emission sequence
        obs_prob_full = torch.log(
            self.B[x]
        )  # b_j(O_t) for all j for all t , shape (T, N)

        # initialize with state starting log-priors
        delta[0] = (
            torch.log(self.pi) + obs_prob_full[0]
        )  # pi_i * b_i(O_1) , shape (T)

        for step, obs_prob in enumerate(
            obs_prob_full[1:]
        ):  # note that first observation is ignored
            belief = self.belief_propagation(
                delta[step, :]
            )  # delta_(step)(i) * a_ij for all j
            # the inferred state by maximizing global function
            psi[step + 1] = torch.argmax(
                belief, 0
            )  # arg_max(delta_(step)(i) * a_ij for all j)
            # and update state and score matrices
            delta[step + 1] = (
                torch.max(belief, 0)[0] + obs_prob
            )  # max(delta_(step)(i) * a_ij for all j) * b_j(O_(step+1)), +1 since first observation is ignored

        # infer most likely last state
        q_star[T - 1] = torch.argmax(delta[T - 1, :], 0)

        for step in range(T - 1, 0, -1):
            # for every timestep retrieve inferred state
            state = q_star[step]
            state_prob = psi[step][state]
            q_star[step - 1] = state_prob

        return q_star, torch.exp(delta)  # turn scores back to probabilities
