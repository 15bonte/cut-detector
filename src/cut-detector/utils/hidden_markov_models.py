# from https://github.com/TreB1eN/HiddenMarkovModel_Pytorch
# cf https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm#Multiple_sequences for clean formula
# and http://www.cs.cmu.edu/~roni/11661/2017_fall_assignments/shen_tutorial.pdf
# and https://courses.grainger.illinois.edu/ECE417/fa2021/lectures/lec16.pdf
# and http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf
# or for alternative http://bozeman.genome.washington.edu/compbio/mbt599_2006/hmm_scaling_revised.pdf

import torch

from cnn_framework.utils.display_tools import display_progress


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
        obs_prob_full = torch.log(self.B[x])  # b_j(O_t) for all j for all t , shape (T, N)

        # initialize with state starting log-priors
        delta[0] = torch.log(self.pi) + obs_prob_full[0]  # pi_i * b_i(O_1) , shape (T)

        for step, obs_prob in enumerate(
            obs_prob_full[1:]
        ):  # note that first observation is ignored
            belief = self.belief_propagation(delta[step, :])  # delta_(step)(i) * a_ij for all j
            # the inferred state by maximizing global function
            psi[step + 1] = torch.argmax(belief, 0)  # arg_max(delta_(step)(i) * a_ij for all j)
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

    def _forward(self, obs_prob_seq):
        """
        obs_prob_seq: b_i(O_t) for all i for all N, shape (T, N)
        """
        T = len(obs_prob_seq)  # number of time stamps
        alpha = torch.zeros([T, self.N], dtype=torch.float64)
        scale_factors = torch.zeros([T], dtype=torch.float64)  # C in the paper
        # initialize with state starting priors
        alpha_0 = self.pi * obs_prob_seq[0]  # pi_i * b_i(O_0) for all i , shape (N)
        # scaling factor c_0 at t=0
        scale_factors[0] = 1.0 / alpha_0.sum()
        # scaled belief at t=0
        alpha[0] = scale_factors[0] * alpha_0
        # propagate belief
        for step, obs_prob in enumerate(obs_prob_seq[1:]):
            # previous state probability
            prev_prob = alpha[step].unsqueeze(0)  # alpha_step(i)
            # transition prior
            prior_prob = torch.matmul(prev_prob, self.A)  # sum(alpha_step(i) * a_ij for all i)
            # forward belief propagation
            forward_score = (
                prior_prob * obs_prob
            )  # sum(alpha_step(i) * a_ij for all i) * b_j(O_(step+1))
            forward_prob = torch.squeeze(forward_score)  # alpha_(step+1)(i)
            # scaling factor
            scale_factors[step + 1] = 1 / forward_prob.sum()
            # Update forward matrix
            alpha[step + 1] = scale_factors[step + 1] * forward_prob

        return alpha, scale_factors

    def _backward(self, obs_prob_seq_rev, scale_factors, beta_normalization):
        T = len(obs_prob_seq_rev)  # number of time stamps
        beta = torch.zeros([T, self.N], dtype=torch.float64)
        # initialize with state ending priors
        beta[0] = scale_factors[T - 1] * torch.ones(
            [self.N], dtype=torch.float64
        )  # initialize to scale_factors[T - 1]
        # propagate belief
        for step, obs_prob in enumerate(obs_prob_seq_rev[:-1]):
            # next state probability
            next_prob = beta[step, :].unsqueeze(1)  # beta_step(j)
            # observation emission probabilities
            obs_prob_d = torch.diag(obs_prob)
            # transition prior
            prior_prob = torch.matmul(self.A, obs_prob_d)  # sum(b_j(O_(step+1)) a_ij for all i)
            # backward belief propagation
            backward_prob = torch.matmul(
                prior_prob, next_prob
            ).squeeze()  # sum(b_j(O_(step+1)) a_ij for all i) * beta_step(j)
            # Update backward matrix
            scale_factor = (
                scale_factors[T - 2 - step]
                if beta_normalization == "original"
                else 1 / backward_prob.sum()
            )
            beta[step + 1] = (
                scale_factor * backward_prob  # from forward pass
            )  # beta_(step+1)(j) (should be -1 but flip in the end)
        beta = torch.flip(beta, [0, 1])
        return beta

    def forward_backward(self, obs_prob_seq, beta_normalization):
        """
        runs forward backward algorithm on observation sequence
        Arguments
        ---------
        - obs_prob_seq : matrix of size T by N, where T is number of timesteps and
            N is the number of states
        Returns
        -------
        - forward : matrix of size T by N representing
            the forward probability of each state at each time step
        - backward : matrix of size T by N representing
            the backward probability of each state at each time step
        - posterior : matrix of size T by N representing
            the posterior probability of each state at each time step
        """
        alpha, scale_factors = self._forward(obs_prob_seq)
        obs_prob_seq_rev = torch.flip(obs_prob_seq, [0, 1])
        beta = self._backward(obs_prob_seq_rev, scale_factors, beta_normalization)

        return alpha, beta

    def re_estimate_transition(self, observed_sequences, alphas, betas):
        summed_xis, gammas, updated_gammas = [], [], []

        for idx, (observed_seq, alpha, beta) in enumerate(zip(observed_sequences, alphas, betas)):
            T = len(observed_seq)  # number of time stamps
            xi = torch.zeros([T - 1, self.N, self.N], dtype=torch.float64)
            for t in range(T - 1):
                tmp_0 = torch.matmul(
                    alpha[t].unsqueeze(0), self.A
                )  # alpha_t(i) * a_ij for all i for all j, shape (1, N)
                tmp_1 = tmp_0 * self.B[observed_seq[t + 1]].unsqueeze(
                    0
                )  # alpha_t(i) * a_ij * b_j(O_(t+1)) for all i for all j, shape (1, N)
                denom = torch.matmul(
                    tmp_1, beta[t + 1].unsqueeze(1)
                ).squeeze()  # alpha_t(i) * a_ij * b_j(O_(t+1)) * beta_(t+1)(j) for all i for all j, scalar

                xi_t = torch.zeros([self.N, self.N], dtype=torch.float64)

                for i in range(self.N):
                    # alpha_t(i) * a_ij * b_j(O_(t+1)) * beta_(t+1)(j)
                    numer = alpha[t, i] * self.A[i, :] * self.B[observed_seq[t + 1]] * beta[t + 1]
                    xi_t[i] = numer / denom

                xi[t] = xi_t

            gamma = xi.sum(2).squeeze()

            # Updated gammas
            prod = (alpha[T - 1] * beta[T - 1]).unsqueeze(0)
            s = prod / prod.sum()
            updated_gamma = torch.cat([gamma, s], 0)
            if idx == 0:  # compute self.prob_state_1, only for future display
                self.prob_state_1.append(gamma[:, 0])

            # Store results for current sequence
            summed_xis.append(xi.sum(0))
            gammas.append(gamma)
            updated_gammas.append(updated_gamma)

        # Update A and pi according to formula 40a and 40b, and 109 for multiple observation sequences
        # Just sum up numerator and denominator and divide
        A_new = torch.stack(summed_xis).sum(0) / torch.stack(
            [gamma.sum(0).unsqueeze(1) for gamma in gammas]
        ).sum(0)

        pi_new = torch.stack([gamma[0, :] for gamma in gammas]).sum(0) / len(gammas)

        return pi_new, A_new, updated_gammas

    def re_estimate_emission(self, observed_sequences, updated_gammas):
        # Update B according to formula 40c
        states_marginal = torch.stack([gamma.sum(0) for gamma in updated_gammas]).sum(0)
        # One hot encoding buffer that you create out of the loop and just keep reusing
        emission_scores = []
        for observed_seq, gamma in zip(observed_sequences, updated_gammas):
            seq_one_hot = torch.zeros([len(observed_seq), self.M], dtype=torch.float64)
            seq_one_hot.scatter_(1, torch.tensor(observed_seq).unsqueeze(1), 1)
            local_emission_score = torch.matmul(seq_one_hot.transpose_(1, 0), gamma)
            emission_scores.append(local_emission_score)
        emission_score = torch.stack(emission_scores).sum(0)
        return emission_score / states_marginal

    def check_convergence(self, new_T0, new_transition, new_emission):
        delta_T0 = torch.max(torch.abs(self.pi - new_T0)).item() < self.epsilon
        delta_T = torch.max(torch.abs(self.A - new_transition)).item() < self.epsilon
        delta_E = torch.max(torch.abs(self.B - new_emission)).item() < self.epsilon

        return delta_T0 and delta_T and delta_E

    def expectation_maximization_step(self, obs_sequences, beta_normalization):
        # probability of emission sequence
        alphas, betas = [], []
        for obs_seq in obs_sequences:
            obs_prob_seq = self.B[obs_seq]
            alpha, beta = self.forward_backward(obs_prob_seq, beta_normalization)

            alphas.append(alpha)
            betas.append(beta)

        new_pi, new_A, updated_gammas = self.re_estimate_transition(obs_sequences, alphas, betas)

        new_B = self.re_estimate_emission(obs_sequences, updated_gammas)

        converged = self.check_convergence(new_pi, new_A, new_B)

        self.pi = new_pi
        self.B = new_B
        self.A = new_A

        return converged

    def Baum_Welch_EM(self, obs_sequences, beta_normalization="original"):
        converged = False
        for i in range(self.max_step):
            converged = self.expectation_maximization_step(obs_sequences, beta_normalization)
            display_progress("Baum-Welch optimization in progress", i + 1, self.max_step)
            if converged:
                print(f"\nConverged at step {i}")
                break

            print("\n------ Reconstructed model ------")
            print("Transition Matrix: ")
            print(self.A)
            print()
            print("Emission Matrix: ")
            print(self.B)
            print()
            print("Initial probabilities: ")
            print(self.pi)
            print()
            print("Reached Convergence: ")
            print(converged)

        # Return numpy arrays for easier display
        return (
            self.pi.detach().numpy(),
            self.A.detach().numpy(),
            self.B.detach().numpy(),
            converged,
        )
