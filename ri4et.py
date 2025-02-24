"""
Repeated Interactions for Electron Transfer using QuTiP
Authors: Matthew S. Teynor, Lea K. Northcote
"""

__version__ = '0.1.0'

import argparse

import numpy as np
import qutip
from qutip import (Qobj, Result, basis, concurrence, destroy, expect, fidelity,
                   ket2dm, mesolve, qeye, sigmam, sigmax, sigmaz, tensor)


class Simulation:
    def __init__(
        self,
        hamiltonian: list[Qobj] | Qobj,
        initial_state: Qobj,
        max_time: float,
        dt_output: float,
        collapse_ops: list[Qobj] = None,
        expect_ops: list[Qobj] = None,
        perturbation_index: int | None = None,
        interaction_ham: list[Qobj] | Qobj | None = None,
        ancilla_initial_state: Qobj | None = None,
        dt_ri: float | None = None,
        trotter_indexes: list[int] | None = None,
        num_trotter: int | None = None,
        fidelity_cutoff: float | None = None
    ):
        self.hamiltonian = hamiltonian
        self.initial_state = initial_state
        self.max_time = max_time
        self.dt_output = dt_output
        self.collapse_ops = collapse_ops if collapse_ops is not None else []
        self.expect_ops = expect_ops if expect_ops is not None else []
        self.perturbation_index = perturbation_index
        self.interaction_ham = interaction_ham
        self.ancilla_initial_state = ancilla_initial_state
        self.dt_ri = dt_ri
        self.trotter_indexes = trotter_indexes
        self.num_trotter = num_trotter
        self.fidelity_cutoff = fidelity_cutoff

        self.times = np.linspace(0, max_time, int(max_time / dt_output) + 1)
        self.mesolve_options = {'nsteps': 10000}

        self.csv_template = ','.join(
            ['{:0.6f}'] + len(expect_ops) * ['{:0.12e}'])
        self.csv_header = '#' + ','.join(
            ['time'] + [f'expect_{x}' for x in range(len(expect_ops))])

        print(f'# et_simulation.py version {
              __version__}. QuTiP version {qutip.__version__}.')

    def lindblad(self):
        if isinstance(self.hamiltonian, list):
            H = np.sum(self.hamiltonian)
        else:
            H = self.hamiltonian

        result = mesolve(
            H = H,
            rho0 = self.initial_state,
            tlist = self.times,
            c_ops = self.collapse_ops,
            e_ops = self.expect_ops,
            options = self.mesolve_options
        )
        self._print_result(result)

    def repeated_interactions(self):
        self._setup_repeated_interactions()

        print(self.csv_header)
        for k in range(int(self.max_time/self.dt_ri)):
            result = mesolve(self.H_total, self.state, self.times_steps, [],
                             self.expect_ops_total,
                             options = self.mesolve_options)

            if k == 0:
                print(self.csv_template.format(
                    0,
                    *[np.real(x[0]) for x in result.expect]
                ))
            current_time = (1 + k) * self.dt_ri
            if np.isclose(0, current_time % self.dt_output):
                print(self.csv_template.format(
                    current_time,
                    *[np.real(x[-1]) for x in result.expect]
                ))

            reduced_state = result.final_state.ptrace([0, 1])
            self.state = tensor(reduced_state, self.ancilla_initial_state)

    def repeated_interactions_trotter(self):
        if self.num_trotter is None or self.num_trotter <= 0:
            raise ValueError('--num_trotter must be greater than 0.')

        self._setup_repeated_interactions()

        ham_terms = [[tensor(x, qeye(2)) for x in self.hamiltonian],
                     self.interaction_ham / np.sqrt(self.dt_ri)]

        self.times_steps = self.times_steps / (2 * self.num_trotter)

        print(self.csv_header)
        print(self.csv_template.format(
            0,
            *[np.real(expect(e_op, self.state))
              for e_op in self.expect_ops_total]
        ))
        for k in range(int(self.max_time/self.dt_ri)):
            for _ in range(self.num_trotter):
                for i, j in self.trotter_indexes:
                    result = mesolve(ham_terms[i][j], self.state,
                                     self.times_steps, [], [],
                                     options=self.mesolve_options)
                    self.state = result.final_state
                for i, j in reversed(self.trotter_indexes):
                    result = mesolve(ham_terms[i][j], self.state,
                                     self.times_steps, [], [],
                                     options=self.mesolve_options)
                    self.state = result.final_state

            if np.isclose(0, ((1 + k) * self.dt_ri) % self.dt_output):
                print(self.csv_template.format(
                    (1 + k) * self.dt_ri,
                    *[np.real(expect(e_op, self.state))
                      for e_op in self.expect_ops_total]
                ))

            reduced_state = result.final_state.ptrace([0, 1])
            self.state = tensor(reduced_state, self.ancilla_initial_state)

    def repeated_interactions_state_prep(self):
        self._setup_repeated_interactions()
        self.H_total = (self.H_total - tensor(
            self.hamiltonian[self.perturbation_index], qeye(2)))

        state_prep_initial_state = tensor(
            *[basis(N, 0) for N in self.hamiltonian[0].dims[0]])
        self.state = tensor(ket2dm(state_prep_initial_state),
                            self.ancilla_initial_state)

        print(self.csv_header)
        for k in range(int(self.max_time/self.dt_ri)):
            result = mesolve(self.H_total, self.state, self.times_steps, [],
                             self.expect_ops_total,
                             options=self.mesolve_options)

            reduced_state = result.final_state.ptrace([0, 1])

            if k == 0:
                print(self.csv_template.format(
                    0,
                    *[np.real(x[0]) for x in result.expect],
                    fidelity(self.initial_state, reduced_state)
                ))
            current_time = (1 + k) * self.dt_ri
            if np.isclose(0, current_time % self.dt_output):
                print(self.csv_template.format(
                    current_time,
                    *[np.real(x[-1]) for x in result.expect],
                    fidelity(self.initial_state, reduced_state)
                ))

            self.state = tensor(reduced_state, self.ancilla_initial_state)

    def repeated_interactions_state_prep_continue(self):
        # Normal state prep
        self._setup_repeated_interactions()
        self.H_unperturbed = (self.H_total - tensor(
            self.hamiltonian[self.perturbation_index], qeye(2)))

        state_prep_initial_state = tensor(
            *[basis(N, 0) for N in self.hamiltonian[0].dims[0]])
        self.state = tensor(ket2dm(state_prep_initial_state),
                            self.ancilla_initial_state)

        print(self.csv_header)
        for k in range(int(self.max_time/self.dt_ri)):
            result = mesolve(self.H_unperturbed, self.state, self.times_steps,
                             [], self.expect_ops_total,
                             options=self.mesolve_options)

            reduced_state = result.final_state.ptrace([0, 1])

            if k == 0:
                print(self.csv_template.format(
                    -self.max_time,  # Start at -max_time
                    *[np.real(x[0]) for x in result.expect],
                    fidelity(self.initial_state, reduced_state)
                ))
            current_time = (1 + k) * self.dt_ri
            current_fidelity = fidelity(self.initial_state, reduced_state)
            if np.isclose(0, current_time % self.dt_output):
                print(self.csv_template.format(
                    current_time - self.max_time,  # Start at -max_time
                    *[np.real(x[-1]) for x in result.expect],
                    current_fidelity
                ))

            self.state = tensor(reduced_state, self.ancilla_initial_state)

            # When fidelity reaches the cutoff break and go to normal dynamics
            if current_fidelity >= self.fidelity_cutoff:
                break
        else:
            raise ValueError('fidelity_cutoff was not reached by max_time')

        # Normal repeated interactions
        for k in range(int(self.max_time/self.dt_ri)):
            result = mesolve(self.H_total, self.state, self.times_steps, [],
                             self.expect_ops_total,
                             options=self.mesolve_options)

            if k == 0:
                print(self.csv_template.format(
                    0,
                    *[np.real(x[0]) for x in result.expect],
                    0
                ))
            current_time = (1 + k) * self.dt_ri
            if np.isclose(0, current_time % self.dt_output):
                print(self.csv_template.format(
                    current_time,
                    *[np.real(x[-1]) for x in result.expect],
                    0
                ))

            reduced_state = result.final_state.ptrace([0, 1])
            self.state = tensor(reduced_state, self.ancilla_initial_state)

    def rhp_measure_lindblad(self):
        """
        Reference:
        Entanglement and Non-Markovianity of Quantum Evolutions
        Ángel Rivas, Susana F. Huelga, and Martin B. Plenio
        Phys. Rev. Lett. 105, 050403
        10.1103/PhysRevLett.105.050403
        """
        if isinstance(self.hamiltonian, list):
            H = np.sum(self.hamiltonian)
        else:
            H = self.hamiltonian

        dim = H.dims[0][0]
        ho_dim = H.dims[0][1]

        H = tensor(H, qeye(dim))
        self.expect_ops = [tensor(x, qeye(dim)) for x in self.expect_ops]
        self.collapse_ops = [tensor(x, qeye(dim)) for x in self.collapse_ops]

        ho_init_state = self.initial_state.ptrace([1])
        ho_init_state = tensor(qeye(dim), ho_init_state, qeye(dim))
        max_entangled_state = np.sum(
            [tensor(basis(dim, i), basis(dim, i)) for i in range(dim)]).unit()
        max_entangled_state = tensor(
            ket2dm(max_entangled_state), qeye(ho_dim)).permute([0, 2, 1])
        self.initial_state = max_entangled_state * ho_init_state

        self.mesolve_options['store_states'] = True

        result = mesolve(
            H=H,
            rho0=self.initial_state,
            tlist=self.times,
            c_ops=self.collapse_ops,
            e_ops=self.expect_ops,
            options=self.mesolve_options
        )

        bipartite_states = [state.ptrace([0, 2]) for state in result.states]
        concurrences = np.array([concurrence(state)
                                for state in bipartite_states])

        print(self.csv_header)
        for row in np.column_stack((result.times, *result.expect,
                                    concurrences)):
            print(self.csv_template.format(*row))

    def rhp_measure_repeated_interactions(self):
        """
        Reference:
        Entanglement and Non-Markovianity of Quantum Evolutions
        Ángel Rivas, Susana F. Huelga, and Martin B. Plenio
        Phys. Rev. Lett. 105, 050403
        10.1103/PhysRevLett.105.050403
        """
        self._setup_repeated_interactions()

        dim = self.H_total.dims[0][0]
        ho_dim = self.H_total.dims[0][1]

        self.H_total = tensor(self.H_total, qeye(dim))
        self.expect_ops_total = [tensor(x, qeye(dim))
                                 for x in self.expect_ops_total]

        ho_ri_ancilla_init_state = self.state.ptrace([1, 2])
        max_entangled_state = np.sum(
            [tensor(basis(dim, i), basis(dim, i)) for i in range(dim)]).unit()
        self.state = tensor(ket2dm(max_entangled_state),
                            ho_ri_ancilla_init_state).permute([0, 2, 3, 1])

        print(self.csv_header)
        for k in range(int(self.max_time/self.dt_ri)):
            result = mesolve(self.H_total, self.state, self.times_steps, [],
                             self.expect_ops_total,
                             options=self.mesolve_options)
            if k == 0:
                print(self.csv_template.format(
                    0,
                    *[np.real(x[0]) for x in result.expect],
                    1
                ))
            current_time = (1 + k) * self.dt_ri
            if np.isclose(0, current_time % self.dt_output):
                print(self.csv_template.format(
                    current_time,
                    *[np.real(x[-1]) for x in result.expect],
                    concurrence(result.final_state.ptrace([0, 3]))
                ))
            reduced_state = result.final_state.ptrace([0, 1, 3])
            self.state = tensor(
                reduced_state,
                self.ancilla_initial_state).permute([0, 1, 3, 2])

    def _setup_repeated_interactions(self):
        if self.dt_ri is None:
            raise ValueError(
                'Must set --dt_ri with --job_type repeated_interactions.')
        if self.dt_output < self.dt_ri:
            raise ValueError('--dt_output cannot be less than --dt_ri')
        if not np.isclose(int(self.dt_output / self.dt_ri),
                          self.dt_output / self.dt_ri):
            raise ValueError('--dt_output should be divisible by --dt_ri')

        # Set total Hamiltonian
        if isinstance(self.hamiltonian, list):
            H = np.sum(self.hamiltonian)
        else:
            H = self.hamiltonian
        H = tensor(H, qeye(2))
        if isinstance(self.interaction_ham, list):
            H_int = np.sum(self.interaction_ham)
        else:
            H_int = self.interaction_ham
        self.H_total = H + (H_int / np.sqrt(self.dt_ri))

        # Make e_ops compatible with tensored Hilbert space
        self.expect_ops_total = [tensor(x, qeye(2)) for x in self.expect_ops]
        self.state = tensor(self.initial_state, self.ancilla_initial_state)

        # Set time steps passed to qutip.mesolve
        self.times_steps = np.array([0, self.dt_ri])

        self.mesolve_options["store_states"] = True

    def _print_result(self, result: Result):
        print(self.csv_header)
        for row in np.column_stack((result.times, *result.expect)):
            print(self.csv_template.format(*row))


class ETSimulation(Simulation):
    def __init__(
        self,
        job_type: str,
        V: float,
        lamda: float,
        kT: float,
        gamma: float,
        dE: float,
        N: float,
        max_time: float,
        dt_output: float,
        dt_ri: float | None = None,
        num_trotter: int | None = None,
        fidelity_cutoff: float | None = None
    ):
        local_vars = locals()
        print("#", {x: local_vars[x] for x in local_vars.keys() if (
            x != 'self' and x != '__class__')})

        nu = np.sqrt(lamda)
        beta = 1 / kT
        nbar = (np.exp(beta) - 1)**-1  # Eq 3.308, Breuer & Petruccione

        # Base Operators
        a = tensor(qeye(2), destroy(N))
        sx = tensor(sigmax(), qeye(N))
        sz = tensor(sigmaz(), qeye(N))
        ho_q = (a.dag() + a) / 2
        ho_p = 1j * (a.dag() - a) / 2
        proj_donor = tensor(ket2dm(basis(2, 0)), qeye(N))

        # Hamiltonian
        h1 = a.dag() * a
        h2 = (dE / 2) * sz
        h3 = V * sx
        h4 = nu * sz * ho_q
        hamiltonian = [h1, h2, h3, h4]

        # Initial state, |D><D| * exp(-beta * <D|H|D>)
        d_H_d = (proj_donor * np.sum(hamiltonian)).ptrace([1])

        initial_state = tensor(ket2dm(basis(2, 0)),
                               (-beta * d_H_d).expm()).unit()

        # Collapse operators
        # delta_q is the shift of the position of the potential energy
        # minimum for the diabatic donor and acceptor states from 0
        delta_q = 0.5 * nu
        relax = np.sqrt(gamma * (1 + nbar)) * (a + sz * delta_q)
        excite = np.sqrt(gamma * nbar) * (a.dag() + sz * delta_q)
        collapse_ops = [relax, excite]

        # Expect operators
        expect_ops = [proj_donor, ho_q, ho_p, a.dag() * a]

        if 'repeated_interactions' in job_type:
            B = tensor(qeye(2), qeye(N), sigmam())
            S = (np.sqrt(gamma * (2 * nbar + 1)) *
                 (tensor(a + sz * delta_q, qeye(2))))
            interaction_ham = [S * B.dag(), S.dag() * B]
            ancilla_initial_state = (nbar * basis(2, 0).proj() +
                                     (nbar + 1) * basis(2, 1).proj()).unit()
        else:
            interaction_ham = None
            ancilla_initial_state = None

        super().__init__(
            hamiltonian=hamiltonian,
            initial_state=initial_state,
            max_time=max_time,
            dt_output=dt_output,
            collapse_ops=collapse_ops,
            expect_ops=expect_ops,
            perturbation_index=2,
            interaction_ham=interaction_ham,
            ancilla_initial_state=ancilla_initial_state,
            dt_ri=dt_ri,
            num_trotter=num_trotter,
            fidelity_cutoff=fidelity_cutoff
        )

        if 'trotter' in job_type:
            # (0, i) for H_system and (1, i) for H_int
            ham_magnitudes = [
                (1, (0, 0)),
                (dE / 2, (0, 1)),
                (V, (0, 2)),
                (nu, (0, 3)),
                (np.sqrt(gamma * (2 * nbar + 1)), (1, 0)),
                (np.sqrt(gamma * (2 * nbar + 1)), (1, 1))
            ]
            self.trotter_indexes = [x for _, x in sorted(
                ham_magnitudes, key=lambda pair: pair[0], reverse=True)]

        self.csv_header = ('#time,donor_population,ho_position,ho_momentum' +
                           ',ho_number')
        if 'state_prep' in job_type:
            self.csv_header += ',fidelity'
            self.csv_template += ',{:0.12e}'
        if 'rhp_measure' in job_type:
            self.csv_header += ',concurrence'
            self.csv_template += ',{:0.12e}'


class DBASimulation(Simulation):
    def __init__(
        self,
        job_type: str,
        V: float,
        kT: float,
        gamma: float,
        energies: list[float],
        positions: list[float],
        N: float,
        max_time: float,
        dt_output: float,
        dt_ri: float | None = None,
        num_trotter: int | None = None,
        fidelity_cutoff: float | None = None
    ):
        local_vars = locals()
        print("#", {x: local_vars[x] for x in local_vars.keys() if (
            x != 'self' and x != '__class__')})

        beta = 1 / kT
        nbar = (np.exp(beta) - 1)**-1  # Eq 3.308, Breuer & Petruccione

        # Base Operators
        a = tensor(qeye(4), destroy(N))
        ho_q = (a.dag() + a) / 2
        ho_p = 1j * (a.dag() - a) / 2
        proj_donor = tensor(basis(4, 0).proj(), qeye(N))
        proj_bridge_1 = tensor(basis(4, 1).proj(), qeye(N))
        proj_bridge_2 = tensor(basis(4, 2).proj(), qeye(N))
        proj_acceptor = tensor(basis(4, 3).proj(), qeye(N))
        coup_d_b1 = tensor(basis(4, 0) * basis(4, 1).dag() +
                           basis(4, 1) * basis(4, 0).dag(), qeye(N))
        coup_b1_b2 = tensor(basis(4, 1) * basis(4, 2).dag() +
                            basis(4, 2) * basis(4, 1).dag(), qeye(N))
        coup_b2_a = tensor(basis(4, 2) * basis(4, 3).dag() +
                           basis(4, 3) * basis(4, 2).dag(), qeye(N))

        # Hamiltonian
        hamiltonian = []
        hamiltonian.append(a.dag() * a)
        hamiltonian.append(energies[0] * proj_donor)
        hamiltonian.append(energies[1] * proj_bridge_1)
        hamiltonian.append(energies[2] * proj_bridge_2)
        hamiltonian.append(energies[3] * proj_acceptor)
        hamiltonian.append(V * coup_d_b1)
        hamiltonian.append(V * coup_b1_b2)
        hamiltonian.append(V * coup_b2_a)
        hamiltonian.append(-2 * positions[0] * ho_q * proj_donor)
        hamiltonian.append(positions[0]**2 * proj_donor)
        hamiltonian.append(-2 * positions[1] * ho_q * proj_bridge_1)
        hamiltonian.append(positions[1]**2 * proj_bridge_1)
        hamiltonian.append(-2 * positions[2] * ho_q * proj_bridge_2)
        hamiltonian.append(positions[2]**2 * proj_bridge_2)
        hamiltonian.append(-2 * positions[3] * ho_q * proj_acceptor)
        hamiltonian.append(positions[3]**2 * proj_acceptor)

        # Initial state, |D><D| * exp(-beta * <D|H|D>)
        d_H_d = (proj_donor * np.sum(hamiltonian)).ptrace([1])
        initial_state = tensor(basis(4, 0).proj(),
                               (-beta * d_H_d).expm()).unit()

        # Collapse operators
        shift_matrix = (
            positions[0] * proj_donor
            + positions[1] * proj_bridge_1
            + positions[2] * proj_bridge_2
            + positions[3] * proj_acceptor
        )
        relax = np.sqrt(gamma * (1 + nbar)) * (a - shift_matrix)
        excite = np.sqrt(gamma * nbar) * (a.dag() - shift_matrix)
        collapse_ops = [relax, excite]

        # Expect operators
        expect_ops = [proj_donor, proj_bridge_1, proj_bridge_2,
                      proj_acceptor, ho_q, ho_p, a.dag() * a]

        if 'repeated_interactions' in job_type:
            B = tensor(qeye(4), qeye(N), sigmam())
            S = (np.sqrt(gamma * (2 * nbar + 1)) *
                 (tensor(a - shift_matrix, qeye(2))))
            interaction_ham = [S * B.dag(), S.dag() * B]
            ancilla_initial_state = (nbar * basis(2, 0).proj() +
                                     (nbar + 1) * basis(2, 1).proj()).unit()
        else:
            interaction_ham = None
            ancilla_initial_state = None

        super().__init__(
            hamiltonian=hamiltonian,
            initial_state=initial_state,
            max_time=max_time,
            dt_output=dt_output,
            collapse_ops=collapse_ops,
            expect_ops=expect_ops,
            perturbation_index=5,
            interaction_ham=interaction_ham,
            ancilla_initial_state=ancilla_initial_state,
            dt_ri=dt_ri,
            num_trotter=num_trotter,
            fidelity_cutoff=fidelity_cutoff
        )

        if 'trotter' in job_type:
            raise NotImplementedError
        if 'rhp_measure' in job_type:
            raise NotImplementedError

        self.csv_header = ('#time,donor_population,bridge_1_population,' +
                           'bridge_2_population,acceptor_population,' +
                           'ho_position,ho_momentum,ho_number')

        if 'state_prep' in job_type:
            self.csv_header += ',fidelity'
            self.csv_template += ',{:0.12e}'
        if 'rhp_measure' in job_type:
            self.csv_header += ',concurrence'
            self.csv_template += ',{:0.12e}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Repeated Interactions for Electron Transport using QuTiP')
    parser.add_argument('-s', '--system', choices=['DA', 'DBA'], default='DA')
    parser.add_argument('-j', '--job_type',
                        choices=['lindblad', 'repeated_interactions',
                                 'repeated_interactions_trotter',
                                 'repeated_interactions_state_prep',
                                 'repeated_interactions_state_prep_continue',
                                 'rhp_measure_lindblad',
                                 'rhp_measure_repeated_interactions'],
                        required=True)
    parser.add_argument('-V', '--coupling', type=float, required=True)
    parser.add_argument('-T', '--kT', type=float, required=True)
    parser.add_argument('-g', '--gamma', type=float, required=True)
    parser.add_argument('-N', '--num_ho_states', type=int, required=True)
    parser.add_argument('-t', '--max_time', type=float, required=True)
    parser.add_argument('-o', '--dt_output', type=float, required=True)
    parser.add_argument('-r', '--dt_ri', type=float, default=None)
    parser.add_argument('-n', '--num_trotter', type=int, default=None)
    parser.add_argument('-f', '--fidelity_cutoff', type=float, default=None)
    # DA parameters
    parser.add_argument('-l', '--lamda', type=float)
    parser.add_argument('-d', '--dE', type=float)
    # DBA parameters
    parser.add_argument('-E', '--energies', type=float, nargs=4)
    parser.add_argument('-p', '--positions', type=float, nargs=4)
    args = parser.parse_args()

    if args.system == 'DA':
        sim = ETSimulation(
            job_type=args.job_type,
            V=args.coupling,
            lamda=args.lamda,
            kT=args.kT,
            gamma=args.gamma,
            dE=args.dE,
            N=args.num_ho_states,
            max_time=args.max_time,
            dt_output=args.dt_output,
            dt_ri=args.dt_ri,
            num_trotter=args.num_trotter,
            fidelity_cutoff=args.fidelity_cutoff
        )
    elif args.system == 'DBA':
        sim = DBASimulation(
            job_type=args.job_type,
            V=args.coupling,
            kT=args.kT,
            gamma=args.gamma,
            energies=args.energies,
            positions=args.positions,
            N=args.num_ho_states,
            max_time=args.max_time,
            dt_output=args.dt_output,
            dt_ri=args.dt_ri,
            num_trotter=args.num_trotter,
            fidelity_cutoff=args.fidelity_cutoff
        )

    if args.job_type == 'lindblad':
        sim.lindblad()
    elif args.job_type == 'repeated_interactions':
        sim.repeated_interactions()
    elif args.job_type == 'repeated_interactions_trotter':
        sim.repeated_interactions_trotter()
    elif args.job_type == 'repeated_interactions_state_prep':
        sim.repeated_interactions_state_prep()
    elif args.job_type == 'repeated_interactions_state_prep_continue':
        sim.repeated_interactions_state_prep_continue()
    elif args.job_type == 'rhp_measure_lindblad':
        sim.rhp_measure_lindblad()
    elif args.job_type == 'rhp_measure_repeated_interactions':
        sim.rhp_measure_repeated_interactions()
