import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
from math import pi
from Arguments import Arguments
args = Arguments()


# load molecular datasets (H5: 10 qubits)
H5datasets = qml.data.load("qchem", molname="H5", basis="STO-3G", bondlength=0.5)
H5data = H5datasets[0]
hamiltonian = H5data.hamiltonian
print("molecular dataset used: {}".format(H5data))


def translator(net):
    assert type(net) == type([])
    updated_design = {}

    q = net[0:10]
    p = net[10:]

    # categories of single-qubit parametric gates
    for i in range(args.n_qubits):
        if q[i] == 0:
            category = 'Rx'
        else:
            category = 'Ry'
        updated_design['rot' + str(i)] = category

    # categories and positions of entangled gates
    for j in range(args.n_qubits):
        category = 'IsingZZ'
        updated_design['enta' + str(j)] = (category, [j, p[j]])

    updated_design['total_gates'] = len(q)
    return updated_design


# Define the device, using lightning.qubit device
dev = qml.device("lightning.qubit", wires=args.n_qubits)

@qml.qnode(dev, diff_method="adjoint")

def quantum_net(q_params, design=None):
    current_design = design
    q_weights = q_params.reshape(args.n_qubits, 2)

    for j in range(args.n_qubits):
        if current_design['rot' + str(j)] == 'Rx':
            qml.RX(q_weights[j][0], wires=j)
        else:
            qml.RY(q_weights[j][0], wires=j)
        if current_design['enta' + str(j)][0] == 'IsingZZ':
            qml.IsingZZ(q_weights[j][1], wires=current_design['enta' + str(j)][1])

    return qml.expval(hamiltonian)


def workflow(q_params, ntrials, design):
    opt = qml.GradientDescentOptimizer(stepsize=0.4)

    for n in range(ntrials):
        q_params, prev_energy = opt.step_and_cost(quantum_net, q_params, design=design)
        # print(f"--- Step: {n}, Energy: {quantum_net(q_params, design=design):.8f}")

    return quantum_net(q_params, design=design)


def Scheme(design):
    np.random.seed(42)

    args = Arguments()
    if torch.cuda.is_available() and args.device == 'cuda':
        print("using cuda device")
    else:
        print("using cpu device")

    total_energy = 0
    for repe in range(1, 11):
        print("get energy repe times: {}".format(repe))
        q_params = 2 * pi * np.random.rand(args.n_qubits * 2)
        energy = workflow(q_params, args.ntrials, design)
        print("energy: {}".format(energy))
        total_energy += energy
    avg_energy = total_energy/repe

    return avg_energy


if __name__ == '__main__':
    net = [0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 2, 5, 1, 2, 9, 8, 3, 3, 7, 6]
    design = translator(net)
    q_params = 2 * pi * np.random.rand(args.n_qubits * 2)
    avg_energy = Scheme(design)
    print("avg energy: {}".format(avg_energy))