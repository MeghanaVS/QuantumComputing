import json
import pennylane as qp
import pennylane.numpy as np
def create_Hamiltonian(h):
    """
    Function in charge of generating the Hamiltonian of the statement.

    Args:
        h (float): magnetic field strength

    Returns:
        (qp.Hamiltonian): Hamiltonian of the statement associated to h
    """
    num_wires = 4
    coeffs = []
    obs = []

    for i in range(num_wires):
        coeffs.append(-1.0)
        obs.append(qml.PauliZ(i) @ qml.PauliZ((i + 1) % num_wires))

    for i in range(num_wires):
        coeffs.append(-float(h))
        obs.append(qml.PauliX(i))

    return qml.Hamiltonian(coeffs, obs)
ev = qp.device("default.qubit", wires=4)

@qp.qnode(dev)
def model(params, H):
    """
    To implement VQE you need an ansatz for the candidate ground state!
    Define here the VQE ansatz in terms of some parameters (params) that
    create the candidate ground state. These parameters will
    be optimized later.

    Args:
        params (numpy.array): parameters to be used in the variational circuit
        H (qp.Hamiltonian): Hamiltonian used to calculate the expected value

    Returns:
        (float): Expected value with respect to the Hamiltonian H
    """
    num_wires = 4
    num_layers = params.shape[0]

    for layer in range(num_layers):
        for i in range(num_wires):
            qml.RY(params[layer, i, 0], wires=i)
            qml.RZ(params[layer, i, 1], wires=i)

        for i in range(num_wires):
            qml.CNOT(wires=[i, (i + 1) % num_wires])

    return qml.expval(H)
def train(h):
    """
    In this function you must design a subroutine that returns the
    parameters that best approximate the ground state.

    Args:
        h (float): magnetic field strength

    Returns:
        (numpy.array): parameters that best approximate the ground state.
    """
    num_wires = 4
    num_layers = 3

    H = create_Hamiltonian(h)

    params = np.zeros((num_layers, num_wires, 2), requires_grad=True)

    opt = qml.AdamOptimizer(stepsize=0.1)

    steps = 150
    for _ in range(steps):
        params, _ = opt.step_and_cost(lambda p: model(p, H), params)

    return params
# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    params = train(ins)
    return str(model(params, create_Hamiltonian(ins)))


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-1
    ), "The expected value is not correct."
# These are the public test cases
test_cases = [
    ('1.0', '-5.226251859505506'),
    ('2.3', '-9.66382463698038')
]
# This will run the public test cases locally
for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")
