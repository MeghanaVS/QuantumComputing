"""
Quantum Computing Hello World
A simple quantum program that creates a Bell state (entangled pair)
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

def hello_quantum_world():
    """Create and simulate a simple quantum circuit"""
    
    # Create a quantum circuit with 2 qubits and 2 classical bits
    qc = QuantumCircuit(2, 2)
    
    # Apply Hadamard gate to first qubit
    qc.h(0)
    
    # Apply CNOT gate to create entanglement
    qc.cx(0, 1)
    
    # Measure both qubits
    qc.measure([0, 1], [0, 1])
    
    print("Quantum Circuit:")
    print(qc)
    
    # Simulate the circuit
    simulator = AerSimulator()
    job = simulator.run(qc, shots=1000)
    result = job.result()
    counts = result.get_counts(qc)
    
    print("\nMeasurement Results (1000 shots):")
    print(counts)
    
    return counts

if __name__ == "__main__":
    print("Hello, Quantum World! 🌍⚛️\n")
    hello_quantum_world()
