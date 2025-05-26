# %%
"""
# Vectors and Vector Spaces for Quantum Computing
## Notebook 1.1: Building Mathematical Intuition Through Code

**Learning Objectives:**
- Understand vectors as quantum state representations
- Master vector operations using NumPy 
- Visualize quantum states and their geometric meaning
- Connect linear algebra to quantum mechanics concepts
- Build intuition for complex vector spaces

**Prerequisites:** Basic Python and NumPy knowledge
**Duration:** 60-90 minutes
"""

# %%
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from IPython.display import display, HTML, Markdown
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('seaborn-v0_8')
np.set_printoptions(precision=3, suppress=True)

print("Welcome to Quantum Vector Spaces!")
print("All libraries loaded successfully ✓")

# %%
"""
## Section 1: What Are Vectors in Quantum Computing?

In classical physics, vectors represent things like velocity or force - they have magnitude and direction.
In quantum computing, vectors represent **quantum states** - the fundamental information units.

Let's start with the simplest quantum system: a **qubit** (quantum bit).
"""

# The two fundamental qubit states
qubit_0 = np.array([1, 0], dtype=complex)  # |0⟩ state
qubit_1 = np.array([0, 1], dtype=complex)  # |1⟩ state

print("Fundamental Qubit States:")
print(f"|0⟩ = {qubit_0}")
print(f"|1⟩ = {qubit_1}")
print()
print("These are like the 'up' and 'down' directions in quantum space")

# Visualization: Classical vs Quantum Vectors
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Classical 2D vector
ax1.arrow(0, 0, 3, 2, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
ax1.set_xlim(-0.5, 4)
ax1.set_ylim(-0.5, 3)
ax1.grid(True, alpha=0.3)
ax1.set_title('Classical Vector\n(velocity, force, etc.)')
ax1.set_xlabel('x-component')
ax1.set_ylabel('y-component')
ax1.annotate('v = (3, 2)', xy=(3, 2), xytext=(3.2, 2.2))

# Quantum state vector (abstract representation)
ax2.bar(['|0⟩', '|1⟩'], [abs(qubit_0[0])**2, abs(qubit_0[1])**2], 
        color=['blue', 'red'], alpha=0.7)
ax2.set_title('Quantum State |0⟩\n(probability amplitudes)')
ax2.set_ylabel('Probability |amplitude|²')
ax2.set_ylim(0, 1.2)

plt.tight_layout()
plt.show()

print("Key Insight: Quantum vectors encode probabilities, not physical directions!")

# %%
"""
## Section 2: Complex Numbers in Quantum States

Unlike classical vectors, quantum state vectors have **complex number** components.
This is crucial because it allows for quantum interference - the heart of quantum computing.
"""

# Create complex quantum states
plus_state = np.array([1, 1], dtype=complex) / np.sqrt(2)    # |+⟩ = (|0⟩ + |1⟩)/√2
minus_state = np.array([1, -1], dtype=complex) / np.sqrt(2)  # |-⟩ = (|0⟩ - |1⟩)/√2
complex_state = np.array([1, 1j], dtype=complex) / np.sqrt(2) # (|0⟩ + i|1⟩)/√2

print("Superposition States (Complex Amplitudes):")
print(f"|+⟩ = {plus_state}")
print(f"|-⟩ = {minus_state}")
print(f"Complex state = {complex_state}")
print()

# Let's visualize complex amplitudes
def plot_complex_amplitudes(state, title):
    """Plot complex amplitudes as magnitude and phase"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    # Magnitude (what we measure)
    magnitudes = np.abs(state)**2
    ax1.bar(['|0⟩', '|1⟩'], magnitudes, color=['blue', 'red'], alpha=0.7)
    ax1.set_title('Probabilities\n|amplitude|²')
    ax1.set_ylabel('Probability')
    ax1.set_ylim(0, 1)
    
    # Real parts
    real_parts = np.real(state)
    colors = ['blue' if x >= 0 else 'red' for x in real_parts]
    ax2.bar(['Re(α₀)', 'Re(α₁)'], real_parts, color=colors, alpha=0.7)
    ax2.set_title('Real Parts')
    ax2.set_ylabel('Real Component')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Imaginary parts
    imag_parts = np.imag(state)
    colors = ['blue' if x >= 0 else 'red' for x in imag_parts]
    ax3.bar(['Im(α₀)', 'Im(α₁)'], imag_parts, color=colors, alpha=0.7)
    ax3.set_title('Imaginary Parts')
    ax3.set_ylabel('Imaginary Component')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    fig.suptitle(f'Complex Amplitudes: {title}', fontsize=14)
    plt.tight_layout()
    return fig

# Plot our complex states
plot_complex_amplitudes(plus_state, '|+⟩ state')
plt.show()

plot_complex_amplitudes(complex_state, 'Complex superposition')
plt.show()

print("Notice: Same probabilities, different interference properties!")

# %%
"""
## Section 3: Vector Operations - The Building Blocks

Now let's implement the essential vector operations that quantum computers use.
These operations have deep physical meaning in quantum mechanics.
"""

def inner_product(psi, phi):
    """Compute inner product ⟨ψ|φ⟩ between two quantum states"""
    return np.conj(psi).T @ phi

def norm(psi):
    """Compute the norm (length) of a quantum state"""
    return np.sqrt(np.real(inner_product(psi, psi)))

def normalize(psi):
    """Normalize a quantum state to unit length"""
    return psi / norm(psi)

def probability(psi, measurement_basis_state):
    """Compute probability of measuring psi in given basis state"""
    amplitude = inner_product(measurement_basis_state, psi)
    return np.real(np.conj(amplitude) * amplitude)

# Test our operations
test_state = np.array([3, 4j], dtype=complex)  # Unnormalized state
print("Vector Operations Demo:")
print(f"Original state: {test_state}")
print(f"Norm: {norm(test_state):.3f}")
print(f"Normalized: {normalize(test_state)}")
print()

# Inner products between basis states
print("Inner Products (Orthogonality Check):")
print(f"⟨0|0⟩ = {inner_product(qubit_0, qubit_0):.3f}")
print(f"⟨1|1⟩ = {inner_product(qubit_1, qubit_1):.3f}")
print(f"⟨0|1⟩ = {inner_product(qubit_0, qubit_1):.3f}")
print(f"⟨1|0⟩ = {inner_product(qubit_1, qubit_0):.3f}")
print("Perfect orthogonality: ⟨0|1⟩ = 0 ✓")

# %%
"""
## Section 4: The Bloch Sphere - 3D Visualization of Qubits

Every qubit state can be visualized as a point on a sphere called the **Bloch sphere**.
This is one of the most important visualizations in quantum computing!

Any qubit state can be written as:
|ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩

Where θ and φ are spherical coordinates.
"""

def bloch_coordinates(theta, phi):
    """Convert spherical coordinates to Bloch sphere Cartesian coordinates"""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi) 
    z = np.cos(theta)
    return x, y, z

def qubit_from_angles(theta, phi):
    """Create qubit state from Bloch sphere angles"""
    return np.array([
        np.cos(theta/2),
        np.exp(1j * phi) * np.sin(theta/2)
    ], dtype=complex)

def angles_from_qubit(psi):
    """Extract Bloch sphere angles from qubit state"""
    # Normalize first
    psi = normalize(psi)
    
    # Extract θ
    theta = 2 * np.arccos(np.abs(psi[0]))
    
    # Extract φ (handle special cases)
    if np.abs(psi[1]) < 1e-10:  # |1⟩ amplitude is essentially zero
        phi = 0
    else:
        phi = np.angle(psi[1] / psi[0]) if np.abs(psi[0]) > 1e-10 else np.angle(psi[1])
    
    return theta, phi

# Create interactive Bloch sphere
def create_bloch_sphere():
    """Create 3D Bloch sphere visualization"""
    
    # Create sphere surface
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Create the plot
    fig = go.Figure()
    
    # Add transparent sphere
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        opacity=0.3,
        colorscale='Blues',
        showscale=False,
        name='Bloch Sphere'
    ))
    
    # Add coordinate axes
    axes_length = 1.2
    
    # X-axis (red)
    fig.add_trace(go.Scatter3d(
        x=[-axes_length, axes_length], y=[0, 0], z=[0, 0],
        mode='lines+text',
        line=dict(color='red', width=4),
        text=['', 'X'],
        textposition='middle right',
        name='X-axis'
    ))
    
    # Y-axis (green) 
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[-axes_length, axes_length], z=[0, 0],
        mode='lines+text',
        line=dict(color='green', width=4),
        text=['', 'Y'],
        textposition='middle right',
        name='Y-axis'
    ))
    
    # Z-axis (blue)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[-axes_length, axes_length],
        mode='lines+text',
        line=dict(color='blue', width=4),
        text=['', 'Z'],
        textposition='middle right',
        name='Z-axis'
    ))
    
    # Add state labels
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[1.1],
        mode='text',
        text=['|0⟩'],
        textfont=dict(size=16, color='blue'),
        name='|0⟩ state'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[-1.1],
        mode='text',
        text=['|1⟩'],
        textfont=dict(size=16, color='red'),
        name='|1⟩ state'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[1.1], y=[0], z=[0],
        mode='text',
        text=['|+⟩'],
        textfont=dict(size=16, color='purple'),
        name='|+⟩ state'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[-1.1], y=[0], z=[0],
        mode='text',
        text=['|-⟩'],
        textfont=dict(size=16, color='orange'),
        name='|-⟩ state'
    ))
    
    # Configure layout
    fig.update_layout(
        title='The Bloch Sphere: 3D Representation of Qubit States',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y', 
            zaxis_title='Z',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        showlegend=False,
        width=700,
        height=600
    )
    
    return fig

# Display the Bloch sphere
bloch_fig = create_bloch_sphere()
bloch_fig.show()

print("The Bloch Sphere represents ALL possible qubit states as points on a unit sphere!")
print("• North pole (0,0,1): |0⟩ state")  
print("• South pole (0,0,-1): |1⟩ state")
print("• Equator: Superposition states like |+⟩ and |-⟩")

# %%
"""
## Section 5: Interactive State Explorer

Let's create an interactive tool to explore different quantum states and see how 
they appear on the Bloch sphere and as probability distributions.
"""

# Create quantum states for common examples
common_states = {
    '|0⟩': np.array([1, 0], dtype=complex),
    '|1⟩': np.array([0, 1], dtype=complex), 
    '|+⟩': np.array([1, 1], dtype=complex) / np.sqrt(2),
    '|-⟩': np.array([1, -1], dtype=complex) / np.sqrt(2),
    '|i⟩': np.array([1, 1j], dtype=complex) / np.sqrt(2),
    '|-i⟩': np.array([1, -1j], dtype=complex) / np.sqrt(2)
}

def analyze_quantum_state(state, name="Custom State"):
    """Comprehensive analysis of a quantum state"""
    
    # Normalize the state
    state = normalize(state)
    
    # Calculate properties
    prob_0 = probability(state, qubit_0)
    prob_1 = probability(state, qubit_1)
    state_norm = norm(state)
    
    # Get Bloch coordinates
    theta, phi = angles_from_qubit(state)
    x, y, z = bloch_coordinates(theta, phi)
    
    print(f"=== Analysis of {name} ===")
    print(f"State vector: {state}")
    print(f"Norm: {state_norm:.6f}")
    print(f"P(measure |0⟩): {prob_0:.3f}")
    print(f"P(measure |1⟩): {prob_1:.3f}")
    print(f"Bloch angles: θ={theta:.3f}, φ={phi:.3f}")
    print(f"Bloch coordinates: ({x:.3f}, {y:.3f}, {z:.3f})")
    print()
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Probability distribution
    ax1.bar(['|0⟩', '|1⟩'], [prob_0, prob_1], color=['blue', 'red'], alpha=0.7)
    ax1.set_title('Measurement Probabilities')
    ax1.set_ylabel('Probability')
    ax1.set_ylim(0, 1)
    for i, (label, prob) in enumerate([('|0⟩', prob_0), ('|1⟩', prob_1)]):
        ax1.text(i, prob + 0.05, f'{prob:.3f}', ha='center', va='bottom')
    
    # 2. Complex amplitudes
    real_parts = [np.real(state[0]), np.real(state[1])]
    imag_parts = [np.imag(state[0]), np.imag(state[1])]
    
    x_pos = np.arange(2)
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, real_parts, width, label='Real', alpha=0.7)
    bars2 = ax2.bar(x_pos + width/2, imag_parts, width, label='Imaginary', alpha=0.7)
    
    ax2.set_title('Complex Amplitudes')
    ax2.set_ylabel('Amplitude Value')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['α₀', 'α₁'])
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 3. Phase visualization (polar plot)
    angles = [np.angle(amp) for amp in state]
    magnitudes = [np.abs(amp) for amp in state]
    
    ax3 = plt.subplot(223, projection='polar')
    colors = ['blue', 'red']
    for i, (angle, mag) in enumerate(zip(angles, magnitudes)):
        ax3.arrow(0, 0, angle, mag, head_width=0.1, head_length=0.05, 
                 fc=colors[i], ec=colors[i], alpha=0.7, width=0.02)
        ax3.text(angle, mag + 0.1, f'α_{i}', ha='center', va='center')
    ax3.set_title('Amplitude Phases')
    ax3.set_ylim(0, 1)
    
    # 4. Bloch sphere projection
    ax4 = plt.subplot(224, projection='3d')
    
    # Draw sphere wireframe
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_wire = np.outer(np.cos(u), np.sin(v))
    y_wire = np.outer(np.sin(u), np.sin(v))
    z_wire = np.outer(np.ones(np.size(u)), np.cos(v))
    ax4.plot_wireframe(x_wire, y_wire, z_wire, alpha=0.1, color='gray')
    
    # Plot state vector
    ax4.scatter([x], [y], [z], color='red', s=100, alpha=0.8)
    ax4.plot([0, x], [0, y], [0, z], 'r-', linewidth=3, alpha=0.8)
    
    # Add axis labels
    ax4.text(1.2, 0, 0, 'X', fontsize=12)
    ax4.text(0, 1.2, 0, 'Y', fontsize=12)
    ax4.text(0, 0, 1.2, 'Z', fontsize=12)
    
    ax4.set_title('Bloch Sphere Representation')
    ax4.set_xlim([-1, 1])
    ax4.set_ylim([-1, 1]) 
    ax4.set_zlim([-1, 1])
    
    plt.tight_layout()
    plt.show()
    
    return {
        'state': state,
        'probabilities': [prob_0, prob_1],
        'bloch_coords': [x, y, z],
        'angles': [theta, phi]
    }

# Analyze common quantum states
print("Let's explore some fundamental quantum states:\n")

for name, state in common_states.items():
    result = analyze_quantum_state(state, name)
    print("-" * 50)

# %%
"""
## Section 6: Building Your Own Quantum States

Now let's create custom quantum states and see how changing the amplitudes 
affects the quantum properties.
"""

def create_custom_qubit(alpha_real=1.0, alpha_imag=0.0, beta_real=0.0, beta_imag=0.0):
    """Create a custom qubit state from amplitude components"""
    state = np.array([alpha_real + 1j*alpha_imag, beta_real + 1j*beta_imag], dtype=complex)
    return normalize(state)

# Interactive state builder
print("=== Custom Quantum State Builder ===")
print("Create your own quantum state by specifying complex amplitudes:")
print("|ψ⟩ = α|0⟩ + β|1⟩")
print("where α = α_real + i*α_imag and β = β_real + i*β_imag")
print()

# Example: Create a state with equal real amplitudes
custom_state_1 = create_custom_qubit(alpha_real=1.0, beta_real=1.0)
analyze_quantum_state(custom_state_1, "Equal Real Amplitudes")

# Example: Create a state with complex phases
custom_state_2 = create_custom_qubit(alpha_real=1.0, beta_real=0.0, beta_imag=1.0)
analyze_quantum_state(custom_state_2, "Complex Phase State")

# %%
"""
## Section 7: Multi-Qubit Systems and Tensor Products

Real quantum computers work with multiple qubits. The mathematical operation 
that combines qubits is called the **tensor product** (⊗).

For two qubits, the combined state space has 4 dimensions:
|00⟩, |01⟩, |10⟩, |11⟩
"""

def tensor_product(state1, state2):
    """Compute tensor product of two quantum states"""
    return np.kron(state1, state2)

def create_two_qubit_state(qubit1, qubit2):
    """Create two-qubit state from individual qubits"""
    return tensor_product(qubit1, qubit2)

# Two-qubit basis states
basis_2q = {
    '|00⟩': create_two_qubit_state(qubit_0, qubit_0),
    '|01⟩': create_two_qubit_state(qubit_0, qubit_1), 
    '|10⟩': create_two_qubit_state(qubit_1, qubit_0),
    '|11⟩': create_two_qubit_state(qubit_1, qubit_1)
}

print("Two-Qubit Basis States:")
for name, state in basis_2q.items():
    print(f"{name} = {state}")
print()

# Create entangled states
bell_state_1 = (basis_2q['|00⟩'] + basis_2q['|11⟩']) / np.sqrt(2)  # |Φ+⟩
bell_state_2 = (basis_2q['|00⟩'] - basis_2q['|11⟩']) / np.sqrt(2)  # |Φ-⟩

print("Famous Entangled States (Bell States):")
print(f"|Φ+⟩ = {bell_state_1}")
print(f"|Φ-⟩ = {bell_state_2}")

# Visualize two-qubit states
def plot_two_qubit_state(state, title):
    """Visualize two-qubit state probabilities"""
    probabilities = np.abs(state)**2
    labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, probabilities, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    plt.title(f'Two-Qubit State: {title}')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    
    # Add probability values on bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')
    
    plt.grid(axis='y', alpha=0.3)
    plt.show()

# Plot our states
plot_two_qubit_state(bell_state_1, '|Φ+⟩ Bell State')
plot_two_qubit_state(create_two_qubit_state(plus_state, plus_state), '|+⟩⊗|+⟩ Product State')

print("Notice: Bell states show quantum entanglement - perfect correlations!")

# %%
"""
## Section 8: Exercises and Challenges

Now it's your turn! Try these exercises to solidify your understanding.
"""

print("=== EXERCISES ===")
print()

print("Exercise 1: Create the |−i⟩ state")
print("This state should have equal probabilities but with -i phase")
print("Expected: α₀=1/√2, α₁=-i/√2")

# Your solution here:
minus_i_state = create_custom_qubit(alpha_real=1/np.sqrt(2), beta_imag=-1/np.sqrt(2))
analyze_quantum_state(minus_i_state, "Your |-i⟩ State")

print("\nExercise 2: Find a state that gives 75% probability of measuring |0⟩")
print("Hint: If P(|0⟩) = |α₀|² = 0.75, then |α₀| = √0.75")

# Your solution here:
prob_75_state = create_custom_qubit(alpha_real=np.sqrt(0.75), beta_real=np.sqrt(0.25))
analyze_quantum_state(prob_75_state, "75% |0⟩ Probability State")

print("\nExercise 3: Create a maximally entangled state different from |Φ+⟩")
print("Try: (|01⟩ + |10⟩)/√2")

# Your solution here:
max_entangled = (basis_2q['|01⟩'] + basis_2q['|10⟩']) / np.sqrt(2)
plot_two_qubit_state(max_entangled, "Your Maximally Entangled State")

print("\n=== ADVANCED CHALLENGES ===")
print()

print("Challenge 1: Implement a function to check if two qubits are entangled")
def is_entangled(two_qubit_state):
    """
    Check if a two-qubit state is entangled using the Schmidt decomposition
    A state is entangled if it cannot be written as a tensor product
    """
    # Reshape state into 2x2 matrix
    state_matrix = two_qubit_state.reshape(2, 2)
    
    # Compute singular values
    singular_values = np.linalg.svd(state_matrix, compute_uv=False)
    
    # Count non-zero singular values (Schmidt rank)
    schmidt_rank = np.sum(singular_values > 1e-10)
    
    # Entangled if Schmidt rank > 1
    return schmidt_rank > 1

# Test entanglement detection
print(f"Bell state |Φ+⟩ entangled: {is_entangled(bell_state_1)}")
print(f"Product state |+⟩⊗|+⟩ entangled: {is_entangled(create_two_qubit_state(plus_state, plus_state))}")

print("\nChallenge 2: Create a three-qubit GHZ state")
print("GHZ state: (|000⟩ + |111⟩)/√2")

# Three-qubit basis
qubit_000 = tensor_product(tensor_product(qubit_0, qubit_0), qubit_0)
qubit_111 = tensor_product(tensor_product(qubit_1, qubit_1), qubit_1)
ghz_state = (qubit_000 + qubit_111) / np.sqrt(2)

print(f"GHZ state dimension: {len(ghz_state)}")
print(f"Non-zero amplitudes: {np.sum(np.abs(ghz_state) > 1e-10)}")

# Visualize GHZ state
labels_3q = ['|000⟩', '|001⟩', '|010⟩', '|011⟩', '|100⟩', '|101⟩', '|110⟩', '|111⟩']
probabilities_3q = np.abs(ghz_state)**2

plt.figure(figsize=(12, 6))
bars = plt.bar(labels_3q, probabilities_3q, alpha=0.7)
plt.title('Three-Qubit GHZ State')
plt.ylabel('Probability') 
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
for bar, prob in zip(bars, probabilities_3q):
    if prob > 1e-10:
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

# %%
"""
## Section 9: Key Takeaways and Next Steps

### What You've Learned

**1. Quantum States as Vectors**
- Qubits are represented as 2D complex vectors
- The components (amplitudes) encode probability information
- Complex numbers enable quantum interference

**2. Essential Vector Operations**
- Inner products compute overlaps between states
- Normalization ensures valid quantum states
- These operations have direct physical meanings

**3. Geometric Visualization**
- The Bloch sphere provides intuitive 3D representation
- Every qubit state corresponds to a point on the sphere
- Rotations on the sphere = quantum operations

**4. Multi-Qubit Systems**
- Tensor products combine individual qubits
- N qubits require 2^N dimensional vectors
- Entanglement creates non-separable states

**5. Mathematical Foundation**
- Linear algebra is the language of quantum mechanics
- Complex vector spaces naturally describe quantum systems
- Geometric intuition helps understand abstract concepts
"""

# Final demonstration: Interactive state manipulator
def interactive_qubit_explorer():
    """Create final interactive demonstration"""
    
    print("=== FINAL DEMO: Interactive Qubit State Explorer ===")
    print("Manipulate a qubit state and see all representations simultaneously!")
    
    # Create sliders for real and imaginary parts
    alpha_real_slider = widgets.FloatSlider(
        value=1.0, min=-2.0, max=2.0, step=0.1,
        description='α (real):', style={'description_width': 'initial'}
    )
    alpha_imag_slider = widgets.FloatSlider(
        value=0.0, min=-2.0, max=2.0, step=0.1,
        description='α (imag):', style={'description_width': 'initial'}
    )
    beta_real_slider = widgets.FloatSlider(
        value=0.0, min=-2.0, max=2.0, step=0.1,
        description='β (real):', style={'description_width': 'initial'}
    )
    beta_imag_slider = widgets.FloatSlider(
        value=0.0, min=-2.0, max=2.0, step=0.1,
        description='β (imag):', style={'description_width': 'initial'}
    )
    
    # Output widget for displaying results
    output = widgets.Output()
    
    def update_state(*args):
        """Update visualization when sliders change"""
        with output:
            output.clear_output(wait=True)
            
            # Create state from slider values
            state = create_custom_qubit(
                alpha_real_slider.value, alpha_imag_slider.value,
                beta_real_slider.value, beta_imag_slider.value
            )
            
            # Quick analysis without full plots
            prob_0 = probability(state, qubit_0)
            prob_1 = probability(state, qubit_1) 
            theta, phi = angles_from_qubit(state)
            x, y, z = bloch_coordinates(theta, phi)
            
            print(f"Normalized State: {state}")
            print(f"P(|0⟩) = {prob_0:.3f}, P(|1⟩) = {prob_1:.3f}")
            print(f"Bloch coordinates: ({x:.3f}, {y:.3f}, {z:.3f})")
            
            # Simple bar chart
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.bar(['|0⟩', '|1⟩'], [prob_0, prob_1], color=['blue', 'red'], alpha=0.7)
            ax.set_ylabel('Probability')
            ax.set_title('Measurement Probabilities')
            ax.set_ylim(0, 1)
            plt.tight_layout()
            plt.show()
    
    # Connect sliders to update function
    for slider in [alpha_real_slider, alpha_imag_slider, beta_real_slider, beta_imag_slider]:
        slider.observe(update_state, names='value')
    
    # Display interface
    display(widgets.VBox([
        widgets.HTML("<h3>Qubit State: |ψ⟩ = α|0⟩ + β|1⟩</h3>"),
        alpha_real_slider, alpha_imag_slider,
        beta_real_slider, beta_imag_slider,
        output
    ]))
    
    # Initial update
    update_state()

# Run the interactive explorer
interactive_qubit_explorer()

# %%
"""
### Connection to Quantum Computing

The vector space concepts you've learned directly translate to:

**Quantum Gates**: Unitary matrices that rotate states on the Bloch sphere
**Quantum Algorithms**: Sequences of rotations that solve computational problems  
**Quantum Error Correction**: Using redundancy in higher-dimensional spaces
**Quantum Entanglement**: Non-separable states in tensor product spaces

### Next Steps

**Immediate (Next Notebook)**:
- Learn how matrices represent quantum operations
- Implement common quantum gates (X, Y, Z, H)
- Understand eigenvalues and measurement

**Medium Term**:
- Study group theory through quantum gate composition
- Learn Fourier analysis through quantum signal processing
- Explore number theory through factoring algorithms

**Advanced Topics**:
- Quantum error correction codes
- Variational quantum algorithms
- Quantum machine learning

### Mathematical Prerequisites Mastered ✓

- ✅ Complex numbers and their geometric interpretation
- ✅ Vector operations (inner products, norms, normalization)
- ✅ Tensor products for multi-particle systems
- ✅ Geometric visualization of abstract spaces
- ✅ Connection between mathematics and physics

You now have the linear algebra foundation needed for quantum computing!
"""

# %%
"""
### Bonus: Quantum State Art Generator

Let's end with something fun - create beautiful visualizations of quantum states!
"""

def create_quantum_art(num_states=50):
    """Generate artistic visualization of random quantum states"""
    
    # Generate random quantum states
    states = []
    bloch_points = []
    
    for _ in range(num_states):
        # Random complex amplitudes
        alpha = np.random.randn() + 1j * np.random.randn()
        beta = np.random.randn() + 1j * np.random.randn()
        state = normalize(np.array([alpha, beta]))
        states.append(state)
        
        # Get Bloch coordinates
        theta, phi = angles_from_qubit(state)
        x, y, z = bloch_coordinates(theta, phi)
        bloch_points.append([x, y, z])
    
    bloch_points = np.array(bloch_points)
    
    # Create artistic 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot transparent sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='lightblue')
    
    # Plot quantum states as colored points
    colors = plt.cm.rainbow(np.linspace(0, 1, num_states))
    scatter = ax.scatter(bloch_points[:, 0], bloch_points[:, 1], bloch_points[:, 2], 
                        c=colors, s=50, alpha=0.8)
    
    # Connect points with lines for artistic effect
    for i in range(num_states - 1):
        ax.plot([bloch_points[i, 0], bloch_points[i+1, 0]],
                [bloch_points[i, 1], bloch_points[i+1, 1]],
                [bloch_points[i, 2], bloch_points[i+1, 2]], 
                color=colors[i], alpha=0.3, linewidth=1)
    
    ax.set_title(f'Quantum State Art: {num_states} Random Qubit States', fontsize=16)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Make it look nice
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    
    plt.tight_layout()
    plt.show()
    
    print(f"Beautiful! Each point represents a valid quantum state.")
    print(f"The colors show the 'trajectory' through quantum state space.")

# Create quantum art
create_quantum_art(30)

# %%
"""
## Summary Statistics

Let's analyze what we've accomplished:
"""

print("=== NOTEBOOK COMPLETION SUMMARY ===")
print()
print("📊 Concepts Covered:")
concepts = [
    "Complex vector spaces",
    "Quantum state representation", 
    "Inner products and norms",
    "Bloch sphere visualization",
    "Tensor product spaces",
    "Quantum entanglement",
    "Multi-qubit systems",
    "Interactive state manipulation"
]

for i, concept in enumerate(concepts, 1):
    print(f"  {i}. ✅ {concept}")

print()
print("🔢 Code Statistics:")
print(f"  • Functions implemented: ~15")
print(f"  • Visualizations created: ~10") 
print(f"  • Interactive widgets: 3")
print(f"  • Quantum states analyzed: ~12")

print()
print("🎯 Learning Objectives Met:")
objectives = [
    "Understand vectors as quantum states ✅",
    "Master NumPy vector operations ✅", 
    "Visualize quantum states geometrically ✅",
    "Connect linear algebra to quantum mechanics ✅",
    "Build intuition for complex spaces ✅"
]

for obj in objectives:
    print(f"  • {obj}")

print()
print("🚀 Ready for Next Notebook: Matrices and Linear Transformations!")
print("   You'll learn how quantum gates manipulate these vector states.")

print("""
╔══════════════════════════════════════════════════════════════╗
║  Congratulations! You've mastered the vector foundations     ║
║  of quantum computing through hands-on Python programming.   ║
║                                                              ║
║  Next: Discover how matrices become quantum gates! 🚪        ║
╚══════════════════════════════════════════════════════════════╝
""")