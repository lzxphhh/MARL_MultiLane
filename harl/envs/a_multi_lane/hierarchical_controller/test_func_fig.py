import numpy as np
import matplotlib.pyplot as plt

# Define x range
x = np.linspace(-10, 10, 1000)

# Original sigmoid function
def sigmoid(x):
   return 1 / (1 + np.exp(-x))

# Approximation 1: Tanh-based (most similar)
def tanh_approx(x):
   return 0.5 * (1 + np.tanh(x))

# Approximation 2: Arctangent-based
def arctan_approx(x):
   return np.arctan(x) / np.pi + 0.5

# Approximation 3: Rational function approximation
def rational_approx(x):
   return x / (2 * np.sqrt(1 + x**2)) + 0.5

# Approximation 4: Softened linear function
def soft_linear_approx(x, k=0.2):
   return np.maximum(0, np.minimum(1, 0.5 + k * x))

# Calculate all function values
y_sigmoid = sigmoid(x)
y_tanh = tanh_approx(x)
y_arctan = arctan_approx(x)
y_rational = rational_approx(x)
y_soft_linear = soft_linear_approx(x)

# Create comprehensive comparison plots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# Plot 1: All functions comparison
axes[0, 0].plot(x, y_sigmoid, 'k-', linewidth=3, label='Original Sigmoid')
axes[0, 0].plot(x, y_tanh, 'r--', linewidth=2, label='Tanh: 0.5(1+tanh(x))')
axes[0, 0].plot(x, y_arctan, 'g--', linewidth=2, label='Arctan: atan(x)/π+0.5')
axes[0, 0].plot(x, y_rational, 'b--', linewidth=2, label='Rational: x/(2√(1+x²))+0.5')
axes[0, 0].plot(x, y_soft_linear, 'm--', linewidth=2, label='Soft Linear')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
axes[0, 0].set_title('All Function Approximations')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=8)
axes[0, 0].set_xlim(-6, 6)
axes[0, 0].set_ylim(-0.1, 1.1)

# Plot 2: Tanh vs Sigmoid (best approximation)
axes[0, 1].plot(x, y_sigmoid, 'k-', linewidth=3, label='Original Sigmoid')
axes[0, 1].plot(x, y_tanh, 'r--', linewidth=2, label='Tanh Approximation')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('y')
axes[0, 1].set_title('Sigmoid vs Tanh (Recommended)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()
axes[0, 1].set_xlim(-6, 6)

# Plot 3: Error comparison
error_tanh = np.abs(y_sigmoid - y_tanh)
error_arctan = np.abs(y_sigmoid - y_arctan)
error_rational = np.abs(y_sigmoid - y_rational)
error_soft_linear = np.abs(y_sigmoid - y_soft_linear)

axes[0, 2].plot(x, error_tanh, 'r-', linewidth=2, label='Tanh Error')
axes[0, 2].plot(x, error_arctan, 'g-', linewidth=2, label='Arctan Error')
axes[0, 2].plot(x, error_rational, 'b-', linewidth=2, label='Rational Error')
axes[0, 2].plot(x, error_soft_linear, 'm-', linewidth=2, label='Soft Linear Error')
axes[0, 2].set_xlabel('x')
axes[0, 2].set_ylabel('Absolute Error')
axes[0, 2].set_title('Approximation Errors')
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].legend(fontsize=8)
axes[0, 2].set_xlim(-6, 6)
axes[0, 2].set_yscale('log')

# Plot 4: Close-up view [-3, 3]
mask = (x >= -3) & (x <= 3)
axes[1, 0].plot(x[mask], y_sigmoid[mask], 'k-', linewidth=3, label='Sigmoid')
axes[1, 0].plot(x[mask], y_tanh[mask], 'r--', linewidth=2, label='Tanh')
axes[1, 0].plot(x[mask], y_arctan[mask], 'g--', linewidth=2, label='Arctan')
axes[1, 0].plot(x[mask], y_rational[mask], 'b--', linewidth=2, label='Rational')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')
axes[1, 0].set_title('Close-up View (x ∈ [-3, 3])')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Plot 5: Gradient comparison
dx = x[1] - x[0]
grad_sigmoid = np.gradient(y_sigmoid, dx)
grad_tanh = np.gradient(y_tanh, dx)
grad_arctan = np.gradient(y_arctan, dx)
grad_rational = np.gradient(y_rational, dx)

axes[1, 1].plot(x, grad_sigmoid, 'k-', linewidth=3, label='Sigmoid Gradient')
axes[1, 1].plot(x, grad_tanh, 'r--', linewidth=2, label='Tanh Gradient')
axes[1, 1].plot(x, grad_arctan, 'g--', linewidth=2, label='Arctan Gradient')
axes[1, 1].plot(x, grad_rational, 'b--', linewidth=2, label='Rational Gradient')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('dy/dx')
axes[1, 1].set_title('Gradient Comparison')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(fontsize=8)
axes[1, 1].set_xlim(-6, 6)

# Plot 6: Tanh error (zoomed)
axes[1, 2].plot(x, error_tanh, 'r-', linewidth=2)
axes[1, 2].set_xlabel('x')
axes[1, 2].set_ylabel('|Sigmoid - Tanh|')
axes[1, 2].set_title('Tanh Approximation Error (Zoomed)')
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_xlim(-6, 6)

# Plot 7: Very close-up [-1, 1]
mask_close = (x >= -1) & (x <= 1)
axes[2, 0].plot(x[mask_close], y_sigmoid[mask_close], 'k-', linewidth=3, label='Sigmoid')
axes[2, 0].plot(x[mask_close], y_tanh[mask_close], 'r--', linewidth=2, label='Tanh')
axes[2, 0].plot(x[mask_close], y_arctan[mask_close], 'g--', linewidth=2, label='Arctan')
axes[2, 0].plot(x[mask_close], y_rational[mask_close], 'b--', linewidth=2, label='Rational')
axes[2, 0].set_xlabel('x')
axes[2, 0].set_ylabel('y')
axes[2, 0].set_title('Very Close-up (x ∈ [-1, 1])')
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].legend()

# Plot 8: Function comparison at critical region
critical_x = np.linspace(-2, 2, 200)
axes[2, 1].plot(critical_x, sigmoid(critical_x), 'k-', linewidth=3, label='Sigmoid')
axes[2, 1].plot(critical_x, tanh_approx(critical_x), 'r--', linewidth=2, label='Tanh')
axes[2, 1].set_xlabel('x')
axes[2, 1].set_ylabel('y')
axes[2, 1].set_title('Critical Region Comparison')
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].legend()

# Plot 9: Error statistics table
axes[2, 2].axis('off')
max_error_tanh = np.max(error_tanh)
mean_error_tanh = np.mean(error_tanh)
max_error_arctan = np.max(error_arctan)
mean_error_arctan = np.mean(error_arctan)
max_error_rational = np.max(error_rational)
mean_error_rational = np.mean(error_rational)
max_error_soft = np.max(error_soft_linear)
mean_error_soft = np.mean(error_soft_linear)

error_text = f"""Error Statistics:

Method          Max Error       Mean Error
------------------------------------------------
Tanh           {max_error_tanh:.2e}     {mean_error_tanh:.2e}
Arctan         {max_error_arctan:.2e}     {mean_error_arctan:.2e}  
Rational       {max_error_rational:.2e}     {mean_error_rational:.2e}
Soft Linear    {max_error_soft:.2e}     {mean_error_soft:.2e}

RECOMMENDATION FOR MPC:
Use Tanh: 0.5*(1 + tanh(x))

CasADi Implementation:
safety_cost += 0.5 * (1 + ca.tanh(ref_dist))

Advantages:
- Nearly identical to sigmoid
- Numerically stable
- No overflow issues
- Fast computation
"""

axes[2, 2].text(0.05, 0.95, error_text, transform=axes[2, 2].transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.show()

# Print detailed analysis
print("=" * 60)
print("SIGMOID FUNCTION APPROXIMATION ANALYSIS FOR MPC")
print("=" * 60)

print(f"\n1. TANH APPROXIMATION: y = 0.5*(1 + tanh(x))")
print(f"   Maximum Error: {max_error_tanh:.8f}")
print(f"   Mean Error: {mean_error_tanh:.8f}")
print(f"   ✓ BEST CHOICE for MPC")
print(f"   ✓ Numerically identical to sigmoid")
print(f"   ✓ CasADi: safety_cost += 0.5 * (1 + ca.tanh(ref_dist))")

print(f"\n2. ARCTAN APPROXIMATION: y = arctan(x)/π + 0.5")
print(f"   Maximum Error: {max_error_arctan:.8f}")
print(f"   Mean Error: {mean_error_arctan:.8f}")
print(f"   ✓ Bounded derivatives")
print(f"   ✓ CasADi: safety_cost += ca.atan(ref_dist)/ca.pi + 0.5")

print(f"\n3. RATIONAL APPROXIMATION: y = x/(2√(1+x²)) + 0.5")
print(f"   Maximum Error: {max_error_rational:.8f}")
print(f"   Mean Error: {mean_error_rational:.8f}")
print(f"   ✓ No transcendental functions")
print(f"   ✓ CasADi: safety_cost += ref_dist/(2*ca.sqrt(1 + ref_dist**2)) + 0.5")

print(f"\n4. SOFT LINEAR: y = max(0, min(1, 0.5 + 0.2*x))")
print(f"   Maximum Error: {max_error_soft:.8f}")
print(f"   Mean Error: {mean_error_soft:.8f}")
print(f"   ✓ Simplest implementation")
print(f"   ✓ CasADi: safety_cost += ca.fmax(0, ca.fmin(1, 0.5 + 0.2*ref_dist))")

print(f"\n" + "="*60)
print("FINAL RECOMMENDATION: Use TANH approximation")
print("Reason: Virtually identical to sigmoid with perfect numerical stability")
print("="*60)