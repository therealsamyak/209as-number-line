import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional


class ParticleSystem:
    """
    Particle on a number line with dynamics including:
    - External potential field φ(y)
    - Speed wobble (velocity noise)
    - Crash probability
    - Sensor noise
    """

    def __init__(
        self,
        mass: float,
        y_0: float,
        v_0: float,
        v_max: float,
        p_c: float,
        phi: Optional[Callable[[float], float]] = None,
        phi_derivative: Optional[Callable[[float], float]] = None,
    ):
        """
        Initialize the particle system.

        Parameters:
        -----------
        mass : float
            Mass of the particle (m)
        y_0 : float
            Initial position of the particle
        v_0 : float
            Initial velocity of the particle
        v_max : float
            Maximum particle velocity in the system
        p_c : float
            Crash probability parameter
        phi : Callable[[float], float], optional
            External potential field function φ(y)
        phi_derivative : Callable[[float], float], optional
            Derivative of potential field dφ/dy
        """
        self.m = mass
        self.v_max = v_max
        self.p_c = p_c

        # State variables
        self.y = y_0  # position
        self.v = v_0  # velocity

        # Potential field and its derivative
        self.phi = phi if phi is not None else lambda y: 0.0
        self.phi_derivative = (
            phi_derivative if phi_derivative is not None else lambda y: 0.0
        )

        # History for plotting
        self.history = {
            "time": [0],
            "position": [y_0],
            "velocity": [v_0],
            "sensed_position": [y_0],
            "applied_force": [0.0],
            "crashed": [False],
        }
        self.time = 0

    def compute_net_force(self, f_i: float) -> float:
        """
        Compute net force on the particle.

        f_net(t) = f_i(t) - dφ/dy

        Parameters:
        -----------
        f_i : float
            Applied force from user input, bounded in [-1, 1]

        Returns:
        --------
        float
            Net force acting on the particle
        """
        # Clamp applied force to [-1, 1]
        f_i = np.clip(f_i, -1.0, 1.0)

        # Net force = applied force - potential gradient
        f_net = f_i - self.phi_derivative(self.y)
        return f_net

    def update_velocity(self, f_net: float, dt: float = 1.0) -> bool:
        """
        Update velocity with crash probability check and speed wobble.

        v[t+1] = v[t] + (1/m)*f_net(t) + N(0, 0.01*v[t]^2)
        with crash probability check

        Parameters:
        -----------
        f_net : float
            Net force acting on the particle
        dt : float
            Time step (default 1.0 for discrete time)

        Returns:
        --------
        bool
            True if crash occurred, False otherwise
        """
        crashed = False

        # Check crash probability
        # Draw uniform random number p ∈ U([0,1])
        p = np.random.uniform(0, 1)
        crash_threshold = self.p_c * (np.abs(self.v) / self.v_max)

        if p <= crash_threshold:
            # Crash! Velocity goes to 0
            self.v = 0.0
            crashed = True
        else:
            # Normal velocity update with speed wobble
            # v[t+1] = v[t] + (1/m)*f_net(t) + N(0, 0.01*v[t]^2)
            speed_wobble = np.random.normal(0, np.sqrt(0.01 * self.v**2))
            self.v = self.v + (1.0 / self.m) * f_net * dt + speed_wobble

        return crashed

    def update_position(self, dt: float = 1.0) -> tuple[float, float]:
        """
        Update position with sensor noise.

        y[t+1] = y[t] + v[t] (true position)
        y_sensed = y_true + N(0, 0.25*v[t]^2) (sensed position with noise)

        Parameters:
        -----------
        dt : float
            Time step (default 1.0 for discrete time)

        Returns:
        --------
        tuple[float, float]
            (true_position, sensed_position)
        """
        # Position update (true position)
        y_true = self.y + self.v * dt

        # Add sensor noise to get sensed position
        # Sensor noise: N(0, 0.25*v[t]^2)
        sensor_noise = np.random.normal(0, np.sqrt(0.25 * self.v**2))
        y_sensed = y_true + sensor_noise

        # Update internal state
        self.y = y_true

        return y_true, y_sensed

    def update(self, f_i: float, dt: float = 1.0) -> tuple[float, float, bool]:
        """
        Update the state of the particle system for one time step.

        Orchestrates the update of both velocity and position using
        the abstracted update functions.

        Parameters:
        -----------
        f_i : float
            Applied force at current time step
        dt : float
            Time step (default 1.0 for discrete time)

        Returns:
        --------
        tuple[float, float, bool]
            (sensed_position, true_position, crashed)
        """
        # Compute net force
        f_net = self.compute_net_force(f_i)

        # Update velocity (with crash probability and speed wobble)
        crashed = self.update_velocity(f_net, dt)

        # Update position (with sensor noise)
        y_true, y_sensed = self.update_position(dt)

        # Update time
        self.time += dt

        # Record history
        self.history["time"].append(self.time)
        self.history["position"].append(self.y)
        self.history["velocity"].append(self.v)
        self.history["sensed_position"].append(y_sensed)
        self.history["applied_force"].append(f_i)
        self.history["crashed"].append(crashed)

        return y_sensed, self.y, crashed

    def get_state(self) -> tuple[float, float]:
        """Get current state (position, velocity)"""
        return self.y, self.v

    def plot_history(self):
        """Plot the history of the particle system"""
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))

        # Position plot
        axes[0].plot(
            self.history["time"],
            self.history["position"],
            label="True Position",
            linewidth=2,
        )
        axes[0].plot(
            self.history["time"],
            self.history["sensed_position"],
            label="Sensed Position",
            alpha=0.6,
            linestyle="--",
        )
        axes[0].set_ylabel("Position y(t)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title("Particle System Dynamics")

        # Velocity plot
        axes[1].plot(
            self.history["time"],
            self.history["velocity"],
            label="Velocity",
            color="orange",
            linewidth=2,
        )
        axes[1].axhline(
            y=self.v_max,
            color="r",
            linestyle="--",
            label=f"v_max = {self.v_max}",
            alpha=0.5,
        )
        axes[1].axhline(y=-self.v_max, color="r", linestyle="--", alpha=0.5)
        axes[1].set_ylabel("Velocity v(t)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Applied force plot
        axes[2].plot(
            self.history["time"],
            self.history["applied_force"],
            label="Applied Force",
            color="green",
            linewidth=2,
        )
        axes[2].set_ylabel("Applied Force f_i(t)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Crash events
        crash_times = [
            t for t, c in zip(self.history["time"], self.history["crashed"]) if c
        ]
        if crash_times:
            axes[3].scatter(
                crash_times,
                [1] * len(crash_times),
                color="red",
                s=100,
                marker="x",
                label="Crash Events",
            )
        axes[3].set_ylabel("Crash Events")
        axes[3].set_xlabel("Time t")
        axes[3].set_ylim([0, 2])
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def example_simulation_constant_force():
    """Example simulation with constant applied force"""
    print("=" * 60)
    print("Particle System Simulation - Constant Force")
    print("=" * 60)

    # Define a simple potential field (harmonic oscillator)
    # φ(y) = 0.5 * k * y^2
    k = 0.1
    phi = lambda y: 0.5 * k * y**2
    phi_derivative = lambda y: k * y

    # Initialize particle system
    particle = ParticleSystem(
        mass=1.0,
        y_0=0.0,
        v_0=0.0,
        v_max=10.0,
        p_c=0.1,
        phi=phi,
        phi_derivative=phi_derivative,
    )

    print(f"Initial state: y={particle.y:.3f}, v={particle.v:.3f}")
    print(
        f"System parameters: m={particle.m}, v_max={particle.v_max}, p_c={particle.p_c}"
    )
    print("\nRunning simulation with constant force f_i = 0.5...")

    # Run simulation
    num_steps = 100
    f_i = 0.5  # Constant applied force

    for step in range(num_steps):
        y_sensed, y_true, crashed = particle.update(f_i)

        if crashed:
            print(f"  Step {step + 1}: CRASH! Position reset to {y_true:.3f}")

    print(f"\nFinal state: y={particle.y:.3f}, v={particle.v:.3f}")
    print(f"Total crashes: {sum(particle.history['crashed'])}")

    # Plot results
    particle.plot_history()


def example_simulation_interactive():
    """Example simulation with varying applied force"""
    print("=" * 60)
    print("Particle System Simulation - Varying Force")
    print("=" * 60)

    # No potential field for simplicity
    particle = ParticleSystem(mass=1.0, y_0=0.0, v_0=0.0, v_max=5.0, p_c=0.15)

    print(f"Initial state: y={particle.y:.3f}, v={particle.v:.3f}")
    print("\nRunning simulation with varying force pattern...")

    # Run simulation with varying force
    num_steps = 150

    for step in range(num_steps):
        # Varying force pattern: sine wave + some constant
        f_i = 0.5 * np.sin(step / 10) + 0.3

        y_sensed, y_true, crashed = particle.update(f_i)

    print(f"\nFinal state: y={particle.y:.3f}, v={particle.v:.3f}")
    print(f"Total crashes: {sum(particle.history['crashed'])}")

    # Plot results
    particle.plot_history()


def main():
    """Main entry point for the particle system simulation"""
    print("Particle on a Number Line - Week 1 Implementation")
    print("Based on 209as course specifications\n")

    # Run example simulations
    print("Select simulation mode:")
    print("1. Constant Force with Harmonic Potential")
    print("2. Varying Force without Potential")
    print("3. Both")

    choice = input("\nEnter choice (1/2/3) [default: 3]: ").strip()

    if choice == "1":
        example_simulation_constant_force()
    elif choice == "2":
        example_simulation_interactive()
    else:
        example_simulation_constant_force()
        print("\n" + "=" * 60 + "\n")
        example_simulation_interactive()


if __name__ == "__main__":
    main()
