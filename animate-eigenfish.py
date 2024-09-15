
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams

# Configure matplotlib to use LaTeX for rendering text
rcParams['text.usetex'] = True
rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amsfonts}"

# ------------------------------
# Eigenfish Class Definition
# ------------------------------

class Eigenfish:
    """
    A class for building an eigenfish plot.

    Parameters:
        matrix (ndarray): A two-dimensional square matrix.
        indices_of_ts (tuple of arrays): Indices of elements in the matrix to be replaced by parameters t_1, t_2, ..., t_n.
    """

    def __init__(self, matrix, indices_of_ts):
        self.matrix = matrix.copy()
        self.indices_of_ts = indices_of_ts
        self.mdim = matrix.shape[0]
        self.n_t = len(self.indices_of_ts[0])

    def eigvals_random_ts(self, ts):
        """
        Calculate eigenvalues for given parameters ts.

        Parameters:
            ts (tuple): Tuple of parameter values.

        Returns:
            ndarray: Eigenvalues of the matrix with substituted ts.
        """
        self.matrix[self.indices_of_ts] = ts
        return np.linalg.eigvals(self.matrix)

# ------------------------------
# Animation Function
# ------------------------------

def animate_eigenfish():
    """
    Generate and animate random eigenfish morphing into each other.
    """
    mdim = 5  # Dimension of the matrices
    n_frames = 200  # Total number of frames in the animation

    # Create a figure and axis for the animation
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#f4f0e8")
    ax.set_aspect('equal', 'box')
    ax.axis('off')

    # Initialize scatter plot
    scatter = ax.scatter([], [], c="#383b3e", s=0.5, alpha=0.8)

    # Function to generate a random Eigenfish
    def generate_random_eigenfish():
        population = np.array([0., -1.j, 1.j, 1., 0.5])
        matrix = np.random.choice(population, (mdim, mdim)) + 0.j
        var_indices = np.unravel_index(
            np.random.choice(np.arange(0, mdim**2), 2, replace=False),
            (mdim, mdim)
        )
        return Eigenfish(matrix, var_indices)

    # Generate two random Eigenfish to morph between
    eigenfish_start = generate_random_eigenfish()
    eigenfish_end = generate_random_eigenfish()

    # Generate random parameters for start and end
    ts_start = np.random.uniform(-10, 10, eigenfish_start.n_t)
    ts_end = np.random.uniform(-10, 10, eigenfish_end.n_t)

    # Interpolation function between start and end parameters
    def interpolate_ts(t):
        return ts_start * (1 - t) + ts_end * t

    # Animation update function
    def update(frame):
        t = frame / (n_frames - 1)
        ts = interpolate_ts(t)
        eigenvalues = eigenfish_start.eigvals_random_ts(ts)

        # Update scatter plot data
        scatter.set_offsets(np.c_[np.real(eigenvalues), np.imag(eigenvalues)])

        # Clear and set limits
        ax.cla()
        ax.set_facecolor("#f4f0e8")
        ax.set_aspect('equal', 'box')
        ax.axis('off')

        # Re-plot the eigenvalues
        ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), c="#383b3e", s=1, alpha=0.8)

        # Optional: Add titles or annotations here
        ax.set_title("Morphing Eigenfish", fontsize=16)

        return scatter,

    # Create the animation
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=50, blit=True
    )

    # Save the animation as a GIF (optional)
    anim.save('eigenfish_animation.gif', writer='pillow', fps=20)

    # Display the animation
    plt.show()

# ------------------------------
# Main Execution
# ------------------------------

if __name__ == "__main__":
    animate_eigenfish()
