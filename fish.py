
# eigenfishes.py

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rcParams

# Configure matplotlib to use LaTeX for rendering text
rcParams['text.usetex'] = True
rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amsfonts}"
# Uncomment the following line if you're using Jupyter Notebook
# %config InlineBackend.figure_format = 'retina'

# ------------------------------
# Eigenfish Class Definition
# ------------------------------

class Eigenfish:
    """
    A class for building an eigenfish plot.

    Parameters:
        matrix (ndarray): A two-dimensional square matrix.
        indices_of_ts (tuple of arrays): Indices of elements in the matrix to be replaced by parameters t_1, t_2, ..., t_n.
                                         For example, (array([0, 5]), array([6, 3])) means that elements (0,6) and (5,3) are replaced by t_1 and t_2.
    """

    def __init__(self, matrix, indices_of_ts):
        self.matrix = matrix.copy()  # Ensure the original matrix is not modified
        self.indices_of_ts = indices_of_ts  # Indices of parameters
        self.mdim = matrix.shape[0]  # Dimension of the matrix
        self.n_t = len(self.indices_of_ts[0])
        self.is_matrix_of_phases = np.all(np.abs(self.matrix) == 1.0)

    def eigvals_random_ts_rect(self, n_ts, r):
        """
        Calculate eigenvalues by sampling parameters from a uniform distribution within [-r, r].

        Parameters:
            n_ts (int): Number of sampled tuples (t_1, t_2, ..., t_n).
            r (float): Support range of the uniform distribution.

        Returns:
            ndarray: Eigenvalues with shape (n_ts * self.mdim,).
        """
        eigenvalues = np.zeros(n_ts * self.mdim, dtype=np.complex128)
        for i in range(n_ts):
            ts = tuple(np.random.uniform(-r, r, self.n_t))
            self.matrix[self.indices_of_ts] = ts
            eigenvalues[i * self.mdim:(i + 1) * self.mdim] = np.linalg.eigvals(self.matrix)
        return eigenvalues

    def eigvals_normal_ts_rect(self, n_ts, s):
        """
        Calculate eigenvalues by sampling parameters from a normal distribution with standard deviation s.

        Parameters:
            n_ts (int): Number of sampled tuples (t_1, t_2, ..., t_n).
            s (float): Standard deviation of the normal distribution.

        Returns:
            ndarray: Eigenvalues with shape (n_ts * self.mdim,).
        """
        eigenvalues = np.zeros(n_ts * self.mdim, dtype=np.complex128)
        for i in range(n_ts):
            ts = tuple(np.random.normal(0, s, self.n_t))
            self.matrix[self.indices_of_ts] = ts
            eigenvalues[i * self.mdim:(i + 1) * self.mdim] = np.linalg.eigvals(self.matrix)
        return eigenvalues

    def eigvals_random_ts_torus(self, n_ts, radius=1.0):
        """
        Calculate eigenvalues by sampling parameters uniformly on a torus.

        Parameters:
            n_ts (int): Number of sampled tuples (t_1, t_2, ..., t_n).
            radius (float): Radius of the torus for each dimension.

        Returns:
            ndarray: Eigenvalues with shape (n_ts * self.mdim,).
        """
        eigenvalues = np.zeros(n_ts * self.mdim, dtype=np.complex128)
        for i in range(n_ts):
            ts = tuple(radius * np.exp(1j * np.random.uniform(0., 2 * np.pi, self.n_t)))
            self.matrix[self.indices_of_ts] = ts
            eigenvalues[i * self.mdim:(i + 1) * self.mdim] = np.linalg.eigvals(self.matrix)
        return eigenvalues

    def gershgorin_circles(self):
        """
        Calculate Gershgorin circles for the matrix.

        Returns:
            tuple of ndarrays: (x_centers, y_centers, radii)
        """
        if self.is_matrix_of_phases:
            diag_elms = np.diag(self.matrix)
            radiuses = (self.mdim - 1) * np.ones(self.mdim)
            return np.real(diag_elms), np.imag(diag_elms), radiuses
        else:
            return None

    def latex_matrix(self, max_real="1"):
        """
        Generate a LaTeX representation of the matrix.

        Parameters:
            max_real (str): Maximum real value for parameter replacement.

        Returns:
            str: LaTeX matrix string.
        """
        def clean(val):
            real, imag = np.real(val), np.imag(val)
            if imag != 0:
                return "i" if imag > 0 else "-i"
            real_str = str(real).replace(".0", "")
            return real_str if real_str in ["0", "0.5", max_real] else "tT"

        template = r" \\ ".join([" & ".join(["{}"] * self.mdim)] * self.mdim)
        filled = template.format(*map(clean, self.matrix.flatten()))
        final = filled.replace("T", "{}").format(*range(1, self.n_t + 1))
        return r"\begin{pmatrix} " + final + r" \end{pmatrix}"

    def create_simple_latex_title_rect(self, rmin, rmax):
        """
        Create a simple LaTeX title for rectangular sampling.

        Parameters:
            rmin (float): Minimum range value.
            rmax (float): Maximum range value.

        Returns:
            str: LaTeX-formatted title string.
        """
        title = self.latex_matrix()
        return f"$ {title} $"

    def create_latex_title_rect(self, rmin, rmax):
        """
        Create a detailed LaTeX title for rectangular sampling.

        Parameters:
            rmin (float): Minimum range value.
            rmax (float): Maximum range value.

        Returns:
            str: LaTeX-formatted title string.
        """
        title = (
            r"\lambda \in \mathbb{C} | \det(A-\lambda I)=0, \quad A="
            + self.latex_matrix()
            + r", \quad "
            + ", ".join([f"t{n+1}" for n in range(self.n_t)])
            + f" \in ({rmin}, {rmax}) "
        )
        return f"$ {title} $"

    def create_latex_title_torus(self):
        """
        Create a LaTeX title for torus sampling.

        Returns:
            str: LaTeX-formatted title string.
        """
        title = (
            r"\lambda \in \mathbb{C} | \det(A-\lambda I)=0, \quad A="
            + self.latex_matrix(max_real="0.2")
            + r", \quad "
            + ", ".join([f"t{n+1}" for n in range(self.n_t)])
            + r" \in S^{1} \times S^{1}"
        )
        return f"$ {title} $"

# ------------------------------
# Helper Functions
# ------------------------------

def configure_axes(ax):
    """
    Configure the axes by removing spines and setting aspect.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to configure.
    """
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.set_aspect('equal', 'box')
    ax.set_axis_off()

def annotate_figure(fig, text, position, fontsize=12):
    """
    Annotate the figure with given text.

    Parameters:
        fig (matplotlib.figure.Figure): The figure to annotate.
        text (str): Text to add.
        position (tuple): Position tuple (x, y).
        fontsize (int): Font size of the annotation.
    """
    plt.annotate(text, position, xycoords="figure points", fontsize=fontsize)

# ------------------------------
# Plotting Functions
# ------------------------------

def plot_random_matrices():
    """
    Plot eigenvalues for randomly generated matrices with rectangular sampling.
    """
    mdim = 5
    population = np.array([0., -1.j, 1.j, 1., 0.5])
    r = 20
    n_matrix = 100000
    list_of_figure = []

    fig, axs = plt.subplots(3, 3, figsize=(20, 20))
    fig.set_facecolor("#f4f0e8")
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    for ax in axs.reshape(-1):
        configure_axes(ax)

        # Generate random matrix
        matrix = np.random.choice(population, (mdim, mdim)) + 0.j

        # Select two unique indices to vary
        var_indices = np.unravel_index(
            np.random.choice(np.arange(0, mdim**2), 2, replace=False),
            (mdim, mdim)
        )

        # Create Eigenfish instance
        eigenfish = Eigenfish(matrix, var_indices)
        list_of_figure.append(eigenfish)

        # Compute eigenvalues
        eigenvalues = eigenfish.eigvals_random_ts_rect(n_matrix, r)

        # Scatter plot of eigenvalues
        ax.scatter(
            np.real(eigenvalues),
            np.imag(eigenvalues),
            c="#383b3e",
            s=0.03,
            linewidths=0.0001,
            alpha=1.0
        )

        # Set title (simple LaTeX)
        ax.set_title(
            eigenfish.create_simple_latex_title_rect(-r, r),
            fontsize=14
        )

        # Adjust plot limits if necessary
        if np.max(np.real(eigenvalues)) > 10:
            ax.set_xlim([-8, 8])

    plt.tight_layout()
    plt.savefig("random_matrices.png", bbox_inches="tight", dpi=300)
    plt.close()

def plot_zoomed_matrix(list_of_figure, chosen=2, n_matrix=500000, r=20):
    """
    Plot a zoomed-in view of a selected eigenfish.

    Parameters:
        list_of_figure (list): List of Eigenfish instances.
        chosen (int): Index of the Eigenfish to zoom in on.
        n_matrix (int): Number of matrices to sample.
        r (float): Range parameter for sampling.
    """
    eigenfish = list_of_figure[chosen]

    fig, ax = plt.subplots(figsize=(20, 20))
    fig.set_facecolor("#f4f0e8")
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    configure_axes(ax)

    # Compute eigenvalues
    eigenvalues = eigenfish.eigvals_random_ts_rect(n_matrix, r)

    # Scatter plot
    ax.scatter(
        np.real(eigenvalues),
        np.imag(eigenvalues),
        c="#383b3e",
        s=0.02,
        linewidths=0.0001,
        alpha=1.0
    )

    # Set title (simple LaTeX)
    ax.set_title(
        eigenfish.create_simple_latex_title_rect(-r, r),
        fontsize=14
    )

    plt.tight_layout()
    annotate_figure(fig, "Simone Conradi, 2023", (1100., 12.), fontsize=12)
    plt.show()

def plot_torus_case():
    """
    Plot eigenvalues for randomly generated matrices with torus sampling.
    """
    mdim = 6
    population = np.array([0., 0., -1.j, 1.j, 0.2])
    r = 20
    n_matrix = 100000
    list_of_figure = []

    fig, axs = plt.subplots(3, 3, figsize=(20, 20))
    fig.set_facecolor("#f4f0e8")
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    for ax in axs.reshape(-1):
        configure_axes(ax)

        # Generate random matrix
        matrix = np.random.choice(population, (mdim, mdim)) + 0.j

        # Select two unique indices to vary
        var_indices = np.unravel_index(
            np.random.choice(np.arange(0, mdim**2), 2, replace=False),
            (mdim, mdim)
        )

        # Create Eigenfish instance
        eigenfish = Eigenfish(matrix, var_indices)
        list_of_figure.append(eigenfish)

        # Compute eigenvalues
        eigenvalues = eigenfish.eigvals_random_ts_torus(n_matrix)

        # Scatter plot of eigenvalues
        ax.scatter(
            np.real(eigenvalues),
            np.imag(eigenvalues),
            c="#383b3e",
            s=0.02,
            linewidths=0.0001,
            alpha=1.0
        )

        # Set title (detailed LaTeX)
        ax.set_title(
            eigenfish.create_latex_title_torus(),
            fontsize=15
        )

    plt.tight_layout()
    plt.savefig("torus_case.png", bbox_inches="tight", dpi=300)
    plt.close()

def plot_single_torus(list_of_figure, chosen=0, n_matrix=500000):
    """
    Plot a single torus case with a zoomed-in view.

    Parameters:
        list_of_figure (list): List of Eigenfish instances.
        chosen (int): Index of the Eigenfish to plot.
        n_matrix (int): Number of matrices to sample.
    """
    eigenfish = list_of_figure[chosen]

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.set_facecolor("#f4f0e8")
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    configure_axes(ax)

    # Compute eigenvalues
    eigenvalues = eigenfish.eigvals_random_ts_torus(n_matrix)

    # Scatter plot
    ax.scatter(
        np.real(eigenvalues),
        np.imag(eigenvalues),
        c="#383b3e",
        s=0.05,
        linewidths=0.0001,
        alpha=1.0
    )

    # Set title (detailed LaTeX)
    ax.set_title(
        eigenfish.create_latex_title_torus(),
        fontsize=12
    )

    plt.tight_layout()
    annotate_figure(fig, "Simone Conradi, 2023", (1100., 12.), fontsize=12)
    plt.show()

def create_video_frames():
    """
    Create frames for a video by varying matrix parameters on a torus.
    """
    mdim = 6
    n_frames = 100
    n_initial_frames = 10
    sym_p_values = np.zeros(n_frames)
    sym_p_values[n_initial_frames:] = np.linspace(0., 2., n_frames - n_initial_frames)
    
    # Define the initial matrix and variable indices
    matrix = np.array([
        [-1.j, 0,    0,   0,    0,    0],
        [0,    0,    1,   1,    0,    1],
        [1.j,  0,    0,   1,    0,    1.j],
        [0,    0,    0, -1.j, -1.j, 1.j],
        [-1.j, 1,    0,   0, -1.j, 1],
        [-1.j, 1,    0,   0,    0,  -1.j]
    ])
    var_indices = (np.array([0, 0]), np.array([2, 5]))
    eigenfish = Eigenfish(matrix, var_indices)

    n_matrix = 100000

    for i in range(n_frames):
        out_path = f"./image{i:04d}.png"

        fig, ax = plt.subplots(figsize=(4, 4))
        fig.set_facecolor("#f4f0e8")
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        configure_axes(ax)

        # Update matrix parameters
        sym_p = sym_p_values[i]
        updated_matrix = np.array([
            [-1.j,    0,    0,      0,      0,      0],
            [0,       0, sym_p, sym_p,    0,   sym_p],
            [1.j,     0,    0,   sym_p,    0,    1.j],
            [0,       0,    0,   -1.j,  -1.j,    1.j],
            [-1.j, sym_p,    0,      0,  -1.j,  sym_p],
            [-1.j, sym_p,    0,      0,      0,  -1.j]
        ])
        eigenfish.matrix = updated_matrix.copy()

        # Compute eigenvalues
        eigenvalues = eigenfish.eigvals_random_ts_torus(n_matrix)

        # Scatter plot
        ax.scatter(
            np.real(eigenvalues),
            np.imag(eigenvalues),
            c="#383b3e",
            s=0.05,
            linewidths=0.0001,
            alpha=1.0
        )

        # Set title with LaTeX (detailed)
        ax.set_title(
            r"$ \{ \lambda \in \mathbb{C} | \det(A-\lambda I)=0, \quad A=\left( \begin{array}{cccccc}"
            r"-i & 0 & t_{1} & 0 & 0 & t_{2} \\\\"
            r"0 & 0 & x & x & 0 & x \\\\"
            r"i & 0 & 0 & x & 0 & i \\\\"
            r"0 & 0 & 0 & -i & -i & i \\\\"
            r"-i & x & 0 & 0 & -i & x \\\\"
            r"-i & x & 0 & 0 & 0 & -i \\\\"
            r"\end{array} \right) , \quad t_1,t_2 \in S^{1} \times S^{1} \}$",
            fontsize=7
        )

        # Set plot limits
        ax.set_xlim([-2.208098042011261, 2.3700588822364805])
        ax.set_ylim([-2.9683619290590286, 0.9622942358255386])

        # Annotate the figure
        annotate_figure(fig, "Simone Conradi, 2023", (250., 5.), fontsize=5)
        annotate_figure(fig, r"$x = %s$" % str(np.round(sym_p, 2)), (20., 5.), fontsize=12)

        # Save the figure
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close()

# ------------------------------
# Main Execution
# ------------------------------

if __name__ == "__main__":
    # Create a LaTeX file to store the plots
    with open("eigenfish_plots.tex", "w") as latex_file:
        # Write LaTeX preamble
        latex_file.write("\\documentclass{article}\n")
        latex_file.write("\\usepackage{graphicx}\n")
        latex_file.write("\\usepackage{subfigure}\n")
        latex_file.write("\\begin{document}\n")
        
        # Plot random matrices with rectangular sampling
        plot_random_matrices()
        latex_file.write("\\begin{figure}[htbp]\n")
        latex_file.write("\\centering\n")
        latex_file.write("\\includegraphics[width=\\textwidth]{random_matrices.png}\n")
        latex_file.write("\\caption{Random matrices with rectangular sampling}\n")
        latex_file.write("\\end{figure}\n")
        
        # Plot torus case
        plot_torus_case()
        latex_file.write("\\begin{figure}[htbp]\n")
        latex_file.write("\\centering\n")
        latex_file.write("\\includegraphics[width=\\textwidth]{torus_case.png}\n")
        latex_file.write("\\caption{Torus case}\n")
        latex_file.write("\\end{figure}\n")
        
        # Close the LaTeX document
        latex_file.write("\\end{document}\n")
    
    print("LaTeX file 'eigenfish_plots.tex' has been created.")
