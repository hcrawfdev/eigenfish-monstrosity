import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams

# Configure matplotlib
rcParams['text.usetex'] = False
plt.style.use('dark_background')

class Eigenfish:
    def __init__(self, matrix, indices_of_ts):
        self.matrix = matrix.copy()
        self.indices_of_ts = indices_of_ts
        self.mdim = matrix.shape[0]
        self.n_t = len(self.indices_of_ts[0])

    def eigvals_random_ts_torus(self, n_ts, radius=1.0):
        eigenvalues = np.zeros(n_ts * self.mdim, dtype=np.complex128)
        for i in range(n_ts):
            ts = tuple(radius * np.exp(1j * np.random.uniform(0., 2 * np.pi, self.n_t)))
            self.matrix[self.indices_of_ts] = ts
            eigenvalues[i * self.mdim:(i + 1) * self.mdim] = np.linalg.eigvals(self.matrix)
        return eigenvalues

class EigenfishAnimator:
    def __init__(self, mdim=6, n_matrix=10000, morph_steps=30, morph_duration=1.5):
        self.mdim = mdim
        self.n_matrix = n_matrix
        self.morph_steps = morph_steps
        self.morph_duration = morph_duration
        self.current_eigenfish = self.generate_random_eigenfish()
        self.next_eigenfish = self.generate_random_eigenfish()
        self.transition_progress = 0
        self.frame_count = 0

    def generate_random_eigenfish(self):
        matrix = np.random.choice([0., -1.j, 1.j, 0.2], (self.mdim, self.mdim)) + 0.j
        var_indices = np.unravel_index(
            np.random.choice(np.arange(0, self.mdim**2), 2, replace=False),
            (self.mdim, self.mdim)
        )
        return Eigenfish(matrix, var_indices)

    def interpolate_matrices(self, progress):
        return (1 - progress) * self.current_eigenfish.matrix + progress * self.next_eigenfish.matrix

    def update(self, frame):
        self.frame_count += 1
        self.transition_progress = (self.frame_count % self.morph_steps) / self.morph_steps

        if self.frame_count % self.morph_steps == 0:
            self.current_eigenfish = self.next_eigenfish
            self.next_eigenfish = self.generate_random_eigenfish()

        interpolated_matrix = self.interpolate_matrices(self.transition_progress)
        interpolated_eigenfish = Eigenfish(interpolated_matrix, self.current_eigenfish.indices_of_ts)
        eigenvalues = interpolated_eigenfish.eigvals_random_ts_torus(self.n_matrix)

        plt.clf()
        plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), c='white', s=0.1, alpha=0.5)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.axis('off')
        plt.tight_layout()

    def animate(self):
        fig = plt.figure(figsize=(10, 10))
        interval = (self.morph_duration * 1000) / self.morph_steps  # Convert to milliseconds
        anim = FuncAnimation(fig, self.update, frames=None, interval=interval, blit=False)
        plt.show()

if __name__ == "__main__":
    animator = EigenfishAnimator(morph_steps=30, morph_duration=0.7)
    animator.animate()
