import numpy as np
import seaborn as sns
from IPython import display
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance_matrix
from skimage import io
from tqdm import tqdm as progress_bar


M = 64  # number of time-step in the simulated sequence
N = 3  # number of propeller wings


def propeller(theta, m):  # Symulujemy śmigło za pomocą poniższej funkcji
    return np.sin(N*theta + m * np.pi / 10)


thetas = np.linspace(-np.pi, np.pi, 1000)
r = propeller(thetas, m=0)


figure = plt.figure(figsize=[5, 5])
plot = plt.polar([])[0]  # empty polar figure


def animate(frame):
    propeller_curve = propeller(thetas, m=frame)
    plot.set_data((thetas, propeller_curve))


animation = FuncAnimation(figure, animate, frames=100, interval=25)

plt.show()


"""_______________________________________________________________________"""
# Sensor part

thetas = np.linspace(-np.pi, np.pi, 1000)  # linspace of angles to plot
rs = propeller(thetas, m=0)


def polar_to_cartesian(theta, r):
    """Convert polar coordinate arrays to cartesian"""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


def capture(func: np.array, resolution: int, threshold: float = 0.1, low: float = -1, high: float = 1) -> np.array:

    grid_x, grid_y = np.meshgrid(np.linspace(low, high, resolution), np.linspace(low, high, resolution))
    grid = np.column_stack([grid_x.flatten(), grid_y.flatten()])

    distances = distance_matrix(grid, func)
    capture = (np.min(distances, axis=1) <= threshold).astype(int).reshape(resolution, resolution)

    return capture


_ = plt.imshow(capture(polar_to_cartesian(thetas, rs), resolution=256, threshold=0.05, low=-np.pi / 2, high=np.pi / 2), cmap="Greys")

plt.show()


# Moving propeller part

thetas = np.linspace(-np.pi, np.pi, 100)  # linspace of angles to plot
ms = propeller(thetas, m=64)


funcs = []

for m in ms.tolist():
    r = propeller(thetas, m=m)
    func = polar_to_cartesian(thetas, r)
    funcs.append(func)

funcs = np.asarray(funcs)


def record(funcs: list, capture_kwargs) -> np.array:
    """Simulate recording by applying capture in loop"""
    return np.asarray([capture(func, **capture_kwargs) for func in progress_bar(funcs)])


recording = record(funcs, capture_kwargs=dict(resolution=256, threshold=0.05, low=-np.pi / 2, high=np.pi / 2))
recording.shape


offset = 0
length = 4  # number of lines CMOS reads per frame
capture = np.zeros([256, 256])  # initialize empty image

for frame in recording:
    capture[offset : offset + length, :] = frame[offset : offset + length, :]
    offset += length

plt.imshow(capture, cmap="Greys")

plt.show()

plt.savefig('propeller_sim.png')