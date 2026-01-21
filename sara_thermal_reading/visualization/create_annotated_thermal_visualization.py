from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from numpy.typing import NDArray


def create_annotated_thermal_visualization(
    aligned_image: NDArray[np.float64],
    polygon_points: NDArray[np.float32],
    tempreature: float,
) -> NDArray[np.uint8]:

    plt.figure()
    plt.clf()
    fig = plt.gcf()
    fig.set_figwidth(6.5)
    fig.set_figheight(4)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax = plt.gca()
    plt.imshow(aligned_image, cmap="RdYlBu_r")
    patch = plt.Polygon(
        polygon_points,
        closed=True,
        fill=None,
        edgecolor="gray",
        linewidth=2,
        label=f"Median temperature: {tempreature:.2f}°C",
    )
    ax.add_patch(patch)
    ax.legend(handles=[patch])
    plt.axis("off")
    plt.colorbar(fraction=0.15, pad=0.02, label="Temperature (°C)")

    fig.canvas.draw()
    # Mypy does not see that canvas is FigureCanvasAgg and buffer_rgba exists
    data_rgba = np.asarray(fig.canvas.buffer_rgba())  # type: ignore[attr-defined]
    data_rgb = data_rgba[:, :, :3]
    plt.close(fig)

    return data_rgb
