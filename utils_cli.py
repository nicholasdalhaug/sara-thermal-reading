from pathlib import Path

import typer

from sara_thermal_reading.dev_utils.create_reference_polygon import (
    create_reference_polygon,
)
from sara_thermal_reading.visualization.plotting import plot_fff_from_path

app = typer.Typer()


@app.command()
def plot_fff(
    file_path: Path,
    polygon_json_path: Path = typer.Option(
        None, help="Path to the polygon JSON file to plot"
    ),
) -> None:
    plot_fff_from_path(file_path, polygon_json_path)


@app.command()
def create_polygon(
    fff_image_path: Path = typer.Argument(
        ..., help="Path to the thermal image (FFF file)"
    ),
    polygon_json_output_path: Path = typer.Option(
        Path("reference_polygon.json"), help="Path to save the polygon JSON"
    ),
) -> None:
    create_reference_polygon(fff_image_path, polygon_json_output_path)


if __name__ == "__main__":
    app()
