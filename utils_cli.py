from pathlib import Path

import matplotlib.pyplot as plt
import typer

from sara_thermal_reading.dev_utils.create_reference_polygon import (
    create_reference_polygon,
)
from sara_thermal_reading.dev_utils.run_fff_workflow_local_files import (
    run_fff_workflow_local_files,
)
from sara_thermal_reading.file_io.file_utils import load_reference_fff_image_and_polygon
from sara_thermal_reading.visualization.plotting import (
    plot_fff_from_path,
    plot_thermal_image,
)

app = typer.Typer()


@app.command()
def run_fff_workflow(
    polygon_path: str = typer.Option(..., help="Path to the polygon JSON file"),
    reference_image_path: str = typer.Option(
        ..., help="Path to the reference FFF image"
    ),
    source_image_path: str = typer.Option(
        None, help="Path to the source FFF image (optional)"
    ),
    tag_id: str = typer.Option("test_tag_id", help="Tag ID"),
    inspection_description: str = typer.Option(
        "test_inspection_description", help="Inspection description"
    ),
) -> None:
    run_fff_workflow_local_files(
        polygon_path,
        reference_image_path,
        source_image_path,
        tag_id,
        inspection_description,
    )


@app.command()
def plot_fff(
    file_path: Path,
    polygon_json_path: Path = typer.Option(
        None, help="Path to the polygon JSON file to plot"
    ),
) -> None:
    plot_fff_from_path(file_path, polygon_json_path)


@app.command()
def plot_current_reference_image_and_polygon(
    installation_code: str = typer.Option(..., help="Installation code"),
    tag_id: str = typer.Option(..., help="Tag ID"),
    inspection_description: str = typer.Option(..., help="Inspection description"),
) -> None:
    image, polygon_points = load_reference_fff_image_and_polygon(
        installation_code, tag_id, inspection_description
    )

    plot_thermal_image(
        image,
        f"Reference Image: {installation_code}/{tag_id}_{inspection_description}",
        polygon_points,
    )
    plt.show()


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
