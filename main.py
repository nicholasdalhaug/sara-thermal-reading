import json
from pathlib import Path

import typer
from loguru import logger

from sara_thermal_reading.file_io.blob import BlobStorageLocation
from sara_thermal_reading.main_workflow import run_thermal_reading_workflow
from sara_thermal_reading.visualization.plotting import plot_fff_from_path

app = typer.Typer()


@app.command()
def plot_fff(file_path: Path) -> None:
    plot_fff_from_path(file_path)


@app.command()
def run_thermal_reading(
    anonymized_blob_storage_location: str = typer.Option(
        ..., help="JSON string for anonymized data blob storage location"
    ),
    visualized_blob_storage_location: str = typer.Option(
        ..., help="JSON string for visualized data blob storage location"
    ),
    tag_id: str = typer.Option(..., help="Tag ID"),
    inspection_description: str = typer.Option(..., help="Inspection description"),
    installation_code: str = typer.Option(..., help="Installation code"),
    temperature_output_file: str = typer.Option(
        "/tmp/temperature_output.txt", help="Temperature output file path"
    ),
) -> None:
    try:
        anonymized_location = BlobStorageLocation.model_validate(
            json.loads(anonymized_blob_storage_location)
        )
        visualized_location = BlobStorageLocation.model_validate(
            json.loads(visualized_blob_storage_location)
        )
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON provided: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Error parsing input: {e}")
        raise typer.Exit(code=1)

    run_thermal_reading_workflow(
        anonymized_location,
        visualized_location,
        tag_id,
        inspection_description,
        installation_code,
        temperature_output_file,
    )


if __name__ == "__main__":
    app()
