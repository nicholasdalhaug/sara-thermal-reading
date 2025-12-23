# SARA Thermal Reading

The **SARA Thermal Reading** provides an automated process to get the temerature in a chosen polygon in a thermal image.

## Dependencies

The dependencies used for this package are listed in `pyproject.toml` and pinned in `requirements.txt`. This ensures our builds are predictable and deterministic. This project uses `pip-compile` (from [`pip-tools`](https://github.com/jazzband/pip-tools)) for this:

```
pip-compile --output-file=requirements.txt pyproject.toml
```

To update the requirements to the latest versions, run the same command with the `--upgrade` flag:

```
pip-compile --output-file=requirements.txt pyproject.toml --upgrade
```

### Setup

For the thermal reading to run, we need a reference image and reference polygon located in a blob store.
Both the image and polygon needs to be stored in a container named `installation_code` and a folder named `tagId_inspectionDescription`. For an example see the saradev, thermalreadingrefdev storage account.

### Install locally

Install with `pip install -r requirements.txt -e .[dev]`

### Example: Run locally

Can be tested by running `python ./example_thermal_reading.py` with the parameters for source and reference correctly set in you .env file. See .env.exampel for which values should be set. Destination storage account is not used in `example_thermal_reading.py`, files are just stored locally to a "results" folder.

### Example .env.example

```bash
SOURCE_STORAGE_ACCOUNT=storageanon1dev
SOURCE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=ht ...
DESTINATION_STORAGE_ACCOUNT=storagevis1dev
DESTINATION_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=ht ...
REFERENCE_STORAGE_ACCOUNT=thermalreadingrefdev
REFERENCE_STORAGE_CONNECTION_STRING=ht ...

```


## Dev utils

### Create reference polygon

Draw polygon directly on image

```bash
python utils_cli.py create-polygon path/to/image.fff
```

Will by default save to `reference_polygon.json` in the current directory

### Plot reference polygon

```bash
python utils_cli.py plot-fff path/to/image.fff --polygon-json-path path/to/reference_polygon.json
```


