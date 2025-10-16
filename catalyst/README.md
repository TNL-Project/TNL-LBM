# Catalyst Frame Exporter Guide

This guide explains how to turn the ADIOS2 output simulations in this repository into PNG frames for post-processing. The
`tnl_lbm_export_frames.py` script reads velocity, lbm_density, and other fields from an
ADIOS2 stream SST or BP file and exports planar slices as images.

## Quick start

1. **Run a simulation** that produces an ADIOS2 output stream or BP file. Example:

         ./build/sim_NSE/sim_1  --adios-config "adios2_sst.xml"
         # or for BP output:
         ./build/sim_NSE/sim_1 --adios-config "adios2.xml"

    This will write into a directory such as `results_sim_1_res04_np001/output_3D.sst` (SST stream) or a BP file `results_sim_1_res04_np001/output_3D.bp`.
    
    **Note:** When launching the exporter for SST, use the stream directory path (e.g. `output_3D`), not the `.sst` file itself.
2. **Launch the exporter**


       # For ADIOS2 stream directory (SST):       
       python3 catalyst/tnl_lbm_export_frames.py \
           --instream results_sim_1_res04_np001/output_3D \
           --outdir frames/velocity \
           --prefix sim1 \
           --plane xy \
           --varname velocity \
           --component magnitude 
           --config 'adios2_sst.xml'

       # For BP file (static output):
       python3 catalyst/tnl_lbm_export_frames.py \
           --instream results_sim_1_res04_np001/output_3D.bp \
           --outdir frames/density \
           --prefix sim1 \
           --plane xy \
           --varname lbm_density \
           --autoscale-colour

## Command-line options

Only the most relevant options are highlighted here. Run `python3
catalyst/tnl_lbm_export_frames.py --help` for the complete list.

| Option | Description |
| --- | --- |
| `--instream` / `-i` | Path to the ADIOS2 stream directory or `.bp` file. **Required.** |
| `--outdir` | Directory where PNG files will be saved (created if missing). |
| `--prefix` | Prefix used in generated filenames. |
| `--plane` | Slice orientation: `xy`, `xz`, `yz`, or `all` (exports each plane). |
| `--plane-index` | Override plane indices, e.g. `--plane-index yz=mid` or `--plane-index xy=75%`. Supports multiple values. |
| `--varname` / `-v` | Name of the scalar or vector field (default `velocity`, also accepts e.g. `lbm_density`). |
| `--component` | For vectors: `x`, `y`, `z`, or `magnitude` (default). |
| `--figsize` & `--dpi` | Control the matplotlib canvas size and resolution. |
| `--config` | Path to ADIOS2 XML config file (default: `adios2.xml`). |
| `--vmin` / `--vmax` | Set colour scale bounds. Accepts float, 'none', or 'auto' for automatic scaling. |
| `--autoscale-colour` | Force automatic colour scaling (ignores `--vmin`/`--vmax`). |
| `--nompi` | Force serial mode even if `mpi4py`/MPI are available. |

### Colour scale control

- By default, the exporter uses fixed colour scale bounds (`--vmin`, `--vmax`).
- To let matplotlib autoscale colours per frame, use `--autoscale-colour`.


### Selecting slice positions

- Indices are interpreted along the axis perpendicular to the chosen plane.
- `--plane-index <plane>=<value>` accepts integers, percentages (e.g. `25%`),
  fractions (`0.5` corresponds to the middle), and aliases (`min`, `mid`, `max`).
- When no override is provided, the script defaults to the mid-plane.

### Running with MPI

The exporter honours MPI execution. When run under `mpirun`, all ranks participate
in reading, but only rank 0 writes PNG files:

```
mpirun -np 4 python3 catalyst/tnl_lbm_export_frames.py \
    --instream results_sim_1_res04_np001/output_3D
```


## Related files

- `tnl_lbm_export_frames.py` – the exporter entry point described in this guide.
- `tnl_lbm_common.py` – shared helpers for MPI setup, plane selection, and data
  reads used by the exporter.
