# ParaView Catalyst In-Situ Plugin

In-situ mode lets you inspect the running simulation interactively in ParaView
without writing any intermediate files to disk.

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| ParaView ≥ 5.11 (with Catalyst support) | Must be built with `PARAVIEW_USE_CATALYST=ON` |
| ADIOS2 ≥ 2.9 (with plugin engine support) | Must include the Fides / ParaView in-situ plugin |
| `CATALYST_IMPLEMENTATION_PATHS` set | Points to the ParaView Catalyst libraries |
| `ADIOS2_PLUGIN_PATH` set | Points to the directory containing the in-situ plugin `.so` |

The `setting.sh` file provides example values for these variables:

```bash
source catalyst/plugin/setting.sh
```

Adjust the paths inside `setting.sh` to match your local installation.

## Quick Start

1. **Set environment variables**

   ```bash
   source catalyst/plugin/setting.sh
   ```

2. **Run a simulation with the plugin config**

   ```bash
   ./build/sim_NSE/sim_1 --adios-config adios2-inline-plugin.xml
   ```

   The simulation will start and create a Catalyst Live connection listener.

3. **Connect from ParaView**

   Open ParaView → **Catalyst** → **Connect…** → enter the port
   (`localhost:22222`). The live pipeline will appear and update each
   time-step.
   