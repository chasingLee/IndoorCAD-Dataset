# IndoorCAD-Dataset

This repo contains our code when processing the data of the dataset-IndoorCAD, including data-processing,model-alignment,Scene-sys and statistic collector. 

The dataset processed is storaged at huggingface https://huggingface.co/datasets/anon-neurips-2026/IndoorCAD . 

We provide a folder of small sample in this repository (./Sample),allowing to easily download and preview the dataset content. 

---


## Pipeline

The full pipeline runs in the following order:

```
CAD_PROCESS(process the original data to STEP\MESH\Brep,as well as rendering the model)  →  
Align_model(the process and prompt of aligning the furniture model with VLM)  →  
Scene_sys(the pipline of transfering 3D-FROMT layout to cad layout,furniture retrivel and placement,rendering)  →  
Statistic_collect
```

---

## Environment Setup

This project requires **Python 3.10** and the `Makedataset` conda environment.

### 1. Install Conda (if not already installed)

Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.

### 2. Create the environment from the config file

```bash
conda env create -f environment.yml
```

### 3. Activate the environment

```bash
conda activate Makedataset
```

### Additional Requirements
- SolidWorks 2025 (required by CAD_PROCESS) 
- A running VLM API endpoint (required by Align_model) 

---

## Module Usage

### 1. CAD Processing

**Location:** `CAD_PROCESS/`

| Script | Description |
|---|---|
| `CAD_proccess_new.py` | Main CAD processing script via SolidWorks API |
| `Category_Proccess.py` | category the data with the label return by VLM |
| `Category_2.py` | Employing the VLM to category and discreption |
| `Render.py` | Renders CAD models to images |

**Usage:**

**Configuration** — edit the variables at the top of each script before running:

| Variable | Script | Description |
|---|---|---|
| `IN` / `OUT` | `CAD_proccess_new.py` | Input raw CAD folder and output dataset folder |
| `LIST_FILE_PATH` / `SOURCE_BASE_DIR` / `TARGET_*` | `Category_Proccess.py` | Source CSV, input dir, output dirs |
| `INPUT_DIRECTORY` | `Render.py` | Folder of processed models to render |
| `API_KEY` | `Category_2.py` | Your VLM API key |

```bash
python CAD_PROCESS/CAD_proccess_new.py
```

---

### 2. Alignment Model

**Location:** `Align_model/`

Uses a two-stage VLM (Vision-Language Model) pipeline to determine the correct upright orientation and front-facing direction of each 3D model.

| Script | Description |
|---|---|
| `Align_model_v2.py` | Automated two-stage VLM alignment (standing + front direction) |
| `manual_align.py` | Manual alignment tool for review/correction |

**Usage:**

```bash
# Process all models
python Align_model/Align_model_v2.py

# Process a specific range (e.g., models 1–500)
python Align_model/Align_model_v2.py 1 500
```

**Configuration** — edit the variables at the top of `Align_model_v2.py`:

| Variable | Description |
|---|---|
| `DATASET_ROOT` | Root folder of the categorised dataset |
| `API_KEY` | Your VLM API key |
| `URL` | Your VLM API endpoint URL |

---

### 3. Scene System

**Location:** `Scene_sys/`

Assembles aligned furniture models into complete room scenes with layout planning and rendering.

| Script | Description |
|---|---|
| `Scene_sys_v2.py` | Main scene assembly and STEP export |
| `front_to_cad.py` | Transfering 3D-FRONT layout dataset to cad layout |
| `index_assets.py` | Indexes available furniture assets |
| `layout_variants.py` | Generates layout variations |
| `preview_layouts.py` | Previews generated layouts |
| `select_templates.py` | Selects room layout templates |
| `render_scenes.py` | Renders final scenes to images |

**Configuration** — edit the path variables at the top of each script:

| Variable | Script | Description |
|---|---|---|
| `FRONT_JSON_DIR` | `front_to_cad.py`, `layout_variants.py`, `select_templates.py` | 3D-FRONT JSON layout folder |
| `MODEL_INFO_PATH` | `front_to_cad.py`, `layout_variants.py`, `select_templates.py` | 3D-FUTURE `model_info.json` |
| `OUTPUT_LAYOUT_DIR` | `front_to_cad.py` | Output CAD layout folder |
| `INDEX_JSON_PATH` | `layout_variants.py` | CAD asset index JSON |
| `LAYOUT_DIR` / `OUTPUT_DIR` | `Scene_sys_v2.py` | Input layout folder and output scene folder |
| `SCENE_DIR` / `OUTPUT_DIR` | `render_scenes.py` | Input STEP scenes and output renders |

**Usage:**

```bash
python Scene_sys/Scene_sys_v2.py

# Limit the number of scenes to process
python Scene_sys/Scene_sys_v2.py --max 100
```

---

### 4. Statistics Collection

**Location:** `Statistic_collect/`

Collects and visualizes geometric and topological statistics over the generated dataset.

| Script | Description |
|---|---|
| `furniture_stats_collect.py` | Collects per-furniture statistics (topology, geometry, mesh metrics) |
| `scene_stats_collect.py` | Collects per-scene statistics |
| `plot_all.py` | Plots all collected statistics |

**Usage:**

```bash
python Statistic_collect/furniture_stats_collect.py
python Statistic_collect/scene_stats_collect.py
python Statistic_collect/plot_all.py
```

**Configuration** — edit path variables at the top of each script (`DATASET_ROOT`, `OUTPUT_JSON`, `BASE`, etc.) to point to your local dataset and output directories.

---

## Directory Structure

```
.
├── Align_model/
│   ├── Align_model_v2.py
│   └── manual_align.py
├── CAD_PROCESS/
│   ├── CAD_proccess_new.py
│   ├── Category_2.py
│   ├── Category_Proccess.py
│   └── Render.py
├── Scene_sys/
│   ├── Scene_sys_v2.py
│   ├── front_to_cad.py
│   ├── index_assets.py
│   ├── layout_variants.py
│   ├── preview_layouts.py
│   ├── render_scenes.py
│   └── select_templates.py
├── Statistic_collect/
│   ├── furniture_stats_collect.py
│   ├── plot_all.py
│   └── scene_stats_collect.py
├── environment.yml
└── Readme.md
```

---

## Notes

- **Windows only**: `CAD_PROCESS` relies on the SolidWorks COM API via `pywin32` and only runs on Windows with SolidWorks 2025 installed.
- **Hardcoded paths**: All scripts have path configuration variables at the top (e.g. `DATASET_ROOT`, `IN`, `OUT`). These must be updated to match your local directory layout before running.
- **VLM API**: `Align_model_v2.py` and `Category_2.py` require a VLM API endpoint and key. Set `API_KEY` and `URL` at the top of each script.
- **3D-FRONT data**: `Scene_sys` scripts expect the [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) JSON layout files and the [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future) `model_info.json`.
