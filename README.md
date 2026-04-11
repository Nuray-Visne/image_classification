This repository is for an image classification task (chicken, egg, balloon) using the Open Images dataset.

## Python virtual environment

Create a `.venv` in the project root (from PowerShell):

```powershell
cd c:\mai\2.Semester\CVAI\image_classification
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

In **Command Prompt** activation is:

```cmd
.\.venv\Scripts\activate.bat
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The notebook also uses **FiftyOne** for the Open Images zoo workflow. After the venv is active, install it in the same environment (large download):

```powershell
pip install fiftyone
```

Use this venv as the **Jupyter kernel** in Cursor/VS Code so `import` and `%pip` target `.venv` (check with `import sys; print(sys.executable)` in a cell—it should end with `.venv\Scripts\python.exe`).

## Dataset: how it is loaded and where it is saved

### Annotation CSV (`image_classification.ipynb`)

If `data/chicken_egg_balloon_annotations.csv` does not exist, the notebook downloads and builds it:

1. **Train bounding boxes:** `https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv` → saved under `data/oidv6-train-annotations-bbox.csv` (large file).
2. **Class names (boxable):** `https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv` → `data/oidv7-class-descriptions-boxable.csv`.
3. Rows are merged on `LabelName`, filtered to classes **Chicken**, **Egg**, and **Balloon**, and written to **`data/chicken_egg_balloon_annotations.csv`**.

If an older copy exists in the **project root**, prefer loading **`data/chicken_egg_balloon_annotations.csv`** in EDA cells so paths stay consistent.

### Image subset (FiftyOne zoo cell)

The notebook can load **`open-images-v7`** via `fiftyone.zoo.load_zoo_dataset` (e.g. detections, selected classes, split such as `train`), then **copy** images into the folder given by **`OUTPUT_DIR`** in that cell.

- Set **`OUTPUT_DIR`** to a real path on your machine (the notebook may still use a placeholder).
- Exported images are placed in **per-class subfolders** (names follow Open Images / FiftyOne labels, e.g. `Egg_(Food)`).

FiftyOne also caches zoo data in its own directory (see [FiftyOne dataset zoo](https://docs.voxel51.com/user_guide/dataset_zoo/index.html)); **`OUTPUT_DIR`** is your explicit export for file-based training or inspection.
