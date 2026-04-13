# Project C - Image Classification (Chicken, Egg, Balloon)

## Objective
This repository implements Project C from `resources/Project C.docx.pdf`: classify Open Images samples into 3 classes (`Chicken`, `Egg (Food)`, `Balloon`) with ResNet50-based experiments and compare the results.

## Dataset and Split
- Source dataset: Open Images (subset for the 3 target classes).
- Effective class set used in training outputs: `Balloon, Chicken, Egg (Food)`.
- Split strategy: 80/20 train/test.
- Final split size used by all reported experiments: 1200 train and 300 test images (`results/*_summary.csv`).
- Per-class counts in confusion matrices indicate 100 test images per class (balanced test set).

## Experiments Performed
1. ResNet50 from scratch (random initialization).
   - Artifacts: `results/resnet50_scratch_summary.csv`, `results/resnet50_scratch_history.csv`, `results/resnet50_scratch_curves.png`, `results/resnet50_scratch_confusion_matrix.png`
2. Transfer learning with ImageNet-pretrained ResNet50.
   - Artifacts: `results/resnet50_transfer_summary.csv`, `results/resnet50_transfer_history.csv`, `results/resnet50_transfer_curves.png`, `results/resnet50_transfer_confusion_matrix.png`
3. Transfer learning + data augmentation (flip, rotation, translation).
   - Artifacts: `results/resnet50_transfer_aug_summary.csv`, `results/resnet50_transfer_aug_history.csv`, `results/resnet50_transfer_aug_curves.png`, `results/transfer_vs_transfer_aug.png`
4. Architecture modification after `conv3_block4_out` with frozen early layers.
   - Artifacts: `results/resnet50_experiment_architecture_summary.csv`, `results/resnet50_experiment_architecture_history.csv`, `results/resnet50_experiment_architecture_curves.png`, `results/resnet50_experiment_architecture_confusion_matrix.png`
5. First-10-epoch scratch vs transfer comparison.
   - Artifact: `results/scratch_vs_transfer_first10.png`
6. Own internet images + Grad-CAM maps.
   - Artifacts in `results/gradcam/` (8 images): `baloon_gradcam.png`, `balloon_egg_gradcam.png`, `balloon_girl_gradcam.png`, `chicken_gradcam.png`, `chicken-farm_gradcam.png`, `chicken_egg_man_gradcam.png`, `egg_flur_gradcam.png`, `eggs_gradcam.png`

## Assignment Checklist (Task Status)
| Requirement from Project C | Status | Evidence | Notes |
|---|---|---|---|
| Explore dataset classes, distribution, imbalance, and observations affecting training | PARTIALLY DONE | `src/image_classification.ipynb`, `results/*_summary.csv` | Classes and balanced split are evidenced; detailed written EDA observations are limited in current report artifacts. |
| Prepare 80/20 train/test split | DONE | `results/resnet50_scratch_summary.csv` | All summaries report 1200 train / 300 test samples (consistent with 80/20). |
| Train ResNet50 from scratch and estimate test accuracy | DONE | `results/resnet50_scratch_summary.csv` | Final test accuracy: 0.710. |
| Transfer learning with ImageNet ResNet50 and estimate test accuracy | DONE | `results/resnet50_transfer_summary.csv` | Final test accuracy: 0.957. |
| Show first 10 epochs loss/accuracy difference: scratch vs pre-trained | DONE | `results/scratch_vs_transfer_first10.png`, `results/resnet50_scratch_history.csv`, `results/resnet50_transfer_history.csv` | Plot is available and history CSVs support it. |
| Data augmentation experiment (random flip, random rotate, random translation) | DONE | `src/train_resnet50_transfer_learning_data_augmentation.ipynb`, `results/resnet50_transfer_aug_summary.csv` | Implemented with horizontal flip, rotation, affine translation. |
| Architecture experiment: modify after conv3_block4_out and freeze conv2 and before | DONE | `results/resnet50_experiment_architecture_summary.csv`, `results/resnet50_experiment_architecture_curves.png` | Modified pretrained model was trained and evaluated. |
| Test own internet images and show activation maps | DONE | `results/gradcam/*.png` | 8 Grad-CAM outputs are present. |
| Report train/test accuracy | DONE | `results/resnet50_scratch_summary.csv`, `results/resnet50_transfer_summary.csv`, `results/resnet50_transfer_aug_summary.csv`, `results/resnet50_experiment_architecture_history.csv` | Final train/test values reported below (modified model train accuracy taken from history). |
| Report infrastructure and inference time | PARTIALLY DONE | `results/resnet50_scratch_summary.csv`, `results/resnet50_experiment_architecture_summary.csv` | Inference time present for scratch and modified; not present in transfer and transfer+aug summary CSVs. |
| Report number of parameters | DONE | `results/*_summary.csv` | Parameter counts available for all major experiments. |
| Show confusion matrix and identify common confusions | PARTIALLY DONE | `results/resnet50_scratch_confusion_matrix.png`, `results/resnet50_transfer_confusion_matrix.png`, `results/resnet50_experiment_architecture_confusion_matrix.png` | Confusion matrices exist for 3 models; transfer+aug confusion matrix artifact is not present. |
| Compare experiment results | DONE | `results/*_summary.csv`, `results/scratch_vs_transfer_first10.png`, `results/transfer_vs_transfer_aug.png` | Direct numeric and curve-based comparison is available. |

## Quantitative Results
All values are taken from the summary/history CSV files in `results/`.

| Experiment | Final Train Acc | Final Test Acc | Final Test Loss | Parameters | Trainable Parameters | Total Train Time (s) | Avg Epoch Time (s) | Avg Inference Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scratch ResNet50 | 0.889 | 0.710 | 1.002 | 23,514,179 | 23,514,179 | 299.010 | 14.950 | 5.704 |
| Transfer (ImageNet) | 1.000 | 0.957 | 0.085 | 23,514,179 | n/a | 307.760 | 15.388 | n/a |
| Transfer + Augmentation | 0.998 | 0.970 | 0.158 | 23,514,179 | n/a | 320.963 | 16.048 | n/a |
| Modified Pretrained Architecture | 0.997* | 0.930 | 0.196 | 13,776,451 | 13,551,107 | 278.993 | 13.950 | 5.138 |

\* Final train accuracy for modified architecture is from `results/resnet50_experiment_architecture_history.csv` (epoch 20), because the summary CSV does not include a train-accuracy column.

## Answers to Project Questions
- **Q: What accuracy can be achieved? Train vs test?**  
  **A:** Best test accuracy in current artifacts is **0.970** (Transfer + Augmentation). Final train/test by experiment: Scratch **0.889 / 0.710**, Transfer **1.000 / 0.957**, Transfer+Aug **0.998 / 0.970**, Modified **0.997 / 0.930**.

- **Q: On what infrastructure did you train it? What is inference time?**  
  **A:** The notebook workflow is configured for GPU use (Colab instructions are included in `src/train_resnet50_transfer_learning_data_augmentation.ipynb`; runs also support local execution). Measured average inference time is available for Scratch (**5.704 ms**) and Modified (**5.138 ms**) in their summary CSVs. Transfer and Transfer+Aug inference time is currently not reported in their summary files.

- **Q: What are the number of parameters of the model?**  
  **A:** Scratch/Transfer/Transfer+Aug all use **23,514,179** parameters. The modified architecture has **13,776,451** parameters, with **13,551,107** trainable.

- **Q: Which categories are most likely to be confused? (confusion matrix)**  
  **A:**  
  - Scratch (`results/resnet50_scratch_confusion_matrix.png`): strongest confusion is **Egg (Food) -> Chicken (36 cases)** and **Balloon -> Egg (21 cases)**.  
  - Transfer (`results/resnet50_transfer_confusion_matrix.png`): small remaining confusions mainly toward **Balloon** (Chicken->Balloon = 3, Egg->Balloon = 3).  
  - Modified (`results/resnet50_experiment_architecture_confusion_matrix.png`): main confusion is **Balloon -> Egg (7 cases)**.

- **Q: Compare the results of the experiments.**  
  **A:** Transfer learning gives a large improvement over scratch (test acc **0.957 vs 0.710**, test loss **0.085 vs 1.002**). Adding augmentation further improves final test accuracy to **0.970**, but with higher final test loss (**0.158**) and longer training time. The modified architecture is faster and smaller, but lower in accuracy (**0.930 test acc**) than standard transfer and transfer+aug.

## Key Comparisons and Discussion
- **Scratch vs Transfer (first 10 epochs):** `results/scratch_vs_transfer_first10.png` shows faster convergence and better test metrics for the pretrained model early in training.
- **Transfer vs Transfer+Aug:** `results/transfer_vs_transfer_aug.png` and summary CSVs show +1.33 percentage points test accuracy for augmentation (0.970 vs 0.957), with extra runtime.
- **Generalization gap (final epoch):**
  - Scratch: 0.889 - 0.710 = **0.179**
  - Transfer: 1.000 - 0.957 = **0.043**
  - Transfer+Aug: 0.998 - 0.970 = **0.028**
  - Modified: 0.997 - 0.930 = **0.067**
- Interpretation: augmentation improves final test accuracy and slightly reduces train-test gap versus plain transfer, while the modified architecture trades some accuracy for speed/size.

## Remaining Work / Limitations
- Explicit written EDA narrative (image quality issues, imbalance discussion, representative examples) should be expanded in a single report section for clearer lecturer review.
- Inference-time measurement is missing in `results/resnet50_transfer_summary.csv` and `results/resnet50_transfer_aug_summary.csv`.
- No dedicated confusion matrix artifact was found for the Transfer+Aug experiment.
- Infrastructure details (exact GPU/CPU/RAM and runtime logs per run) are not centrally documented in `results/`.

## Reproducibility
- Setup environment:
  - `python -m venv .venv`
  - `.\\.venv\\Scripts\\Activate.ps1`
  - `python -m pip install --upgrade pip`
  - `pip install -r requirements.txt`
- Main notebook workflow (includes transfer and transfer+augmentation):
  - `src/train_resnet50_transfer_learning_data_augmentation.ipynb`
- Script-based baseline:
  - `python src/train_resnet50_scratch.py`
- Re-generated artifacts are written to `results/` (CSV summaries/history and PNG plots).
