def create_classification_dataset_from_openimages(
    class_names,
    output_dir,
    max_images_per_class=200,
    split="train",
    max_samples=800,
    dataset_name="open-images-clean",
    seed=42,
):
    from pathlib import Path
    import fiftyone as fo
    import fiftyone.zoo as foz
    from fiftyone import ViewField as F

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # remove old dataset
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)

    # ----------------
    # 1. LOAD DATASET
    # ----------------
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        label_types=["detections"],
        classes=class_names,
        max_samples=max_samples,
        shuffle=True,
        dataset_name=dataset_name,
    )

    print("Loaded samples:", len(dataset))

    # ----------------
    # 2. FILTER LABELS
    # ----------------
    view = dataset.filter_labels(
        "ground_truth",
        F("label").is_in(class_names)
    )

    view = view.match(
        F("ground_truth.detections.label").contains(class_names)
    )

    print("After filtering:", len(view))

    # ----------------
    # 3. PATCHES
    # ----------------
    patches = view.to_patches("ground_truth")
    print("Total patches:", len(patches))

    # ----------------
    # 4. BALANCE
    # ----------------
    balanced_views = []

    for class_name in class_names:
        class_view = patches.match(F("ground_truth.label") == class_name)
        class_view = class_view.take(max_images_per_class, seed=seed)

        if len(class_view) > 0:
            balanced_views.append(class_view)

    if not balanced_views:
        raise ValueError("No samples found for given classes!")

    final_view = balanced_views[0]
    for v in balanced_views[1:]:
        final_view = final_view.concat(v)

    print("Balanced dataset size:", len(final_view))

    # ----------------
    # 5. EXPORT
    # ----------------
    final_view.export(
        export_dir=str(output_dir),
        dataset_type=fo.types.ImageClassificationDirectoryTree,
        label_field="ground_truth",
    )

    # ----------------
    # 6. SUMMARY
    # ----------------
    print("\nFinal dataset per class:")
    for class_name in class_names:
        count = final_view.match(
            F("ground_truth.label") == class_name
        ).count()
        print(f"{class_name}: {count}")

    print("\nSaved to:", output_dir.resolve())

    return final_view

#dataset = create_classification_dataset_from_openimages(
#    class_names=["Egg (Food)", "Chicken", "Balloon"],
#    output_dir="src/data/openimages_subset/classification",
#    max_images_per_class=200,
#)