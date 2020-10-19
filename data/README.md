# Datasets

## Provided files

We provide the following zip archives, to be un-zipped within the ``data/`` directory:
* [SurfaceSamples.zip](https://drive.google.com/file/d/1DmWIK5k0Ehi5jDBrI_n9JjcBbkjOtyi9/view?usp=sharing) : Point clouds samplings of [ShapeNet](https://www.shapenet.org/) objects from the 13 categories used in the paper. These are obtained by running the sampling strategy form the authors of [DeepSDF](https://github.com/facebookresearch/DeepSDF) on the isosurfaces provided by the authors of [DISN](https://github.com/Xharlie/DISN).
* [renderings_rgb.zip](https://drive.google.com/file/d/1ZHkVBEtdOOshAG83AGLiMWXTRjIgGise/view?usp=sharing) : RGB renderings of the above ShapeNet shapes, using the pipeline from DISN's authors. As explained in the main paper, we had to re-run the rendering sript because the provided depth maps are clipped (see [this issue](https://github.com/Xharlie/ShapenetRender_more_variation/issues/4)). As a results, the viewpoints do not correspond to the ones released by the authors of DISN.
* [renderings_depth.zip](https://drive.google.com/file/d/1AUJJj2ubYJliHdN-AlsIy3YjIvwjbp2e/view?usp=sharing) : full un-clipped depth maps renderings of the above ShapeNet shapes, using the pipeline from DISN's authors.
* [inferred_cameras.zip](https://drive.google.com/file/d/1MIOeRFUxF4sbQcOYilmxKIIi7_B03Jr8/view?usp=sharing) : predicted cameras for the above viewpoints, as inferred by the auxiliary pose estimator.
* [inferred_depth.zip](https://drive.google.com/file/d/1tAZXD0eidsm8pTbG9IM4zpHhxv1fjCu0/view?usp=sharing) : predicted depthmaps for the above viewpoints, as inferred by the auxiliary depth estimator.

In addition, please place [normalization_parameters.pck](https://drive.google.com/file/d/1qBs6uNHyzKysfWrDR_cKYV8I_x5qIP1o/view?usp=sharing) directly at the root of ``data/``. This pickled file contains translation and scaling parameters applied to original ShapeNet meshes to match the point clouds given in SurfaceSamples.zip. It is required by the dataloader.

Upon downloading and extracting the above files, the ``data/`` directory structure should look like:
```dirstruct
|-- SurfaceSamples
|   |-- ShapeNet
|   |   |-- 02691156
|   |   ...
|-- inferred_cameras
|   |-- 02691156
|   ...
|-- inferred_depth
|   |-- 02691156
|   ...
|-- normalization_parameters.pck
|-- renderings_depth
|   |-- 02691156
|   ...
|-- renderings_rgb
|   |-- ShapeNet
|   |   |-- 02691156
|   |   ...
|-- splits
|   |-- all_13_classes_test.json
|   |-- all_13_classes_train.json
|   |-- cars_test.json
|   `-- cars_train.json
```

## Choose data source for training/testing
Depth maps and cameras inferred by the auxiliary networks are pre-generated. To choose whether to use ground truth or inferred ones for training or testing an UCLID-Net model, please see the options at the top of the [dataloader](../dataloaders/UCLID_Net.py).

## Acknowledgements
We warmly thank the authors of [ShapeNet](https://www.shapenet.org/), [DISN](https://github.com/Xharlie/DISN) and [DeepSDF](https://github.com/facebookresearch/DeepSDF) for their datasets and pre-processing pipelines. Please consider citing them!
