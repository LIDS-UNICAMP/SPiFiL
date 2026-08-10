# `examples/data` — the training images behind the published results

These are the **training sets the paper's encoders were learned from**: one
image per class, drawn from each of three stratified 50/50 train/test splits,
for all three parasitology datasets. They are what you need to reproduce the
SPiFiL encoders.

They are a small subset of the **Intestinal Parasites Datasets** published by
LIDS-UNICAMP:

> https://github.com/LIDS-UNICAMP/intestinal-parasites-datasets

Go there for the full data — every image, its mask, and the split definitions
— which is what you need to train and evaluate the downstream classifiers.
The filename convention below (`{class:06d}_{image:08d}.png`) is theirs, so
files here can be traced straight back to it.

| Dataset  | Classes | Images per split | Splits | Size |
|---|---|---|---|---|
| `larvae` | 2 | 2 | `split1`–`split3` | 200x200 RGB |
| `cysts`  | 6 | 6 | `split1`–`split3` | 200x200 RGB |
| `eggs`   | 8 | 8 | `split1`–`split3` | 200x200 RGB |

## Layout

```
examples/data/<dataset>/<split>/
├── images/000001_00000333.png     the image
└── masks/000001_00000333.png      its semantic mask
```

Two rules, and that is the whole format:

1. **The class is the integer prefix of the filename.** `000002_00001142.png`
   is class 2. Anything after the first non-digit is ignored, so `2_sample.png`
   works too, while `sample_2.png` is read as class 0.
2. **A mask pairs with an image by name.** The lookup tries the image's full
   filename first and then `<stem>.png`, so a `.jpg` image can pair with a
   `.png` mask. Any nonzero pixel counts as inside the region; superpixels are
   placed only there, which keeps every candidate filter on the class of
   interest instead of on background.

The masks here are the datasets' own semantic masks — no annotation was added
for SPiFiL. Masks are optional in general: without one, the whole image is
segmented, background included.

## Reproducing the published encoders

```bash
spifil-fit data.images_dir=examples/data/eggs/split1/images \
           data.masks_dir=examples/data/eggs/split1/masks \
           superpixels=disf n_superpixels=50 arch=paper
```

`arch=paper` is the 16/32/48 stack; `superpixels=disf` needs the optional DISF
backend (see the top-level README) and is what the results were produced with.
Repeat over `split1`–`split3` for the mean and standard deviation the paper
reports.

The filter counts a fit produces are worth checking against, because they are
the clearest signal that the pipeline is configured as published:

| Dataset | Requested | Built | Why |
|---|---|---|---|
| `eggs`   | 16/32/48 | 16/32/48 | 8 classes divide all three |
| `larvae` | 16/32/48 | 16/32/48 | 2 classes divide all three |
| `cysts`  | 16/32/48 | **12/30/48** | 6 classes leave a remainder in 16 and 32 |

On `larvae` and `cysts` the fit warns that the covariance is rank-deficient
(`N <= D`). That is expected at one image per class: with 50 superpixels per
image there are fewer seed patches than patch features at the deeper layers.
Shrinkage keeps the metric defined; see "Choosing `n_superpixels` and the
number of images" in the top-level README.

## Using your own data

Point the tools at any folder pair with the layout above:

```python
data = SpifilDataset.from_folders("my_images/", "my_masks/")
```

## Provenance and terms of use

| Here | Upstream dataset | Classes upstream | Classes here |
|---|---|---|---|
| `eggs`   | `helminth-eggs` (5,112 images)    | 9 (8 species + impurities) | 8 |
| `cysts`  | `protozoan-cysts` (9,568 images)  | 7 (6 species + impurities) | 6 |
| `larvae` | `helminth-larvae` (3,514 images)  | 2 (1 species + impurities) | 2 |

Filters are learned from parasite classes only, so the impurity class is left
out for `eggs` and `cysts`; `larvae` keeps both of its classes.

**These images are not covered by this repository's Apache-2.0 licence**,
which applies to the code. They belong to the dataset authors, and the
upstream repository states no licence of its own. Treat them as available for
reproducing and verifying the published results; for anything beyond that, and
before redistributing them further, check with the dataset authors at the
repository linked above.
