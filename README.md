# SPiFiL — Superpixel Filter Learning

**Convolutional encoders learned without backpropagation.** A superpixel
algorithm partitions each training image inside its mask; the center of every
superpixel becomes a candidate feature point; patches around those points are
ranked by how well they separate classes (Fisher score under a Mahalanobis
metric) and selected with a diversity-aware greedy pick; and the chosen patches
*become* the convolution kernels, with z-score normalization folded into the
weights and bias. Repeat per layer and the result is a plain PyTorch
`nn.Module` --- no optimizer, no epochs, no gradients anywhere in the fit.

This is the reference implementation for the paper *Representation Learning
from Superpixels: Lightweight CNN Encoders Under Data Scarcity*, which learns
encoders from **one image per class** and reports competitive accuracy against
CNN baselines two to three orders of magnitude larger.

```python
from spifil import ArchSpec, LayerSpec, Learner, SpifilDataset, load_encoder

data = SpifilDataset.from_folders("images/", "masks/")   # class = filename prefix
learn = Learner(data, ArchSpec(layers=[LayerSpec(kernel_size=3, out_channels=n)
                                       for n in (16, 32, 48)]))
learn.fit()
learn.export("runs/my-model")

encoder = load_encoder("runs/my-model").freeze()         # a torch nn.Module
```

## Install

```bash
pip install -e .            # or, from a clone with uv:  uv sync
```

Python 3.10+, and the only hard dependencies are torch, numpy, scikit-image and
hydra-core. This installs everything except DISF, and defaults to SLIC
superpixels so the whole pipeline runs out of the box on any platform.

### The DISF backend (optional, needed to reproduce the paper)

The published results use **DISF** superpixels and its geodesic centers, which
come from PyIFT — a compiled extension, shipped here as a prebuilt wheel in
[`vendor/`](vendor/):

```bash
sudo apt-get install -y liblapack3 libblas3    # PyIFT links these dynamically

uv sync --group disf                           # with uv
pip install vendor/pyift-0.1-cp312-cp312-linux_x86_64.whl    # with pip
```

The wheel is built for **CPython 3.12 on Linux x86-64**; anywhere else it
installs nothing and SLIC remains available. SLIC partitions differently, so
it does not reproduce the paper's numbers.

> Install PyIFT from the wheel above, not from PyPI.

## Quickstart

**[examples/quickstart.ipynb](examples/quickstart.ipynb)** walks the whole path
with pictures: the dataset, superpixels and seeds, the fit, the selected
patches that became filters, what each layer encodes, export and reload, and a
task head trained on the frozen encoder. It runs on a fresh clone, on the
bundled data below. The test suite executes it, so it cannot silently stop
working.

### The bundled datasets

[`examples/data/`](examples/data/README.md) ships the **training images the
published encoders were learned from** --- one image per class, for each of
three stratified splits, on all three parasitology datasets:

| Dataset  | Classes | Images per split | Splits |
|---|---|---|---|
| `larvae` | 2 | 2 | `split1`–`split3` |
| `cysts`  | 6 | 6 | `split1`–`split3` |
| `eggs`   | 8 | 8 | `split1`–`split3` |

```
examples/data/eggs/split1/
├── images/000001_00000333.png    class 1, from the integer filename prefix
└── masks/000001_00000333.png     where superpixels may be placed
```

They are a subset of the [Intestinal Parasites
Datasets](https://github.com/LIDS-UNICAMP/intestinal-parasites-datasets) from
LIDS-UNICAMP --- go there for the full data and the split definitions, which
is what the downstream classifiers are trained and evaluated on.
[`examples/data/README.md`](examples/data/README.md) has the class mapping and
the terms of use.

That layout is the whole data format, and your own data follows it: the class
is the integer prefix of the filename, and a mask pairs with an image by name.
Masks are the dataset's own semantic masks --- SPiFiL asks for **no extra
annotation**. They keep every candidate filter on the class of interest rather
than on background. Images without a mask still work; the whole image is then
employed to extract superpixels and guide filter learning.

### Reproducing the published encoders

```bash
spifil-fit data.images_dir=examples/data/eggs/split1/images \
           data.masks_dir=examples/data/eggs/split1/masks \
           superpixels=disf n_superpixels=50 arch=paper
```

`arch=paper` is the 16/32/48 stack and `superpixels=disf` needs the optional
backend above; repeat over `split1`–`split3` for the reported mean and
standard deviation. A quick way to confirm the configuration is right is the
filter count a fit reports: `eggs` and `larvae` build 16/32/48, while `cysts`
builds **12/30/48**, because 6 classes divide 16 and 32 with a remainder
(excluding impurities).

### Fitting, in full

```python
from spifil import (
    DISF, ArchSpec, DiversitySelector, FisherScorer, LayerSpec, Learner,
    Mahalanobis, Medoids, ProgressBar, SpifilDataset,
)

data = SpifilDataset.from_folders("images/", "masks/")
learn = Learner(
    data,
    ArchSpec(layers=[LayerSpec(kernel_size=3, out_channels=n)
                     for n in (16, 32, 48)]),
    superpixels=DISF(),              # SLIC() with no optional dependency
    seed_extractor=Medoids(),        # the paper's seed rule
    metric=Mahalanobis(),
    scorer=FisherScorer(),
    selector=DiversitySelector(alpha=0.5, pool_factor=3),
    cbs=[ProgressBar()],
    n_superpixels=50,
)
learn.fit()
learn.export("runs/my-model")
```

Then use it like any other torch encoder:

```python
from torch import nn
from spifil import load_encoder

encoder = load_encoder("runs/my-model").freeze()   # color transform + fitted stack
model = nn.Sequential(encoder, my_head)            # images in, predictions out

features = encoder(images)             # (B, H, W, 3) uint8 -> (B, C, H', W')
mid = encoder(images, upto=2)          # an earlier, higher-resolution layer
```

`load_encoder` returns the color transform *with* the network, and that
pairing matters: a `SpifilNet` alone consumes layer-0 features rather than
images, so handing a downstream model the network by itself is handing it half
an artifact --- and the failure is silent, since the shapes match and only the
numbers are wrong.

## From the command line

Experiments are Hydra config groups, so components are swapped and swept
without touching code:

```bash
spifil-fit data.images_dir=... data.masks_dir=...
spifil-fit scorer=distance_sum selector.alpha=0.7
spifil-fit -m selector.alpha=0.3,0.5,0.7                   # a sweep
```

Hydra's run directory *is* the output directory, and the resolved config lands
in `.hydra/config.yaml` beside the results.

| Group | Options | Default |
|---|---|---|
| `superpixels` | `slic`, `disf` | `slic` |
| `seed_extractor` | `medoid`, `centroid`, `geodesic` | `medoid` |
| `color` | `labnorm` | `labnorm` |
| `metric` | `mahalanobis`, `euclidean` | `mahalanobis` |
| `scorer` | `fisher`, `distance_sum` | `fisher` |
| `allocator` | `uniform` | `uniform` |
| `selector` | `diversity`, `topn` | `diversity` |
| `strategy` | `sequential` | `sequential` |
| `arch` | `flim2`, `flim3`, `flim4`, `paper` | `flim3` |

## Extending it

Every stage is a `Protocol`, injected into the `Learner`, with a one-line
default. Implement the call signature and pass your object in --- no
subclassing, no registry, no edits to the pipeline:

| Protocol | Replaces | Signature |
|---|---|---|
| `SuperpixelAlgorithm` | how images are partitioned | `(features, mask, n) -> labels` |
| `SeedExtractor` | which pixel represents a region | `(labels, features) -> coords` |
| `ColorTransform` | layer-0 features | `(image) -> (C, H, W)` |
| `DistanceMetric` | the geometry scoring uses | `.fit(feats)`, `.pairwise(a, b)` |
| `Scorer` | what "discriminative" means | `(patches, metric) -> scores` |
| `FilterAllocator` | per-class filter budget | `(layer, spec, patches, scores) -> dict` |
| `Selector` | which candidates become filters | `(patches, ranks, budget) -> mask` |
| `ConvBlockFactory` | the block a layer builds | `(spec, in_channels) -> module` |
| `FitStrategy` | the order layers are visited | `.run(learner)` |
| `Callback` | observing or steering the fit | event methods on `Learner` |

Scoring and selection are `O(N²)` in the number of seed patches and run on
whatever device the tensors live on, so `device="cuda"` works without any
change to the components.

## Choosing `n_superpixels` and the number of images

Every scorer inverts a `D x D` covariance, where `D = kernel_size² x in_channels`
--- 27 at layer 1 with 3x3 kernels over 3 bands, and hundreds deeper in the
stack. With `N` seed patches, `N <= D` makes the empirical covariance
rank-deficient. SPiFiL uses Ledoit-Wolf shrinkage, so the metric stays
*defined* for any `N`, and warns when this happens.

Defined is not informative. For a statistically meaningful metric aim for
`N > D`, and get there by **adding images rather than superpixels**: seeds
merge under pooling, so `N` saturates as `n_superpixels` grows, while each
added image contributes an independent grid of candidates.

## Development

```bash
./ci.sh                     # lint, types, tests, and a clean-install check
./ci.sh --quick             # same, minus the clean-install check
```

Individually, if you prefer:

```bash
uv run pytest                                      # the full suite
uv run ruff check src tests examples
uv run mypy src
```

The tests ship no images: `tests/conftest.py` draws small synthetic datasets
from a fixed seed, so `pytest` works on a fresh clone with nothing downloaded.

## Porting disclaimer

SPiFiL was first written in C, and this package is a port of that
implementation to Python and PyTorch, published so the method is usable and
extensible without the original toolchain.

Reproducing the C results exactly was a requirement to validate the port, and a
few design decisions were made to serve it. They are faithful, not necessarily
optimal, and they are documented where they live in the code:

- **The color transform** (`spifil.color.LabNorm`) reproduces the original
  normalized-Lab conversion including its quirks --- an asymmetric BT.2020 /
  BT.601 round trip, no sRGB gamma decode, fixed normalization constants, and
  float32 rounding at specific intermediate steps. It is not a textbook CIELAB
  conversion.
- **Filter budgets are split by plain integer division**
  (`spifil.allocation.UniformAllocator`), so a layer asked for 16 filters over
  6 classes produces 12 and the remainder is dropped rather than distributed.
- **The candidate-pool cap is shared across classes**
  (`spifil.selection.DiversitySelector`), so a class with few ranked seeds can
  limit the pool available to a well-stocked one.
- **The layer loop is strictly forward and single-pass**
  (`spifil.strategies.SequentialStrategy`), with no revisiting of earlier
  layers once later ones exist.

Future versions will lift these limits and optimize the framework with Python
in mind alone, rather than as a translation: a color transform chosen on its
merits, remainder-aware and difficulty-aware allocation, per-class pooling, and
alternative fit strategies. Each is already a swappable component, so the
change will be in the defaults rather than in the architecture.

## Citation

```bibtex
@inproceedings{spifil,
  title     = {Representation Learning from Superpixels:
               Lightweight CNN Encoders Under Data Scarcity},
  author    = {TODO},
  booktitle = {TODO},
  year      = {2026}
}
```

## License

Apache 2.0 --- see [LICENSE](LICENSE). Two things here are not covered by it:
the vendored PyIFT wheel, a third-party component under its own terms, and the
images under `examples/data/`, which belong to the [dataset
authors](https://github.com/LIDS-UNICAMP/intestinal-parasites-datasets).