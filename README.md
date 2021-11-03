# koi

A bunch of work-in-progress Pytorch scripts on VAE, used to experiment with anomaly detection and other stuff!

## Execution

Run example files in the `examples/` folder. Run Tensorboard to monitor training, using `tensorboard --logdir=<path>`.
Remember to set up in `BaseConfig.py` the same logdir.

## Development

Use `coverage report -m` to view the current accumulated recorded coverage for tests. Rememeber to `cd` inside
the `koi/` directory.

You can also run `pytest --cov=koi --cov-report=html` in the root directory to view the html output.

## Current progress

* Models:
* [x] VAE
* [x] V-ABC
* [x] VAE with output variance
* [ ] IWAE?
* Metrics (todo):
* [ ] Reconstruction error
* [ ] Density estimation
* [ ] ELBO
* [ ] ...
* Datasets:
* [x] Toy dataset "moons"
* [ ] Toy dataset circles
* [ ] MNIST
* [ ] Songs (or something else...)
* AI-Dev
* [x] Tensorboard
* [x] 2d plots for toy datasets
* [ ] "Gradient field"
* [ ] Manifolds for MNIST
* [ ] Latent space comparison
* SW-Dev
* [x] Pytest
* [x] Coverage
* [ ] CI tools (Travis or similar)
