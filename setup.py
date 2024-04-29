from setuptools import setup, find_packages

package_name = "maskedvae"
version = "1.0"
exclusions = []
_packages = find_packages(exclude=exclusions)

_base = [
    "numpy",
    "matplotlib",
    "scipy",
    "seaborn",
    "scikit-learn",
    "torch",
    "pyyaml",
    "wandb",
    "torchvision",
    "h5py",
    "ipykernel",
]

_extras = {
    "dev": [
        "autoflake",
        "black",
        "deepdiff",
        "flake8",
        "isort",
        "jupyter",
        "pep517",
        "pytest",
        "pyyaml",
    ]
}

setup(
    name=package_name,
    version=version,
    description="Modeling conditional distributions of neural and behavioral data with masked variational autoencoders",
    author="Auguste Schulz",
    author_email="auguste.schulz@uni-tuebingen.de",
    url="https://github.com/mackelab/neuro-behavior-conditioning.git",
    packages=_packages,
    install_requires=_base,
    extras_require=_extras,
    license="MIT",
)
