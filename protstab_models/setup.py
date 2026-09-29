from setuptools import find_packages, setup

setup(
    name="protstab_models",
    version="0.1.0",
    description="Standalone neural network models for protein stability "
                "prediction (SPURS-compatible).",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10",
        "numpy",
        "fair-esm",
        "omegaconf",
    ],
)
