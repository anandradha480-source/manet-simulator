from setuptools import setup, find_packages

setup(
    name="rl-arhm",
    version="1.0.0",
    description="MANET Simulator with DDPG Next-Hop Routing, AODV, and GPSR Baselines",
    author="Antigravity Research Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
    ],
    python_requires=">=3.8",
)
