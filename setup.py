from pathlib import Path
from setuptools import find_packages, setup


def read_requirements(path: str) -> list[str]:
    requirements_file = Path(path)
    if not requirements_file.exists():
        return []
    return [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


setup(
    name="appold-citak-2026",
    description="Simulation and analysis code for Appold/Citak 2026.",
    packages=find_packages(include=["source", "source.*"]),
    install_requires=read_requirements("requirements.txt"),
    python_requires=">=3.11,<3.12",
)
