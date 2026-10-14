# Installs the verl fork (Apache-2.0, see LICENSE and Notice.txt) and the mrtpo package.
# Dependencies are pinned in environments/requirements.lock.txt and installed by environments/install.sh.
from pathlib import Path

from setuptools import find_packages, setup

version = (Path(__file__).parent / "verl" / "version" / "version").read_text().strip()

setup(
    name="verl-mrtpo",
    version=version,
    packages=find_packages(include=["verl", "verl.*", "mrtpo", "mrtpo.*"]),
    package_data={
        "verl": ["version/*", "trainer/config/*.yaml", "trainer/config/*/*.yaml"],
        "mrtpo": ["sql/*.json"],
    },
    include_package_data=True,
    license="Apache-2.0",
    python_requires=">=3.10",
)
