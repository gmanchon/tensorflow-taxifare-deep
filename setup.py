
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [c.strip() for c in f.readlines()]

setup(name="tensorflow taxifare deep",
      version="0.1",
      description="tensorflow taxifare deep",
      packages=find_packages(),
      include_package_data=True,  # use MANIFEST.in
      install_requires=requirements)
