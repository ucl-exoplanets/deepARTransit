import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="deepartransit",
    version="1.0.1",
    author="Mario Morvan",
    author_email="mario.morvan.18@ucl.ac.uk",
    description="A library for interpolating transit light curves.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ucl-exoplanets/deepARTransit",
    packages=setuptools.find_packages(),
    classifiers=[],
    install_requires=requirements,
    python_requires='>=3.6',
)
