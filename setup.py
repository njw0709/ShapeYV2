import setuptools

# reading long description from file
with open("README.md") as file:
    long_description = file.read()


# specify requirements of your package here
REQUIREMENTS = [
    "tqdm >= 4.48.0",
    "cupy-cuda102",
    "matplotlib>=3.3.1",
    "h5py>=2.10.0",
    "twine",
    "black",
    "pylint",
    "pytest",
    "pytest-profiling",
    "mypy",
    "numpy>=1.21",
    "bidict",
    "dacite",
    "Pillow>=9.5.0",
    "gdown",
]

# some more details
CLASSIFIERS = [
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.9",
    "Development Status :: 4 - Beta",
]

# calling the setup function
# TODO: separate dev / test / deploy setup with options
setuptools.setup(
    name="ShapeYModular",
    version="2.0.3",
    description="Benchmark that tests shape recognition",
    long_description=long_description,
    url="https://github.com/njw0709/ShapeYV2",
    author="Jong Woo Nam",
    author_email="namj@usc.edu",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=CLASSIFIERS,
    install_requires=REQUIREMENTS,
    keywords="tests shape recognition capacity",
    package_data={"shapeymodular": ["utils/*.json", "utils/*.txt"]},
)
