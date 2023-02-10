import os
import numpy
from setuptools import find_packages, setup, Extension

# Numpy include
NUMPY_INCLUDE = numpy.get_include()

# Detect Cython
try:
    import Cython

    ver = Cython.__version__
    _CYTHON_INSTALLED = ver >= "0.28"
except ImportError:
    _CYTHON_INSTALLED = False

if not _CYTHON_INSTALLED:
    print("Required Cython version >= 0.28 is not detected!")
    print('Please run "pip install --upgrade cython" first.')
    exit(-1)


def requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]


def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()

# Cython Extensions
extensions = [
    Extension(
        "aiq.ops.libs.rolling",
        ["aiq/ops/libs/rolling.pyx"],
        language="c++",
        include_dirs=[NUMPY_INCLUDE],
    ),
    Extension(
        "aiq.ops.libs.expanding",
        ["aiq/ops/libs/expanding.pyx"],
        language="c++",
        include_dirs=[NUMPY_INCLUDE],
    ),
]


def version():
    version_file = 'aiq/version.py'
    with open(version_file, encoding='utf-8') as f:
        exec (compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


setup(
    name="aiq",
    version=version(),
    packages=find_packages(exclude=(
            'tests',
            'docs',
            'examples',
            'requirements',
            '*.egg-info',
    )),
    author="darrenwang",
    author_email="wangyang9113@gmail.com",
    description="aiq",
    long_description=readme(),
    long_description_content_type="text/markdown",
    install_requires=requirements('requirements/requirements.txt'),
    ext_modules=extensions,
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6"
)
