from setuptools import find_packages, setup


def requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]


def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()


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
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6"
)
