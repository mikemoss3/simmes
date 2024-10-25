from setuptools import setup, find_packages

try:
    with open("README.md", "r") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ""

setup(
    name='simmes',
    version='0.1.12',
    description='GRB measurement simulation packages',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["simmes","simmes.*"]),
    package_data={"":["util_packages/files-det-ang-dependence/*","util_packages/files-swiftBAT-resp-mats/*"]},
    license="MIT",
    keywords="gamma-ray bursts",
    author='Michael Moss',
    author_email='mikejmoss3@gmail.com',
    url='https://github.com/mikemoss3/simmes',
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires=">=3.8",
)