from setuptools import setup, find_packages

setup(
    name='simmes',
    version='0.1.3',
    description='GRB measurement simulation packages',
    packages=find_packages(include=["simmes","simmes.*"]),
    package_data={"":["util_packages/files-det-ang-dependence/*","util_packages/files-swiftBAT-resp-mats/*"]},
    license="MIT",
    keywords="gamma-ray bursts",
    author='Michael Moss',
    author_email='mikejmoss3@gmail.com',
    url='https://github.com/mikemoss3/simmes',
)