from setuptools import setup, find_packages

setup(
    name='simmes',
    version='0.1.1',
    description='GRB measurement simulation packages',
    packages=find_packages(include=["simmes","simmes.*"]),
    license="MIT",
    keywords="gamma-ray bursts",
    author='Michael Moss',
    author_email='mikejmoss3@gmail.com',
    url='https://github.com/',
)