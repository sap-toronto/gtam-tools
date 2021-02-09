from os import path
from pkg_resources import safe_version
from setuptools import setup, find_packages

version = {}
with open(path.join(path.dirname(path.realpath(__file__)), 'gtam_tools', 'version.py')) as fp:
    exec(fp.read(), {}, version)

setup(
    name='wsp-gtam-tools',
    version=safe_version(version['__version__']),
    description='A Python package for handling GTAModel data',
    url='https://github.com/sap-toronto/gtam-tools',
    author='WSP',
    maintatiner='Brian Cheung',
    maintainer_email='brian.cheung@wsp.com',
    classifiers=[
        'License :: OSI Approved :: MIT License'
    ],
    packages=find_packages(),
    package_data={'': ['resources/*.csv']},
    python_requires='>=3.6',
    install_requires=[
        'geopandas',
        'numpy',
        'pandas==0.23.4',
        'plotly',
        'wsp-balsa',
        'wsp-cheval'
    ]
)
