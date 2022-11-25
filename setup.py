from setuptools import find_packages, setup

import versioneer

setup(
    name='wsp-gtam-tools',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='A Python package for handling GTAModel data',
    url='https://github.com/sap-toronto/gtam-tools',
    author='WSP',
    maintainer='Brian Cheung',
    maintainer_email='brian.cheung@wsp.com',
    classifiers=[
        'License :: OSI Approved :: MIT License'
    ],
    packages=find_packages(),
    package_data={'': ['resource_data/*.csv']},
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=[
        'bokeh<3.0.0',
        'geopandas>=0.9.0',
        'numpy',
        'pandas',
        'pyproj',
        'wsp-balsa>=1.2.2',
        'wsp-cheval'
    ]
)
