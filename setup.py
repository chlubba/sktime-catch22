# -*- coding: utf-8 -*-

from setuptools import setup

SHORTDESC = "catch22Forest - a forest classifier for time series based on catch22 for sktime"

DESC = """
It provides a random forest classification based on catch22 features
The package is provided under the GPLv3 license.
"""

setup(
    name="catch22Forest",
    version="0.0.0",
    author="Carl H Lubba",
    author_email="c.lubba15@imperial.ac.uk",
    url="https://github.com/chlubba/catch22",
    description=SHORTDESC,
    long_description=DESC,
    license="GPLv3",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Programming Language :: Cython",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
    ],
    setup_requires=[
        'numpy>=1.14.2',
        'setuptools>=18.0',
    ],
    install_requires=["scikit-learn", "sktime"],
    python_requires=">=3.4.0",
    provides=["catch22Forest"],
    keywords=["machine learning", "time series distance"],
    packages=["catch22Forest"],
    # package_data={
    #     'catch22Forest': ['*.pxd', '*.pyx', '*.c'],
    # },
    zip_safe=False
)