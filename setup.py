from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
import unittest

def my_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('test', pattern='test_*.py')
    return test_suite

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = 'onsager',
    packages = ['onsager'],
    version = '1.3.3',
    description = 'A package to compute Onsager coefficients for vacancy-mediated diffusion and interstitial elastodiffusion tensors',
    long_description=long_description,
    author = 'Dallas R. Trinkle',
    author_email = 'dtrinkle@illinois.edu',
    license='MIT',  # LICENSE.txt
    url = 'https://github.com/DallasTrinkle/onsager', # use the URL to the github repo
    download_url = 'https://github.com/DallasTrinkle/onsager/tarball/v1.3.3', # for when we upload
    keywords = ['diffusion', 'elastodiffusion', 'mass transport',
                'Onsager coefficients', 'transport coefficieints'],
    package_data = {
        'onsager': ['onsager/*.pl']
        },
    include_package_data = True,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=['numpy', 'scipy', 'pyyaml', 'h5py'],
    test_suite = 'setup.my_test_suite'
)
