from distutils.core import setup
import unittest

def my_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('test', pattern='test_*.py')
    return test_suite

setup(
    name = 'onsager',
    packages = ['onsager'], 
    version = '1.1',
    description = 'A package to compute Onsager coefficients for vacancy-mediated diffusion and interstitial elastodiffusion tensors',
    author = 'Dallas R. Trinkle',
    author_email = 'dtrinkle@illinois.edu',
    license='LICENSE.txt',
    url = 'https://github.com/DallasTrinkle/onsager', # use the URL to the github repo
    download_url = 'https://github.com/DallasTrinkle/onsager/tarball/v1.1', # for when we upload
    keywords = ['diffusion', 'elastodiffusion', 'Onsager', 'mass transport'],
    classifiers = [], requires=['numpy', 'scipy', 'yaml', 'h5py'],
    test_suite = 'setup.my_test_suite'
)
