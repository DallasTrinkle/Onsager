from distutils.core import setup
setup(
    name = 'onsager',
    packages = ['onsager'], 
    version = '0.9.1',
    description = 'A package to compute Onsager coefficients for vacancy-mediated diffusion and interstitial elastodiffusion tensors',
    author = 'Dallas R. Trinkle',
    author_email = 'dtrinkle@illinois.edu',
    license='LICENSE.txt',
    url = 'https://github.com/DallasTrinkle/onsager', # use the URL to the github repo
    download_url = 'https://github.com/DallasTrinkle/onsager/tarball/0.9.1', # for when we upload
    keywords = ['diffusion', 'elastodiffusion', 'Onsager', 'mass transport'],
    classifiers = [], requires=['numpy', 'scipy', 'yaml', 'h5py']
)
