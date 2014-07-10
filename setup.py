from distutils.core import setup
setup(
    name = 'onsager',
    packages = ['onsager', 'test'], # including the test?
    version = '0.1',
    description = 'A package to compute Onsager coefficients for vacancy-mediated diffusion',
    author = 'Dallas R. Trinkle',
    author_email = 'dtrinkle@illinois.edu',
    license='LICENSE.txt',
    url = 'https://github.com/DallasTrinkle/onsager', # use the URL to the github repo
    download_url = 'https://github.com/DallasTrinkle/onsager/tarball/0.1', # for when we upload
    keywords = ['diffusion', 'Onsager', 'mass transport'],
    classifiers = [],
)
