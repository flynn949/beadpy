from setuptools import setup
  
setup(name='beadpy',
    version='0.2',
    description='flow-stretching data pipeline',
    url = "https://github.com/flynn949/beadpy",
    author='Flynn Hill',
    author_email='flynn@uow.edu.au',
    license='MIT',
    packages=['beadpy'],
    install_requires = ['numpy>=1.7', 
                    'scipy>=0.12',
                    'pandas>=0.13', 
                    'matplotlib',
                    'numba'],
    zip_safe=False)