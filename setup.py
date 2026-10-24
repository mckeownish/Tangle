from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy


extensions = [

    Extension(
        "wiggle.writhe", 
        ["cython_src/writhe_c.pyx"],
        extra_compile_args=['-O3',
                            '-march=native',
                            '-funroll-loops', 
                            '-fopenmp',
                            '-ffast-math',],

        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()])
    ]

    

# Use cythonize on the extensions
setup(
    name='wiggle',
    version='0.8',
    packages=find_packages(),
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
        }
    ),
    install_requires=[
        'numpy',
        'scipy',
        'plotly',
        'pandas',
        'biobox',
        'biopython',
        'tqdm',
        'selenium'
    ],
    author='Josh McKeown',
    author_email='josh.j.mckeown@durham.ac.uk',
    description='Package used in my No Loose Ends Lab, where I explore a subfield of mathematics known as knot theory, specifically a metric of self entanglement referred to as the writhe. Im looking into its potential as a geometrical representation to help solve real world problems such as proteins folding!'
)
