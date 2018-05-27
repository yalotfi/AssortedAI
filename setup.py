from setuptools import setup, find_packages
from codecs import open
from os import path


here = path.abspath(path.dirname(__file__))


with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()


setup(
    # High level package information
    name='AssortedAI',
    version='0.1.0',
    description='An assortment of ML algorithms and tools',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    license="MIT License",

    # Maintainer information
    url='https://github.com/yalotfi/AssortedAI',
    author='Yaseen Lotfi',
    author_email='yalotfi@outlook.com',

    # Make the package searchable
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords='machine learning artificial intelligence',

    # Dependency Information
    packages=find_packages(),
    install_requires=['mkl', 'numpy', 'matplotlib']
)
