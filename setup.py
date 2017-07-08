from setuptools import find_packages, setup

def get_readme():
    readme = ''
    try:
        import pypandoc
        readme = pypandoc.convert('README.md', 'rst')
    except (ImportError, IOError):
        with open('README.md', 'r') as file_data:
            readme = file_data.read()
    return readme


setup(
    name='pynoci',
    version='0.0.1',
    author='Daniel Mercader',
    author_email='mercaderd@yahoo.es',
    description=('Three methods for optimal number of independent components determination'),
    long_description=get_readme(),
    license='MIT',
    keywords='ica noci jade',
    url='',
    packages=find_packages(),
    package_data={
        'pynoci': ['*.mat']
    },
    install_requires=['numpy', 'matplotlib', 'scipy'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
