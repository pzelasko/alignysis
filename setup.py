from setuptools import find_packages, setup

setup(
    name='alignysis',
    version='0.1',
    python_requires='>=3.6.0',
    description='Phone alignment analysis package.',
    author='Piotr Żelasko',
    license='Apache-2.0 License',
    packages=find_packages(),
    install_requires=['pandas', 'seaborn', 'kaldialign', 'scikit-learn', 'plotly', 'jupyterlab', 'matplotlib', 'numpy', 'tqdm', 'iteration_utilities', 'scipy', 'networkx'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed"
    ],
)
