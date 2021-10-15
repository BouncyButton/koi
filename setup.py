from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='koi',
    packages=find_packages(exclude=['build', 'doc', 'templates']),
    version='0.0.0',
    install_requires=[
        # "numpy",
        # "scipy",
        # "torch"
    ],
    python_requires='>=3.6',
    license="MIT",
    description='koi: wip package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Luca Bergamin',
    author_email='luca.bergamin95+github@gmail.com',
    url='https://github.com/bouncybutton/koi',
    download_url='https://github.com/bouncybutton/koi',
    keywords=['pytorch', 'machine-learning', 'variational-autoencoder', 'anomaly-detection'],
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
    ]
)
