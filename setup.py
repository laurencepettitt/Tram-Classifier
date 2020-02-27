from distutils.core import setup

setup(
    name="tram-classifier",
    version="0.1.0",
    description="An audio signal based classifier for Prague's Trams",
    authors=["Laurence Pettitt"],
    license="MIT",
    python_requires='>=3.7.0',
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'librosa',
        'matplotlib',
        'tensorflow',
        'dload'
    ],
)
