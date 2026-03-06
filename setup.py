from setuptools import setup, find_packages

setup(
    name="jhcodec",
    version="0.1.1",
    description="JHCodecDAC: A neural audio codec with vector quantization",
    author="Anonymous",
    author_email="jhcodec843@gmail.com",
    packages=find_packages(),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio",
    ],
    keywords="audio codec neural network vector quantization machine learning",
    project_urls={
        "Bug Reports": "https://github.com/jhcodec843/jhcodec/issues",
        "Source": "https://github.com/jhcodec843/jhcodec",
    },
)
