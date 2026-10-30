from setuptools import setup, find_packages

setup(
    name="speech_corpus",
    version="1.0.0",
    description="Speech Corpus Pipeline - Steps 1 to 4",
    author="Amr Gamal",
    author_email="amrgamaljimy@gmail.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.9.0",
        "pandas>=1.5.0",
        "matplotlib>=3.6.0",
        "yt-dlp>=2024.0.0",
        "torch>=2.0.0",
    ],
)
