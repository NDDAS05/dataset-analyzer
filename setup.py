from setuptools import setup, find_packages

setup(
    name='dataset-analyzer',
    version='1.0.0',
    author='Nirupam Das',
    description='Automated Dataset Analyzer — full EDA report from any CSV',
    python_requires='>=3.8',
    packages=find_packages(),
    install_requires=[
        'streamlit>=1.32.0',
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.12.0',
        'scikit-learn>=1.1.0',
        'jinja2>=3.0.0',
    ],
    entry_points={
        'console_scripts': [
            'analyze=main:main',
        ],
    },
    include_package_data=True,
    package_data={
        'src': ['templates/*.html'],
    },
)
