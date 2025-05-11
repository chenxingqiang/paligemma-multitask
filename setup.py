from setuptools import setup, find_packages

setup(
    name="detection-reporter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "Pillow>=9.0.0",
        "numpy>=1.21.0",
    ],
    entry_points={
        'console_scripts': [
            'generate-report=inference:main',
        ],
    },
    author="Xingqiang Chen",
    description="A tool for generating technical documentation from PaliGemma model results",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
) 