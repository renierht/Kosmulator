from setuptools import setup, find_packages

setup(
    name="Kosmulator",  # Package name
    version="0.1.0",  # Initial version
    description="A Python package for Modified Gravity MCMC simulations.",
    long_description=open("README.md").read(),  # Use the README for the long description
    long_description_content_type="text/markdown",  # Content type for the long description
    author="Renier T. Hough",  # Replace with your name
    author_email="renierht@gmail.com",  # Replace with your email
    url="https://github.com/yourusername/Kosmulator",  # Replace with your GitHub repo
    packages=find_packages(),  # Automatically find packages
    include_package_data=True,  # Include non-code files specified in MANIFEST.in
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "emcee>=3.0.0",  # Example dependency for MCMC
        "getdist"
        "h5py",                      # Add h5py for file handling
    ],  # Specify dependencies
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum Python version
)
