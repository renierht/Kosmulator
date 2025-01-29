from setuptools import setup, find_packages
import shutil
import sys

# Function to check for LaTeX installation
def check_latex():
    if shutil.which('latex'):
        print("LaTeX is installed.")
    else:
        print(
            "\033[91mWarning:\033[0m LaTeX is not installed on your system. "
            "Please install LaTeX for better quality plots. Visit the following link for installation instructions:\n"
            "https://www.latex-project.org/get/\n"
            "Alternatively, install it via command-line:\n"
            "- Windows: https://miktex.org/howto/install-miktex\n"
            "- macOS: `brew install mactex`\n"
            "- Linux: `sudo apt install texlive-full`"
        )
        sys.exit(1)
        
setup(
    name="Kosmulator",
    version="0.1.0",
    description="A Python package for Modified Gravity MCMC simulations.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Renier T. Hough",
    author_email="renierht@gmail.com",
    url="https://github.com/renierht/Kosmulator",  # Update with your GitHub URL
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "emcee>=3.0.0",
        "getdist",
        "h5py",
    ],
    extras_require={
        "latex": ["latex"], 
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
