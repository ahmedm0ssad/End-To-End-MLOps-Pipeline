# Use the official Miniconda3 image as the base image
# This gives us a Linux environment with Conda pre-installed
FROM continuumio/miniconda3

# Set the working directory inside the container to /app
# All subsequent commands will run from this directory
WORKDIR /app

# Copy ONLY the environment file first (before the rest of the code)
# This leverages Docker layer caching — packages won't reinstall
# unless environment.yml actually changes
COPY environment.yml .

# Create the conda environment using the yml file
# This installs Python 3.10 + all packages (numpy, pandas, mlflow, etc.)
# Runs ONCE at build time, not every time the container starts
RUN conda env create -f environment.yml

# Change the default shell so all future RUN commands
# automatically execute inside the 'mlops-dev' conda environment
# Without this, RUN commands would use system Python (wrong!)
SHELL ["conda", "run", "-n", "mlops-dev", "/bin/bash", "-c"]

# Copy the rest of the project files into /app
# Done AFTER environment setup to maximize layer cache efficiency
# Code changes won't trigger a full package reinstall
COPY . .

# Default command to run when the container starts
# Explicitly runs train.py inside the mlops-dev conda environment
# Can be overridden at runtime: docker run my-image python evaluate.py
CMD ["conda", "run", "-n", "mlops-dev", "python", "src/train.py"]