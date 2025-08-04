# Use NVIDIA CUDA base image with Python 3.12.9
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.12.9

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    libssl-dev \
    libffi-dev \
    libsqlite3-dev \
    libreadline-dev \
    libbz2-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libgdbm-dev \
    liblzma-dev \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12.9 from source
RUN cd /tmp && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make -j $(nproc) && \
    make altinstall && \
    cd / && \
    rm -rf /tmp/Python-${PYTHON_VERSION}*

# Create symlinks for python3.12
RUN ln -sf /usr/local/bin/python3.12 /usr/bin/python3
RUN ln -sf /usr/local/bin/python3.12 /usr/bin/python
RUN ln -sf /usr/local/bin/pip3.12 /usr/bin/pip

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set work directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY subtask2_final_gradio.py .

# Create directory for model weights
RUN mkdir -p experiments/exp027

# Set CUDA environment variables
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV CUDA_VISIBLE_DEVICES=0

# Expose port
EXPOSE 7860

# Create non-root user for security
RUN useradd -m -u 1002 appuser && chown -R appuser:appuser /app
USER appuser

# Command to run the application
CMD ["python", "subtask2_final_gradio.py"]
