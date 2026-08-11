## Hardware and Software Environment

The proposed MC-STGAT-IDS model was developed and evaluated on a personal workstation running **Microsoft Windows 11**. GPU acceleration was utilized for training and evaluation of the deep learning and graph neural network models.

### Hardware Configuration

| Component | Specification |
|---|---|
| Operating System | Microsoft Windows 11 |
| CPU | AMD Ryzen 7 5800X 8-Core Processor |
| GPU | NVIDIA GeForce RTX 3070 |
| GPU Memory | 8 GB |
| System Memory (RAM) | 32 GB |
| GPU Driver | NVIDIA 596.49 |
| CUDA Version Reported by Driver | 13.2 |
| Primary Storage | Samsung SSD 970 EVO Plus 500GB |
| Additional Storage | 1 TB HDD + 2 TB HDD |

### Software Environment

| Software / Library | Version |
|---|---:|
| Python | 3.11.9 |
| PyTorch | 2.13.0+cu126 |
| PyTorch Geometric | 2.8.0.post1 |
| Torchvision | 0.28.0+cu126 |
| Torchaudio | 2.11.0+cu126 |
| CUDA Build | 12.6 |
| NumPy | 2.4.6 |
| Pandas | 3.0.3 |
| Scikit-learn | 1.9.0 |
| SciPy | 1.17.1 |
| NetworkX | 3.6.1 |
| Matplotlib | 3.11.0 |
| Seaborn | 0.13.2 |
| UMAP-learn | 0.5.12 |
| Pillow | 12.3.0 |
| Plotly | 6.9.0 |
| tqdm | 4.70.0 |

### Deep Learning and Graph Learning Framework

The implementation was primarily developed using **PyTorch 2.13.0+cu126** for deep learning operations and **PyTorch Geometric 2.8.0.post1** for graph-based deep learning and graph neural network operations. GPU acceleration was enabled using the CUDA-enabled PyTorch build.

The experiments were performed using an **NVIDIA GeForce RTX 3070 with 8 GB of VRAM**, an **AMD Ryzen 7 5800X processor**, and **32 GB of system memory**. The NVIDIA driver version was **596.49**. The driver reported CUDA **13.2** compatibility, while the installed PyTorch package was built with **CUDA 12.6**.

### Development Environment

The experiments were conducted within a Python virtual environment (`.venv`) on a native Windows 11 installation. Jupyter-based tools were used for interactive experimentation and development.

| Development Tool | Version |
|---|---:|
| JupyterLab | 4.6.2 |
| Jupyter Notebook | 7.6.1 |
| IPython | 9.16.1 |
