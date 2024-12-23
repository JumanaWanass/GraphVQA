

### Step 1: Install Anaconda
Ensure Anaconda is installed on your system. If not, [download and install it here](https://www.anaconda.com/).

---

### Step 2: Create a Conda Environment
1. Open your terminal or Anaconda Prompt.
2. Run the following command to create a new environment with Python 3.6:
   ```bash
   conda create -n graphvqa python=3.6 anaconda
   ```

3. Activate the environment:
   ```bash
   conda activate graphvqa
   ```

---

### Step 3: Install PyTorch with GPU Support
1. Install the GPU-compatible PyTorch version:
   ```bash
   pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. Verify the installation:
   ```bash
   python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
   ```
   This should display `1.4.0` for PyTorch and `10.0` for CUDA.

---

### Step 4: Install PyTorch Geometric Dependencies
1. Install the `torch-scatter` library:
   ```bash
   pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
   ```

2. Install the `torch-sparse` library:
   ```bash
   pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
   ```

3. Install the `torch-cluster` library:
   ```bash
   pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
   ```

4. Install the `torch-spline-conv` library:
   ```bash
   pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
   ```

5. Finally, install the `torch-geometric` library:
   ```bash
   pip install torch-geometric
   ```

---

### Step 5: Install Other Required Libraries
1. Install `nltk` (compatible version):
   ```bash
   pip install nltk==3.5
   ```

2. Install `numpy`, `pandas`, and `matplotlib`:
   ```bash
   pip install numpy pandas matplotlib
   ```

3. Install any other project-specific dependencies listed in the repository's `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---
### Step 6: Import NLTK
Inside the conda environment, run the following to download the required NLTK data:
   ```python
   python
   import nltk
   nltk.download('wordnet')
   ```
### Step 7: Verify the Environment Setup
1. Run the following script to confirm all required libraries are installed:
   ```bash
   python -c "import torch; import torch_geometric; import nltk; print('Setup successful!')"
   ```

2. Ensure GPU compatibility is enabled:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   This should return `True` if GPU support is available.

---


### Notes:
- Ensure your GPU drivers and CUDA Toolkit version (10.0) are installed and match the PyTorch CUDA compatibility.
- If a library still fails to install, verify its compatibility with the specific versions of PyTorch and CUDA you're using.

