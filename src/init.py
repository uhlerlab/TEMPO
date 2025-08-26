import os
import sys
import torch
from pykeops.torch import LazyTensor


def set_cuda_path_in_conda_env():
    # Get the base directory of your Conda environment
    conda_env_path = os.path.dirname(os.path.dirname(sys.executable))
    print("Base directory of the Conda environment is: conda_env_path")
    print("Setting ...")
    os.environ['CUDA_HOME'] = conda_env_path
    # Add the CUDA compiler's bin directory to the PATH
    # This ensures the compiler (nvcc) can be found.
    bin_path = os.path.join(conda_env_path, 'bin')
    os.environ['PATH'] = bin_path + os.pathsep + os.environ.get('PATH', '')
    
    # Add the CUDA library path for the linker
    # This helps find libraries like libcudart.so
    lib_path = os.path.join(conda_env_path, 'lib')
    os.environ['LD_LIBRARY_PATH'] = lib_path + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
    
    print(f"Set CUDA_HOME to: {os.environ['CUDA_HOME']}")
    print("Updated PATH and LD_LIBRARY_PATH for the kernel.")
    return


# Inside your_package/utils.py

def check_cuda_setup():
    """
    Checks if the pykeops CUDA compilation is working correctly.  
    Important for fast and memory-efficient computation of OT plans with pykeops and geomloss. 
    """
    # Define the ANSI codes as constants for readability
    BOLD = '\033[1m'
    RESET = '\033[0m'

    print("Checking pykeops GPU configuration...")
    try:
        # A simple pykeops operation that forces a compilation check
        import torch
        from pykeops.torch import LazyTensor
        _ = LazyTensor(torch.randn(1, 3, device='cuda')).sum(dim=1)
        print("{BOLD} Success! Your pykeops installation is correctly configured for GPU usage. {RESET}")
        return True
    except Exception as e:
        print(f"\n {BOLD} Error: Pykeops failed to compile for GPU.")
        print(f"This is likely an issue with your CUDA Toolkit installation. \n{RESET} We use pykeops and geomloss for efficient GPU computations of optimal transport plans with 'LazyTensor'.")
        print("For this to work, you must have a working NVIDIA CUDA Toolkit installation in your environment.")
        print("Please ensure 'nvcc' compiler is available and is in your PATH and also that CUDA_HOME is set.")
        print(f"{BOLD}We recommend installing the CUDA Toolkit within a conda environment for a self-contained setup via: {RESET}")
        print("conda install -c nvidia cuda-toolkit.")
        print("Then, once you are running TEMPO in your conda environment, you can configure PATH and CUDA_HOME by running the provided 'configure_cuda_in_conda_env()' function.")
        print("Afterwards, please call 'check_cuda_setup()' to verify it worked correctly.")
        print(f"\n{BOLD}If you are unable to do this, you have two options: {RESET}")
        print("  a) You can still use geomloss + pykeops with CPU")
        print("  b) or you can switch to `POT` (PythonOT) for optimal transport map computation")
        print("Please note that both of these options likely reduce computational and/or memory efficiency of computing optimal transport plans.")
        print("\nOriginal error message:")
        print(e)
        return False