# function to look at .h5 file contents

import h5py

def h5disp(filename):
    """
    Display the structure and contents of an HDF5 file.
    
    Parameters:
        filename (str): Path to the HDF5 file.
    """
    def display_item(name, obj, indent=0):
        """Recursively display information about HDF5 groups and datasets."""
        spacing = '  ' * indent
        if isinstance(obj, h5py.Group):
            print(f"{spacing}Group: {name}")
            for key, item in obj.items():
                display_item(key, item, indent + 1)
        elif isinstance(obj, h5py.Dataset):
            print(f"{spacing}Dataset: {name}")
            print(f"{spacing}  Shape: {obj.shape}")
            print(f"{spacing}  Data type: {obj.dtype}")
            if obj.size < 20:  # Display small datasets inline
                print(f"{spacing}  Data: {obj[()]}")

    try:
        with h5py.File(filename, 'r') as h5file:
            print(f"HDF5 file: {filename}")
            for name, item in h5file.items():
                display_item(name, item)
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")

# Example usage:
# h5disp('example.h5')


def h5read(filename, dataset_path):
    """
    Read data from a specified dataset in an HDF5 file.
    
    Parameters:
        filename (str): Path to the HDF5 file.
        dataset_path (str): Path to the dataset within the HDF5 file.
    
    Returns:
        numpy.ndarray: The data from the specified dataset.
    """
    try:
        with h5py.File(filename, 'r') as h5file:
            if dataset_path in h5file:
                data = h5file[dataset_path][()]
                return data
            else:
                raise KeyError(f"Dataset '{dataset_path}' not found in file '{filename}'.")
    except Exception as e:
        print(f"Error reading dataset '{dataset_path}' from file '{filename}': {e}")
        return None


# # Import the function
# from h5_utils import h5read

# # Path to the HDF5 file
# filename = 'example.h5'

# # Path to the dataset within the file
# dataset_path = '/group1/dataset1'

# # Read and print the dataset
# data = h5read(filename, dataset_path)
# print(f"Dataset data: {data}")
