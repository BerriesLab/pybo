import pickle


def read_pickle_file(file_path):
    """Reads a pickle file and prints its content to the screen."""
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print("Pickle file content:\n", data)
    except Exception as e:
        print("An error occurred while reading the pickle file:", e)


# Example usage
if __name__ == "__main__":
    # Replace 'example.pkl' with the actual path to your pickle file
    read_pickle_file('../../data/experiment.dat')