from src.dataset import Dataset, SplitDataset


if __name__ == "__main__":
    d = SplitDataset(sequence_size=48 * 4,
                     encoding_type='timestep',
                     polyphonic=False,
                     chord_encoding_type='extended',
                     chord_extension_count=7,
                     transpose_mode='all')

    d.create()

    # d.load()
    # print(d.tensor_dataset)