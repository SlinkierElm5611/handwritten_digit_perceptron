import gzip


class IDXDataset:
    """
    A class to represent an IDX dataset which is parsed from a gzip file
    """

    def __init__(self, file_name: str):
        with open(file_name, "rb") as data:
            zipped_data: bytes = data.read()
            unzipped_data: bytes = gzip.decompress(data=zipped_data)
            self.dimensions: int = int.from_bytes(unzipped_data[3:4], byteorder="big")
            self.dimension_sizes: list[int] = [
                int.from_bytes(unzipped_data[4 + i : 8 + i], byteorder="big")
                for i in range(0, self.dimensions * 4, 4)
            ]
            self.data: list[int] = list(unzipped_data[self.dimensions * 4 + 4 :])
