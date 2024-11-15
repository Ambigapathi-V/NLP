from dataclasses import dataclass

@dataclass
class DataIngestionArtifacts:
    imblanced_data_path: str
    raw_data_file_path : str