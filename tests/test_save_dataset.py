from ..ctf import Referee
import os
import pandas as pd

path_to_submission_directory = "tests/test_dataset_pollos_petrel/"
path_to_complete_dataset = path_to_submission_directory + "complete_dataset.csv"

ctf = Referee(path_to_complete_dataset)

def test_save_training_dataset_includes_na():
    path_to_training = ctf.get_training_path()
    if os.path.exists(path_to_training):
        os.remove(path_to_training)
    ctf.save_training_dataset()
    obtained_count_na = pd.read_csv(path_to_training).isna().sum().sum()
    expected_count_na = 31
    assert expected_count_na == obtained_count_na
    os.remove(path_to_training)


