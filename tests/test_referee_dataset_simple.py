from ..ctf import get_submission_list, load_submission, Referee
import os
import pandas as pd

path_to_complete_dataset = "tests/test_dataset_simple/complete_dataset.csv"
referee = Referee(path_to_complete_dataset)
path_to_submission = "tests/test_dataset_simple/test_submission.csv"
path_to_submission_directory = "tests/test_dataset_simple"


def test_load_complete_dataset():
    data = referee.load_complete_dataset()
    obtained_length = len(data)
    expected_length = 27
    assert expected_length == obtained_length


def test_get_training_length():
    obtained_length = referee.get_training_length()
    expected_length = round(27 * 0.8)
    assert expected_length == obtained_length


def test_get_testing_length():
    obtained_length = referee.get_testing_length()
    expected_length = round(27 * 0.2)
    assert expected_length == obtained_length


def test_get_training_dataset():
    train = referee.get_training_dataset()
    obtained_rows, obtained_cols = train.shape
    expected_rows = referee.get_training_length()
    assert expected_rows == obtained_rows
    expected_cols = referee.data.shape[1]
    assert expected_cols == obtained_cols


def test_get_testing_dataset():
    test = referee.get_testing_dataset()
    obtained_rows = len(test)
    expected_rows = referee.get_testing_length()
    assert expected_rows == obtained_rows
    obtained_column_names = list(test.columns)
    expected_column_names = ["x"]
    assert obtained_column_names == expected_column_names


def test_get_example_submission():
    example_submission = referee.get_example_submission()
    obtained_rows = len(example_submission)
    expected_rows = referee.get_testing_length()
    assert expected_rows == obtained_rows
    obtained_column_names = list(example_submission.columns)
    expected_column_names = ["target"]
    assert expected_column_names == obtained_column_names
    obtained_example_target = example_submission["target"].iloc[0]
    assert obtained_example_target >= 0


def test_get_behind_the_wall_solution():
    solution = referee.get_behind_the_wall_solution()
    obtained_rows = len(solution)
    expected_rows = referee.get_testing_length()
    assert expected_rows == obtained_rows
    obtained_column_names = list(solution.columns)
    expected_column_names = ["target"]
    assert expected_column_names == obtained_column_names
    obtained_solution_target = solution["target"].iloc[0]
    expected_solution_target = referee.data["target"].iloc[referee.get_training_length()]
    assert expected_solution_target == obtained_solution_target


def test_get_training_path():
    obtained_path = referee.get_training_path()
    expected_path = "tests/test_dataset_simple/train.csv"
    assert expected_path == obtained_path


def test_get_testing_path():
    obtained_path = referee.get_testing_path()
    expected_path = "tests/test_dataset_simple/test.csv"
    assert expected_path == obtained_path


def test_get_example_submission_path():
    obtained_path = referee.get_example_submission_path()
    expected_path = "tests/test_dataset_simple/example_submission.csv"
    assert expected_path == obtained_path


def test_save_training_dataset():
    path_to_training = referee.get_training_path()
    if os.path.exists(path_to_training):
        os.remove(path_to_training)
    referee.save_training_dataset()
    assert os.path.exists(path_to_training)
    obtained_first_column = pd.read_csv(path_to_training).columns[0]
    expected_first_column = "id"
    assert expected_first_column == obtained_first_column
    os.remove(path_to_training)


def test_save_testing_dataset():
    path_to_testing = referee.get_testing_path()
    if os.path.exists(path_to_testing):
        os.remove(path_to_testing)
    referee.save_testing_dataset()
    assert os.path.exists(path_to_testing)
    obtained_first_column = pd.read_csv(path_to_testing).columns[0]
    expected_first_column = "id"
    assert expected_first_column == obtained_first_column
    os.remove(path_to_testing)


def test_save_example_submission():
    path_to_example_submission = referee.get_example_submission_path()
    if os.path.exists(path_to_example_submission):
        os.remove(path_to_example_submission)
    referee.save_example_submission()
    assert os.path.exists(path_to_example_submission)
    obtained_first_column = pd.read_csv(path_to_example_submission).columns[0]
    expected_first_column = "id"
    assert expected_first_column == obtained_first_column
    os.remove(path_to_example_submission)


def test_init():
    referee.init()
    path_to_training = referee.get_training_path()
    assert os.path.exists(path_to_training)
    os.remove(path_to_training)
    path_to_testing = referee.get_testing_path()
    assert os.path.exists(path_to_testing)
    os.remove(path_to_testing)
    path_to_example_submission = referee.get_example_submission_path()
    assert os.path.exists(path_to_example_submission)
    os.remove(path_to_example_submission)


def test_load_submission():
    submission = load_submission(path_to_submission)
    obtained_length = len(submission)
    expected_length = referee.get_testing_length()
    assert expected_length == obtained_length


def test_get_mean_absolute_error():
    obtained_mean_absolute_error = round(referee.get_mean_absolute_error(path_to_submission), 16)
    expected_mean_absolute_error = 0.4246
    assert expected_mean_absolute_error == obtained_mean_absolute_error


def test_get_submission_list():
    obtained_submission_list = get_submission_list(path_to_submission_directory)
    expected_submission_list = [
        "tests/test_dataset_simple/test_submission.csv",
        "tests/test_dataset_simple/test2_submission.csv",
    ]
    assert sorted(expected_submission_list) == sorted(obtained_submission_list)


def test_get_mean_absolute_error_list():
    obtained_mean_absolute_error_list = referee.get_mean_absolute_error_list(
        path_to_submission_directory
    )
    expected_mean_absolute_error_list = pd.DataFrame(columns=["submission", "mean_absolute_error"])
    expected_mean_absolute_error_list = expected_mean_absolute_error_list.append(
        {
            "submission": "tests/test_dataset_simple/test2_submission.csv",
            "mean_absolute_error": 0.2446,
        },
        ignore_index=True,
    )
    expected_mean_absolute_error_list = expected_mean_absolute_error_list.append(
        {
            "submission": "tests/test_dataset_simple/test_submission.csv",
            "mean_absolute_error": 0.4246,
        },
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(
        expected_mean_absolute_error_list.reset_index(drop=True),
        obtained_mean_absolute_error_list.reset_index(drop=True),
    )
