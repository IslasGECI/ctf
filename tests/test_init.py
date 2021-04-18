import common_task_framework as ctf

path_to_complete_dataset = "tests/test_data/complete_dataset.csv"
data = ctf.load_complete_dataset(path_to_complete_dataset)


def test_load_complete_dataset():
    data = ctf.load_complete_dataset(path_to_complete_dataset)
    obtained_length = len(data)
    expected_length = 27
    assert expected_length == obtained_length


def test_get_training_length():
    obtained_length = ctf.get_training_length(data)
    expected_length = round(27 * 0.8)
    assert expected_length == obtained_length


def test_get_testing_length():
    obtained_length = ctf.get_testing_length(data)
    expected_length = round(27 * 0.2)
    assert expected_length == obtained_length


def test_get_training_dataset():
    train = ctf.get_training_dataset(data)
    obtained_rows, obtained_cols = train.shape
    expected_rows = ctf.get_training_length(data)
    assert expected_rows == obtained_rows
    expected_cols = data.shape[1]
    assert expected_cols == obtained_cols


def test_get_testing_dataset():
    test = ctf.get_testing_dataset(data)
    obtained_rows, obtained_cols = test.shape
    expected_rows = ctf.get_testing_length(data)
    assert expected_rows == obtained_rows
    expected_cols = data.shape[1] - 1
    assert expected_cols == obtained_cols


def test_get_example_submission():
    example_submission = ctf.get_example_submission(data)
    obtained_rows = len(example_submission)
    expected_rows = ctf.get_testing_length(data)
    assert expected_rows == obtained_rows
    obtained_column_names = list(example_submission.columns)
    expected_column_names = ["id", "target"]
    assert expected_column_names == obtained_column_names
    obtained_example_target = example_submission["target"].iloc[0]
    assert obtained_example_target >= 0


def test_get_behind_the_wall_solution():
    solution = ctf.get_behind_the_wall_solution(data)
    obtained_rows = len(solution)
    expected_rows = ctf.get_testing_length(data)
    assert expected_rows == obtained_rows
    obtained_column_names = list(solution.columns)
    expected_column_names = ["id", "target"]
    assert expected_column_names == obtained_column_names
    obtained_solution_target = solution["target"].iloc[0]
    expected_solution_target = data["target"].iloc[ctf.get_training_length(data)]
    assert expected_solution_target == obtained_solution_target
