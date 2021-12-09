from ..ctf import get_submission_list, Referee
import pytest

path_to_submission_directory = "tests/test_dataset_incomplete_submission/"
path_to_complete_dataset = path_to_submission_directory + "complete_dataset.csv"
ctf = Referee(path_to_complete_dataset)


@pytest.mark.parametrize(
    "file", ["incomplete_submission.csv", "overfull_submission.csv", "unexpected_submission.csv"]
)
def test_load_submission(file):
    path_to_submission = path_to_submission_directory + file
    with pytest.raises(ValueError, match="^La propuesta a solución no tiene la forma esperada$"):
        ctf.get_mean_absolute_error(path_to_submission)


def test_get_submission_list():
    obtained_submission_list = get_submission_list(path_to_submission_directory)
    expected_submission_list = [
        path_to_submission_directory + "incomplete_submission.csv",
        path_to_submission_directory + "overfull_submission.csv",
        path_to_submission_directory + "unexpected_submission.csv",
    ]
    assert sorted(expected_submission_list) == sorted(obtained_submission_list)
