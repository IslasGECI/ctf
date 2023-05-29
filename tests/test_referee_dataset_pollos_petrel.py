from ..ctf import get_submission_list, load_submission, Referee
from .utils_tests import _test_length_from_dataset, _test_get_path
import pandas as pd
import pytest

path_to_submission_directory = "tests/test_dataset_pollos_petrel/"
path_to_complete_dataset = path_to_submission_directory + "complete_dataset.csv"
referee = Referee(path_to_complete_dataset)
path_to_submission = path_to_submission_directory + "test_a_submission.csv"


@pytest.mark.parametrize(
    "expected_length, dataset, referee",
    [(round(10 * 0.8), "training", referee), (round(10 * 0.2), "testing", referee)],
)
def test_length_from_dataset(expected_length, dataset, referee):
    _test_length_from_dataset(expected_length, dataset, referee)


def test_load_complete_dataset():
    data = referee.load_complete_dataset()
    obtained_length = len(data)
    expected_length = 10
    assert expected_length == obtained_length


def test_get_training_length():
    obtained_length = referee.get_training_length()
    expected_length = round(10 * 0.8)
    assert expected_length == obtained_length


def test_get_testing_length():
    obtained_length = referee.get_testing_length()
    expected_length = round(10 * 0.2)
    assert expected_length == obtained_length


def test_get_testing_dataset():
    test = referee.get_testing_dataset()
    obtained_rows = len(test)
    expected_rows = referee.get_testing_length()
    assert expected_rows == obtained_rows
    obtained_column_names = list(test.columns)
    expected_column_names = [
        "Peso",
        "Longitud_tarso",
        "Longitud_ala",
        "Longitud_pico",
        "Longitud_pluma_interior_de_la_cola",
        "Longitud_pluma_exterior_de_la_cola",
    ]
    assert obtained_column_names == expected_column_names


@pytest.mark.parametrize(
    "obtained_path, expected_path",
    [
        (referee.get_testing_path(), path_to_submission_directory + "test.csv"),
        (referee.get_training_path(), path_to_submission_directory + "train.csv"),
        (
            referee.get_example_submission_path(),
            path_to_submission_directory + "example_submission.csv",
        ),
    ],
)
def test_get_path(obtained_path, expected_path):
    _test_get_path(obtained_path, expected_path)


def test_load_submission():
    submission = load_submission(path_to_submission)
    obtained_length = len(submission)
    expected_length = referee.get_testing_length()
    assert expected_length == obtained_length


def test_get_mean_absolute_error():
    obtained_mean_absolute_error = round(referee.get_mean_absolute_error(path_to_submission), 16)
    expected_mean_absolute_error = 9.5
    assert expected_mean_absolute_error == obtained_mean_absolute_error


def test_get_submission_list():
    obtained_submission_list = get_submission_list(path_to_submission_directory)
    expected_submission_list = [
        path_to_submission_directory + "test_a_submission.csv",
        path_to_submission_directory + "test_b_submission.csv",
    ]
    assert sorted(expected_submission_list) == sorted(obtained_submission_list)


def test_get_mean_absolute_error_list():
    obtained_mean_absolute_error_list = referee.get_mean_absolute_error_list(
        path_to_submission_directory
    )
    expected_mean_absolute_error_list = pd.DataFrame(columns=["submission", "mean_absolute_error"])
    expected_mean_absolute_error_list = pd.concat(
        [
            expected_mean_absolute_error_list,
            pd.DataFrame(
                [
                    {
                        "submission": path_to_submission_directory + "test_a_submission.csv",
                        "mean_absolute_error": 9.5,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    expected_mean_absolute_error_list = pd.concat(
        [
            expected_mean_absolute_error_list,
            pd.DataFrame(
                [
                    {
                        "submission": path_to_submission_directory + "test_b_submission.csv",
                        "mean_absolute_error": 20.5,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(
        expected_mean_absolute_error_list.reset_index(drop=True),
        obtained_mean_absolute_error_list.reset_index(drop=True),
        check_dtype=False,
    )
