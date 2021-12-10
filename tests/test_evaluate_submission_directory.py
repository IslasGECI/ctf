from ..ctf import Referee

path_to_submission_directory = "tests/test_dataset_simple/"
path_to_complete_dataset = path_to_submission_directory + "complete_dataset.csv"
path_to_submission = path_to_submission_directory + "test_submission.csv"

referee = Referee(path_to_complete_dataset)


def test_evaluate_submission_directory(capsys):
    referee.evaluate_submission_directory(path_to_submission_directory)
    captured = capsys.readouterr()
    obtained_output = captured.out
    expected_output = "| submission                                     |   mean_absolute_error |\n|:-----------------------------------------------|----------------------:|\n| tests/test_dataset_simple/test2_submission.csv |                0.2446 |\n| tests/test_dataset_simple/test_submission.csv  |                0.4246 |\n"
    assert expected_output == obtained_output
