from ..ctf import Referee

path_to_submission_directory = "tests/test_dataset_simple/"
path_to_complete_dataset = path_to_submission_directory + "complete_dataset.csv"
path_to_submission = path_to_submission_directory + "test_submission.csv"

referee = Referee(path_to_complete_dataset)


def test_evaluate_single_submission(capsys):
    referee.evaluate_single_submission(path_to_submission)
    captured = capsys.readouterr()
    obtained_output = captured.out
    expected_output = "Submission: tests/test_dataset_simple/test_submission.csv\nMean absolute error: 0.42460000000000003\n"
    assert expected_output == obtained_output
