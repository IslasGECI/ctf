import os


def _test_length_from_dataset(expected_length, dataset, referee):
    assert_length_dataset(expected_length, dataset, referee)


def assert_length_dataset(expected_length, dataset, referee):
    get_length_from_dataset = {
        "testing": referee.get_testing_length(),
        "training": referee.get_training_length(),
    }
    obtained_length = get_length_from_dataset[dataset]
    assert expected_length == obtained_length


def _test_get_path(obtained_path, expected_path):
    assert expected_path == obtained_path


def _remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)
