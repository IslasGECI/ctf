class Test_Referee:
    def test_length_from_dataset(self, expected_length, dataset, referee):
        self.__assert_length_dataset(expected_length, dataset, referee)

    def __assert_length_dataset(self, expected_length, dataset, referee):
        get_length_from_dataset = {
            "testing": referee.get_testing_length(),
            "training": referee.get_training_length(),
        }
        obtained_length = get_length_from_dataset[dataset]
        assert expected_length == obtained_length
