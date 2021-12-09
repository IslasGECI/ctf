from os import path
from pathlib import Path
import glob
import numpy as np
import pandas as pd


def load_submission(path_to_submission):
    submission = pd.read_csv(path_to_submission, index_col="id")
    return submission


def get_submission_list(path_to_submission_directory):
    submission_list_path = path.join(path_to_submission_directory, "*_submission.csv")
    submission_list = glob.glob(submission_list_path)
    return submission_list


class Referee:
    def __init__(self, path_to_complete_dataset: str):
        self.path_to_complete_dataset = path_to_complete_dataset
        self.data = self.load_complete_dataset()
        self.__get_dataset = {
            "training": self.get_training_dataset(),
            "testing": self.get_testing_dataset(),
            "submission": self.get_example_submission(),
        }
        self.__get_path_to = {
            "training": self.get_training_path(),
            "testing": self.get_testing_path(),
            "submission": self.get_example_submission_path(),
        }

    def load_complete_dataset(self):
        data = pd.read_csv(self.path_to_complete_dataset, index_col="id")
        self.__check_explanatory_variables_are_not_empty(data)
        return data

    def __check_explanatory_variables_are_not_empty(self, data):
        is_column_of_insterest = data.columns != "target"
        is_record_with_no_values = data.loc[:, is_column_of_insterest].isnull().all(axis=1).any()
        if is_record_with_no_values:
            raise ValueError("There is a record with no values for any explanatory variable")

    def get_training_length(self):
        total_length = len(self.data)
        training_proportion = 0.8
        training_length = round(total_length * training_proportion)
        return training_length

    def get_testing_length(self):
        total_length = len(self.data)
        training_length = self.get_training_length()
        testing_length = total_length - training_length
        return testing_length

    def get_training_dataset(self):
        training_length = self.get_training_length()
        train = self.data.head(training_length)
        return train

    def get_testing_dataset(self):
        testing_length = self.get_testing_length()
        test = self.data.iloc[-testing_length:, self.data.columns != "target"]
        return test

    def get_example_submission(self):
        testing_length = self.get_testing_length()
        example_submission = self.get_testing_dataset().copy()
        example_submission["target"] = np.random.rand(testing_length)
        example_submission = example_submission[["target"]]
        return example_submission

    def get_behind_the_wall_solution(self):
        testing_length = self.get_testing_length()
        solution = self.data[["target"]].tail(testing_length)
        return solution

    def get_training_path(self):
        return self.__get_path("train.csv")

    def get_testing_path(self):
        return self.__get_path("test.csv")

    def get_example_submission_path(self):
        return self.__get_path("example_submission.csv")

    def __get_path(self, file_csv):
        directory = str(Path(self.path_to_complete_dataset).parents[0])
        return path.join(directory, file_csv)

    def save_training_dataset(self):
        self.__save_dataset("training")

    def save_testing_dataset(self):
        self.__save_dataset("testing")

    def save_example_submission(self):
        self.__save_dataset("submission")

    def __save_dataset(self, csv):
        dataset = self.__get_dataset[csv]
        path = self.__get_path_to[csv]
        dataset = dataset.fillna("NA")
        dataset.to_csv(path)

    def init(self):
        self.save_training_dataset()
        self.save_testing_dataset()
        self.save_example_submission()

    def get_mean_absolute_error(self, path_to_submission):
        solution = self.get_behind_the_wall_solution()
        submission = self.__load_if_submission_is_correct(solution, path_to_submission)
        error = solution["target"] - submission["target"]
        mean_absolute_error = error.abs().mean()
        return mean_absolute_error

    def __load_if_submission_is_correct(self, solution, path_to_submission):
        submission = load_submission(path_to_submission)
        if set(solution.index) != set(submission.index):
            raise ValueError("La propuesta a solución no tiene la forma esperada")
        return submission

    def evaluate_single_submission(self, path_to_submission):
        mean_absolute_error = self.get_mean_absolute_error(path_to_submission)
        print(f"Submission: {path_to_submission}\nMean absolute error: {mean_absolute_error}")

    def get_mean_absolute_error_list(self, path_to_submission_directory):
        submission_list = get_submission_list(path_to_submission_directory)
        mean_absolute_error = [
            self.get_mean_absolute_error(submission) for submission in submission_list
        ]
        mean_absolute_error_list = pd.DataFrame.from_dict(
            {
                "submission": submission_list,
                "mean_absolute_error": mean_absolute_error,
            }
        )
        return mean_absolute_error_list.sort_values(by=["mean_absolute_error"])

    def evaluate_submission_directory(self, path_to_submission_directory):
        mean_absolute_error_list = self.get_mean_absolute_error_list(path_to_submission_directory)
        print(mean_absolute_error_list.to_markdown(index=False))
