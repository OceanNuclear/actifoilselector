"""Functions used to "talk" to the user."""

from pathlib import Path

import numpy as np
import pandas as pd

from foilselector.fluxconversion.filereading import open_csv

__all__ = ["ask_question", "ask_yn_question", "get_column_interactive"]


def ask_yn_question(question: str):
    """
    Ask a yes no question.

    Parameters
    ----------
    question : string containing the question to be displayed.

    question displayed
    ------------------
    question+"('y'/'n')"

    accepted inputs
    ---------------
    answer.lowercase() must be "yes", "y", "no" or "n".
    e.g.
    yes: ["yes", "y", "Yes", "YES", "Y"]
    no: ["no", "n", "No", "NO", "N"]

    Returns
    -------
    Boolean (True/False)
    """
    while True:
        answer = input(question + "('y','n') ")
        if answer.lower() in {"yes", "y"}:
            return True
        if answer.lower() in {"no", "n"}:
            return False
        print(f"Option '{answer}' not recognized; please retry: ")


def ask_question(question: str, expected_answer_list: list[str], *, check: bool = True):
    """
    Ask the user a multiple choice question.

    Parameters
    ----------
    question:
        entire string of the question.

    expected_answer_list:
        list of strings which are the expected answers.
        If the user gives an answer that isn't included the list, and check=True,
        then their answer will be discarded and the quesiton will be asked again until
        a matching answer is found.

    check:
        see expected_answer_list

    Returns
    -------
    :
        answer given by user
    """
    while True:
        answer = input(f"{question} ")
        if (not check) or (answer in expected_answer_list):
            break
        print(f"Option {answer} not recognized; please retry: ")
    print()
    return answer


def _check_is_numpy_numeric(data: np.ndarray):
    """
    Ensure that a numpy array is indeed a numeric type (int64, float64, etc.).

    Parameters
    ----------
    data:
        A numpy array.

    Raises
    ------
    ValueError
        Raised if the data array is not of a numeric data type.
    """
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("Expected numerical data, got " f"{type(data.dtype)}.")


def get_column_interactive(
    directory: Path,
    datatypename: str,
    *,
    first_time_use: bool = False,
    output_full_file_path: bool = False,
    file_path_given: Path | None = None,
) -> np.ndarray[float] | tuple[np.ndarray[float], str]:
    """
    Ask the user for the column in a csv file within the specified {directory},
    containing the {datatypename}.
    Keep asking until it is successfully found.
    Data must be numeric in type.

    Parameters
    ----------
    directory:
        location to look for csv.
    datatypename:
        name of the data type which is displayed to the user when asking the question.
    first_time_use:
        if False, modifies the prompt question by appending the string
        "(Can be the same file as above)", so that the user intuitively understands that
        the same file as the one used to answer the question in the previous call to this
        function can be used.

    Returns
    -------
    dataseries:
        A 1D np.array containing the data of the user-chosen column in the csv.
    """

    def one_loop(file_path) -> tuple[pd.Series | None, int]:
        """
        Perform a single attempt of opening a csv and interactively finding the column.

        Parameters
        ----------
        file_path: str
            csv file to open

        Returns
        -------
        column_data: pd.Series | None
            column chosen
        exit_status: int
            If exit status is non-zero, then an error has occured
        """
        try:
            file_path = Path(file_path)
            df, col = open_csv(file_path)
            print(f"Opened {file_path.name}\n", df.head(), "\n...")
            colname = input(
                f"Please input the index/name of the column where {datatypename} is/are "
                "contained.\n(column name options include {list(col)}) ",
            )
            col_i = colname if colname in col else col[int(colname)]
            _check_is_numpy_numeric(df[col_i].to_numpy()[0])

            return df[col_i], 0

        except (FileNotFoundError, IsADirectoryError) as e:
            print(e, f"Please enter a valid file in {directory}.")
        except ValueError as e:
            print(e, "Perhaps the column name/ index is wrong. Please try again.")

        return None, 1

    exit_status = 1
    if file_path_given:
        dataseries, exit_status = one_loop(file_path_given)
        if exit_status == 0:
            full_file_path = file_path_given

    while exit_status:
        prompt = f"Which of the above file contains values for the {datatypename}?"
        if not first_time_use:
            prompt += "(It may be the same file as previously used.)"
        fname = input(f"{prompt} ")
        full_file_path = Path(directory, fname)
        dataseries, exit_status = one_loop(full_file_path)

    print(f"Data chosen =\n{dataseries.head()}\n...")

    if output_full_file_path:
        return dataseries.to_numpy(), full_file_path
    return dataseries.to_numpy()
