"""
functions used to interact with the user
"""
from os.path import join

from foilselector.fluxconversion.filereading import open_csv

def get_column_interactive(directory, datatypename, first_time_use=False):
    """
    Ask the user for the column in a csv file within the specified {directory}, containing the {datatypename}.
    Keep asking until it is successfully found.
    Parameters
    ----------
    directory : location to look for csv.
    datatypename : name of the datatype which is displayed to the user when asking the question.
    first_time_use : if False, modifies the prompt question by appending the string "(Can be the same file as above)",
                     so that the user intuitively understands that the same file as the one used to answer the question in the previous call to this function can be used.
    """
    while True:
        try:
            prompt = ""
            if not first_time_use:
                prompt = "(Can be the same file as above)"
            fname = input(f"Which of the above file contains values for the {datatypename}?"+prompt)
            df, col = open_csv(join(directory,fname))
            print("Opened\n", df.head(), "\n...")
            colname = input(f"Please input the index/name of the column where {datatypename} is/are contained.\n(column name options include {list(col)})")
            if colname in col:
                col_i = colname
            else:
                col_i = col[int(colname)]
            break
        except Exception as e:
            print(e, "perhaps the file name/ column name is wrong/ index is too high. Please try again.")
    dataseries = df[col_i]
    print(f"using\n{dataseries.head()}\n...")
    return dataseries.values

def ask_question(question, expected_answer_list, check=True):
    """
    Ask the user a multiple choice question.
    Parameters
    ----------
    question : entire string of the question.
    expected_answer_list : list of strings which are the expected answers.
                           If the user gives an answer that isn't included the list, and check=True,
                           then their answer will be discarded and the quesiton will be asked again until a matching answer is found.
    check : see expected_answer_list

    Returns
    -------
    answer given by user
    """
    while True:
        answer = input(question)
        if (not check) or (answer in expected_answer_list):
            break
        print(f"Option {answer} not recognized; please retry:")
    print()
    return answer

def ask_yn_question(question):
    """
    Ask a yes no question
    parameters
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

    returns
    -------
    Boolean (True/False)
    """
    while True:
        answer = input(question+"('y','n')")
        if answer.lower() in ['yes', 'y']:
            return True
        elif answer.lower() in ['no', 'n']:
            return False
        else:
            print(f"Option '{answer}' not recognized; please retry:")
