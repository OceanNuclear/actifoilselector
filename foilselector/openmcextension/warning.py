import numpy as np
import contextlib # to silence numpy error


class SilenceNumpyDivisionError(contextlib.ContextDecorator):
    def __enter__(self):
        self.prev_divide_error_state = np.geterr()["divide"] # record current state of error handling style
        np.seterr(divide="ignore") # force ignore all division errors
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        np.seterr(divide=self.prev_divide_error_state) # undo error silencing
        if exc_type is None:
            return True
        else: # any type of error
            return False