import sys
from src.logger import logging

def error_message_details(error, error_detail:sys):
    """
    This function captures detailed error information including:
    - The file name where the error occurred.
    - The line number where the error occurred.
    - The error message itself.
    """
    
    _,_,exc_tb = error_detail.exc_info()          # exc_tb: returns in which file & line the error has occured.
    file_name = exc_tb.tb_frame.f_code.co_filename # Getting the file name
    line_number = exc_tb.tb_lineno                 # Getting the line number where error occurred
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, line_number, str(error)
    )

    return error_message

# Customized Exception Handling class inheriting from base Exception class
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        """
        Initializes the CustomException with detailed error information.
        """

        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail = error_detail)

    def __str__(self):
        """
        Returns the error message as a string representation of the object.
        """
        return self.error_message