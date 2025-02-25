'''
Here we are making our own custorm Error Exception handler which shall be
handling the errors and will be giving the custon error messages.
'''

import sys 
import logging

def error_message_detail(error, error_detail : sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = 'Error occured in Python script:\t[{0}]\nline number\t[{1}]\nerror message\t[{2}]'.format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message