import sys
from src.logger import logging

class CustomException(Exception):
    def __init__(self, error_message, err_detail:sys) -> None:
        super().__init__(error_message)
        self.error_message = self.error_message_detail(error_message, error_detail=err_detail)
        
    def __str__(self) -> str:
        return self.error_message
    
    def error_message_detail(self, error, error_detail:sys):
        _,_, exc_tb = error_detail.exc_info()
        file_name=exc_tb.tb_frame.f_code.co_filename
        error_message="Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(file_name,
                                                                                                                exc_tb.tb_lineno,
                                                                                                                str(error))
        logging.error(error_message)
        return error_message