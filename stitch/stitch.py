"""

"""
# see https://github.com/jupyter/nbconvert/blob/master/nbconvert/preprocessors/execute.py
CODE = 'code'


class Stitcher:
    '''
    Takes a tokenized document and

    1. Executes the code chunts
    2. Stitches the output into the token stream
    '''

    def __init__(self, tokens):
        self._tokens = tokens
        self._engines = {}

    def get_engine(self, chunk):
        pass

    def stitch(self, tokens):
        output = []

        for chunk in tokens:
            if include_chunk(chunk):
                output.append(chunk)
            if is_executable(chunk):
                engine = self.get_engine(chunk)
                result = wrap_result(chunk, execute_chunk(chunk, engine))
                if include_output(chunk):
                    output.append(result)


def is_executable(chunk):
    return chunk['type'] == CODE  # TOOD: and execuable


def include_chunk(chunk):
    return True  # TODO: echo=False


def include_output(chunk):
    return True


def wrap_result(chunk, result):
    pass


def execute_chunk(chunk, engine):
    with engine.execution_context() as ctx:  # todo
        result = ctx.execute(chunk)
    return result

