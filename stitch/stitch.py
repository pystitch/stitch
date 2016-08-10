"""
The short version is

1. Use pandoc / pypandoc to convert markdown (or whaterver)
to pandoc's JSON AST
2. Walk the AST looking for code-blocks that are to be executed
3. Submit code-blocks to the appropriate kernel
4. Capture output
5. Convert outupt to pandoc's JSON AST
6. stitch output in after the original code block
7. Use pandoc / pypandoc to convert the AST to output.

"""
import json
from queue import Empty
from collections import namedtuple
from jupyter_client.manager import start_new_kernel
from pandocfilters import Para, Str
import pypandoc

# see https://github.com/jupyter/nbconvert/blob/master/nbconvert/preprocessors/execute.py
CODE = 'code'
CODEBLOCK = 'CodeBlock'

KernelPair = namedtuple("KernelPair", "km kc")


def convert(source, to, extra_args=(), outputfile=None, filters=None):
    newdoc = stitch(source)
    return pypandoc.convert_text(newdoc, to, format='json', outputfile=outputfile)


def stitch(source: str, kernel_name='python'):

    meta, blocks = tokenize(source)
    needed_kernels = extract_kernel_names(blocks)
    kernels = {name: KernelPair(*start_new_kernel(kernel_name=kernel_name))
               for name in needed_kernels}

    new_blocks = []
    for block in blocks:
        new_blocks.append(block)
        if to_execute(block):
            result = execute_block(block, kernels)
            result = wrap(result)
            new_blocks.append(result)

    doc = json.dumps([meta, new_blocks])
    return doc


def extract_kernel_names(blocks):
    return set(x['c'][0][1][0] for x in blocks if to_execute(x))


def wrap(output):
    # TODO: All kinds of output, not just text.
    return Para([Str(output[0]['text/plain'])])


def tokenize(source: str) -> dict:
    return json.loads(pypandoc.convert_text(source, 'json', 'markdown'))


def to_execute(x):
    return x['t'] == CODEBLOCK and ['eval', 'False'] not in x['c'][0][2]


def execute_block(block, kc, timeout=None):
    # see nbconvert.run_cell
    code = block['c'][1]
    outs = run_code(code, kc, timeout=timeout)
    return outs
    # log


def run_code(code, kp, timeout=None):
    msg_id = kp.kc.execute(code)
    print(msg_id)
    while True:
        try:
            msg = kp.kc.shell_channel.get_msg(timeout=timeout)
        except Empty:
            print("emtpy")
            # self.log.error(
            #     "Timeout waiting for execute reply (%is)." % self.timeout)
            # if self.interrupt_on_timeout:
            #     self.log.error("Interrupting kernel")
            #     self.km.interrupt_kernel()
            # else:
            #     try:
            #         exception = TimeoutError
            #     except NameError:
            #         exception = RuntimeError
            #     raise exception(
            #         "Cell execution timed out, see log for details.")

        if msg['parent_header'].get('msg_id') == msg_id:
            break
        else:
            # not our reply
            continue

    outs = []

    while True:
        try:
            # We've already waited for execute_reply, so all output
            # should already be waiting. However, on slow networks, like
            # in certain CI systems, waiting < 1 second might miss messages.
            # So long as the kernel sends a status:idle message when it
            # finishes, we won't actually have to wait this long, anyway.
            msg = kp.kc.iopub_channel.get_msg(timeout=4)
            print(msg, end='n' + '*' * 80 + '\n\n')
        except Empty:
            # self.log.warn("Timeout waiting for IOPub output")
            print("Timeout waiting for IOPub output")
            # if self.raise_on_iopub_timeout:
            #    # raise RuntimeError("Timeout waiting for IOPub output")
        if msg['parent_header'].get('msg_id') != msg_id:
            # not an output from our execution
            continue

        msg_type = msg['msg_type']
        # self.log.debug("output: %s", msg_type)
        content = msg['content']

        if msg_type == 'status':
            if content['execution_state'] == 'idle':
                break
            else:
                continue
        elif msg_type == 'execute_input':
            continue
        elif msg_type == 'clear_output':
            outs = []
            continue
        elif msg_type.startswith('comm'):
            continue

        try:
            out = output_from_msg(msg)
        except ValueError:
            # self.log.error("unhandled iopub msg: " + msg_type)
            print("bad")
        else:
            outs.append(out)

    return outs


def output_from_msg(msg):
    return msg['content']['data']


