import subprocess
import typing
from typing import IO


def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    stdout = typing.cast(IO[str], popen.stdout)
    for stdout_line in iter(stdout.readline, ""):
        yield stdout_line
    stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def execute_and_print(cmd):
    for line in execute(cmd):
        print(line, end="")
