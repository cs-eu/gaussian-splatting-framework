import subprocess
import logging

logger = logging.getLogger(__name__)


def run_command(command: str, check=True, silent=True, **kwargs):
    import os
    final_command = ["bash", "-c", command]

    # Copy all environment variables to the Popen process
    env = kwargs.pop("env", None)
    if env is not None:
        # Merge provided env with os.environ, with provided env taking precedence
        merged_env = os.environ.copy()
        merged_env.update(env)
    else:
        merged_env = os.environ.copy()

    process = subprocess.Popen(
        final_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=merged_env,
        **{k: v for k, v in kwargs.items() if k not in ["capture_output", "text", "env"]},
    )

    output_lines = []
    for line in process.stdout:
        if not silent:
            logger.info(line.rstrip())  # Stream to info
        else:
            logger.debug(line.rstrip())  # Stream to debug
        output_lines.append(line)
    process.stdout.close()
    returncode = process.wait()
    if returncode != 0 and check:
        raise subprocess.CalledProcessError(
            returncode, final_command, output="".join(output_lines)
        )

    # Mimic subprocess.CompletedProcess
    class Result:
        def __init__(self, stdout):
            self.stdout = stdout

    return Result("".join(output_lines))
