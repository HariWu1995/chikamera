import subprocess
from threading import Thread


def run_command(command: list) -> int:
    process = subprocess.run(command)
    return process.returncode


def run_command_in_thread(command: list) -> Thread:
    thread = Thread(target=run_command, args=(command,))
    thread.start()
    return thread

