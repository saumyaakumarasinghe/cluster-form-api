import subprocess
import sys

# define available tasks and their respective shell commands
tasks = {
    "format": ["black", "."],  # run Black to format code
    "start": ["python", "main.py", "run"],  # start the API server
    "start-flask": ["flask", "run"],
}

# check if the user provided a valid task name as a command-line argument
if len(sys.argv) < 2 or sys.argv[1] not in tasks:
    print("Usage: python tasks.py [format|start]")  # show usage info
    sys.exit(1)  # exit with an error code

# execute the selected task
subprocess.run(tasks[sys.argv[1]])

# run - python tasks.py start
