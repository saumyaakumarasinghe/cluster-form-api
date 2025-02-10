import subprocess
import sys

# Define available tasks and their respective shell commands
tasks = {
    "format": ["black", "."],  # Run Black to format code
    "start": ["python", "main.py", "run"],  # Start the API server
    "start-flask": ["flask", "run"],
}

# Check if the user provided a valid task name as a command-line argument
if len(sys.argv) < 2 or sys.argv[1] not in tasks:
    print("Usage: python tasks.py [format|start]")  # Show usage info
    sys.exit(1)  # Exit with an error code

# Execute the selected task
subprocess.run(tasks[sys.argv[1]])

# run - python tasks.py start
