import sys
import psutil


def display_progress(
    message,
    current,
    total,
    precision=1,
    additional_message="",
    cpu_memory=False,
):
    percentage = round(current / total * 100, precision)
    padded_percentage = str(percentage).ljust(precision + 3, "0")
    display_message = f"\r{message}: {padded_percentage}%"
    # Display additional message
    if additional_message:
        display_message += " | " + additional_message
    # Display CPU memory usage
    if cpu_memory:
        cpu_available = round(
            psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
        )
        cpu_message = f"CPU available: {cpu_available}%"
        display_message += " | " + cpu_message
    sys.stdout.write(display_message)
    sys.stdout.flush()
