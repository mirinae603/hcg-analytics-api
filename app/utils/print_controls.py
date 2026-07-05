from rich import print

# Rich color log functions
def print_info(msg): print(f"[bold blue][INFO][/bold blue] {msg}")
def print_success(msg): print(f"[bold green][SUCCESS][/bold green] {msg}")
def print_error(msg): print(f"[bold red][ERROR][/bold red] {msg}")
