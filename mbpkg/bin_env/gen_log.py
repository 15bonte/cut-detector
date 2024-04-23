from os.path import join

LOG_DIR = "src/cut_detector/data/mid_bodies_movies_test/log"

def make_log_file(name: str) -> str:
    return join(LOG_DIR, name)

