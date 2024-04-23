from os.path import join

DATA_DIR = "src/cut_detector/data/mid_bodies_movies_test/data"

def make_data_path(name: str) -> str:
    return join(DATA_DIR, name)

