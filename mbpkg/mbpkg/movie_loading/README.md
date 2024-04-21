# movie_loading module

Movie Loading module wraps the mitosis movie loading operations.

It provides two main elements:
- a `load_movie` function that takes the path to a TIFF file and its format.
- a `Movie` object that can help with querying the underlying np array.

Other elements exist as helpers:
- `MovieFmt` wraps the movie format string to ensure it is valid. It is is not,
  a `MovieFmtError` is raised.
- `Source` wraps the Tuple (str, fmt) into an object. A source can be later
  loaded as a `Movie` (with `Source.load_movie`) or to a `np.ndarray` with
  `Source.load_data`