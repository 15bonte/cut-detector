"""Utility functions for annotations."""

NAMES_DICTIONARY = {
    "No membrane cut, double MT": 0,
    "No membrane cut, simple MT": 1,
    "No membrane cut, no MT": 2,
    "Membrane cut, simple MT": 3,
    "Membrane cut, no MT": 4,
    "No membrane cut, double MT ?": 5,
    "No membrane cut, simple MT ?": 6,
    "No membrane cut, no MT ?": 7,
    "Membrane cut, simple MT ?": 8,
    "Membrane cut, no MT ?": 9,
}


def get_class_ids_after_first_mt_cut() -> list[int]:
    """
    Get ids of classes after first MT cut.

    Returns
    -------
    list[int]
        List of ids.
    """
    return [
        NAMES_DICTIONARY["No membrane cut, simple MT"],
        NAMES_DICTIONARY["No membrane cut, no MT"],
        NAMES_DICTIONARY["Membrane cut, simple MT"],
        NAMES_DICTIONARY["Membrane cut, no MT"],
        NAMES_DICTIONARY["No membrane cut, simple MT ?"],
        NAMES_DICTIONARY["No membrane cut, no MT ?"],
        NAMES_DICTIONARY["Membrane cut, simple MT ?"],
        NAMES_DICTIONARY["Membrane cut, no MT ?"],
    ]


def get_class_ids_after_second_mt_cut() -> list[int]:
    """
    Get ids of classes after second MT cut.

    Returns
    -------
    list[int]
        List of ids.
    """
    return [
        NAMES_DICTIONARY["No membrane cut, no MT"],
        NAMES_DICTIONARY["Membrane cut, no MT"],
        NAMES_DICTIONARY["No membrane cut, no MT ?"],
        NAMES_DICTIONARY["Membrane cut, no MT ?"],
    ]


def get_class_ids_after_first_membrane_cut() -> list[int]:
    """
    Get ids of classes after first membrane cut.

    Returns
    -------
    list[int]
        List of ids.
    """
    return [
        NAMES_DICTIONARY["Membrane cut, simple MT"],
        NAMES_DICTIONARY["Membrane cut, no MT"],
        NAMES_DICTIONARY["Membrane cut, simple MT ?"],
        NAMES_DICTIONARY["Membrane cut, no MT ?"],
    ]
