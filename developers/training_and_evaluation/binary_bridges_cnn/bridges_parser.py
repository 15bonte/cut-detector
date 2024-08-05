from cnn_framework.utils.parsers.transfer_parser import TransferParser


class BridgesParser(TransferParser):
    """
    Parsing class for bridges inference.
    """

    def __init__(self):
        super().__init__()

        self.arguments_parser.add_argument(
            "--hmm_bridges_parameters_file",
            help="Path to bridges HMM parameters file.",
        )
