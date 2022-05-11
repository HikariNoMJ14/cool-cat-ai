import sys
import logging

from src.dataset import MelodyDataset

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter('%(levelname)7s - %(message)s')

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

if __name__ == "__main__":
    d = MelodyDataset(
        sequence_size=48 * 4 + 1,
        encoding_type='duration',
        polyphonic=False,
        chord_encoding_type='extended',
        chord_extension_count=7,
        duration_correction=0,
        transpose_mode='none',
        logger=logger
    )

    d.create()

    # d.load()
