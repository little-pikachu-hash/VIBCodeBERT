import random
import torch
import numpy as np
from log import logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.warning(f'Setting seed to {seed}')


def write_to_csv(scores, params, outputfile):
    """This function writes the parameters and the scores with their names in a
    csv file."""
    # creates the file if not existing.
    file = open(outputfile, 'w')
    # If file is empty writes the keys to the file.
    params_dict = vars(params)

    # Writes the configuration parameters
    for key in params_dict.keys():
        file.write(key + ",")
    for i, key in enumerate(scores[0].keys()):
        ending = "," if i < len(scores[0].keys()) - 1 else ""
        file.write(key + ending)
    file.write("\n")

    for row in scores:
        for key in params_dict:
            file.write(str(params_dict[key]).replace(',', ';') + ',')
        for key in scores[0].keys():
            file.write(str(row[key]).replace(',', ';') + ',')
        file.write("\n")
    file.close()
