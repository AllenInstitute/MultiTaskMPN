"""
Build a saved run's network, trained or as an untrained control.

The untrained controls of the multi-task analyses (`state_space_shift.py
--untrained`, `untrained_clustering.py`) need the same thing: a run's
architecture and config with fresh weights, drawn reproducibly from an init
seed, and none of the trained weights. `build_network` gives both that and
the trained network, so the analyses run the two through one code path.
"""
import numpy as np
import torch

import mpn


def build_network(checkpoint, untrained_init_seed=None):
    """The network to analyze and the state dict its weights are read from.

    By default the checkpoint's trained weights are loaded. With
    ``untrained_init_seed`` the same architecture and config
    (``checkpoint["net_params"]``) is initialized afresh after seeding
    NumPy's and torch's global generators with it, and no weights are loaded:
    an untrained network whose eta keeps its initial value. This is not the
    training run's own initialization, which was never saved and cannot be
    replayed, since training drew other random numbers before building its
    network.
    """
    if untrained_init_seed is not None:
        np.random.seed(untrained_init_seed)
        torch.manual_seed(untrained_init_seed)
    model = mpn.DeepMultiPlasticNet(checkpoint["net_params"], verbose=False, forzihan=True)
    if untrained_init_seed is None:
        missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=True)
        print("missing:", missing)
        print("unexpected:", unexpected)
        return model, checkpoint["state_dict"]
    return model, {name: value.detach().cpu() for name, value in model.state_dict().items()}
