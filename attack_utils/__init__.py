from attack_utils.adv_loss import *

__all__ = {
    'AttentiveFusion': attFusionLoss,
    'CoAlign': coalignLoss,
    'Where2comm': where2commLoss,
    'V2VAM': v2vamLoss,
}

def get_adv_loss(model_name):
    assert model_name in __all__.keys()
    return __all__[model_name]