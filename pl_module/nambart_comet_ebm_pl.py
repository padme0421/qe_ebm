from transformers import PreTrainedTokenizer

from pl_module.nambart_ebm_pl import NAMBARTSsl_EBMPL
from energy_model.comet_energy_model import COMET_EnergyModel

def prepare_energy_model(config, active_config):
    
    energy_model = COMET_EnergyModel(config, active_config)

    return energy_model


class NAMBARTSsl_COMETEBMPL(NAMBARTSsl_EBMPL):
    def __init__(self, active_config, config, device, tokenizer: PreTrainedTokenizer, 
                 by_steps: bool = False, warmup: bool = False):
        super().__init__(active_config, config, device, tokenizer, prepare_energy_model, by_steps, warmup)
    
