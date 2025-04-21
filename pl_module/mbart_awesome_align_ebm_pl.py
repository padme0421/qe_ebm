from transformers import PreTrainedTokenizer

from pl_module.mbart_ebm_pl import MBARTSsl_EBMPL
from energy_model.awesome_align_energy_model import AwesomeAlign_EnergyModel
    

def prepare_energy_model(config, active_config):
    '''
    1) load energy model
    2) make embeddings linear
    3) configure adapters & param requires_grad attibutes
    4) register backward hooks
    '''

    energy_model = AwesomeAlign_EnergyModel(config, active_config)

    return energy_model


class MBARTSsl_AlignEBMPL(MBARTSsl_EBMPL):
    def __init__(self, active_config, config, device, tokenizer: PreTrainedTokenizer, 
                 by_steps: bool = False, warmup: bool = False):
        super().__init__(active_config, config, device, tokenizer, prepare_energy_model, by_steps, warmup)
