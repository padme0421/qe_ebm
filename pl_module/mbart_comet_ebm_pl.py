from transformers import PreTrainedTokenizer

from pl_module.mbart_ebm_pl import MBARTSsl_EBMPL
from energy_model.comet_energy_model import COMET_EnergyModel
from custom_dataset import HuggingfaceDataModule

def prepare_energy_model(config, active_config):
    
    energy_model = COMET_EnergyModel(config, active_config)

    return energy_model


class MBARTSsl_COMETEBMPL(MBARTSsl_EBMPL):
    def __init__(self, active_config, config, device, 
                 tokenizer: PreTrainedTokenizer, datamodule: HuggingfaceDataModule,
                 by_steps: bool = False, warmup: bool = False):
        super().__init__(active_config, config, device, tokenizer, 
                         datamodule, prepare_energy_model, by_steps, warmup)
        