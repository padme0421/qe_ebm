import torch
from torch import nn, Tensor

class EnergyModel(nn.Module):
    def __init__(self, config, active_config):
        super(EnergyModel, self).__init__()
        self.model = None
        self.tokenizer = None
        self.base_model: nn.Module = None
    
    def set_optimizers(self, optim):
        self.optim = optim
    
    @torch.autocast("cuda")
    def forward(self, src_input_ids: Tensor, mt_input_logits: Tensor, mt_input_ids: Tensor) -> Tensor:
        """Forward function.

        Args:
            src_input_ids: 
            X [SEP] [PAD], size = [bsz, src_length]

            mt_input_logits:
            Y [SEP] [PAD], size = [bsz, mt_length, vocab_size]

            mt_input_ids: 
            Y [SEP] [PAD], size  = [bsz, mt_length]            

        Raises:
            Exception: Invalid model word/sent layer if self.{word/sent}_layer are not
                valid encoder model layers .

        Returns:
            Tensor
        """

        raise NotImplemented()

    def set_params_grad(self, mode: bool):
        raise NotImplemented()
    
    def parameters(self):
        return self.base_model.parameters()

    def zero_grad(self, set_to_none: bool = True):
        self.base_model.zero_grad(set_to_none)
    
    def optimizers(self):
        return self.optim