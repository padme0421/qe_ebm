import s3fs
import os
from train_strategy.train_utils import move_checkpoint_to_s3, get_root_dir

s3_dir = s3fs.S3FileSystem()
config = {'dir_name': '/data/gahyunyoo/reinforce/7oog1fia/',
         'function': 'semisupervised_train_ebm',
          'model': 'mbart',
          'score': 'comet_kiwi'
          }

active_config = {
            'src': 'en',
            'trg': 'mr'
            }

root_dir = get_root_dir(active_config, config)
move_checkpoint_to_s3(config, root_dir)
