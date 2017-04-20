import unittest
import os

from constrained_decoding.translation_model.nematus_tm import NematusTranslationModel


class TestNematusTM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # A dict specifying all of the Nematus config options
        # Working: use the same test model assets that Nematus uses
        # Working: this requires training and adding some tiny test models
        # Working: see Nematus tests for training of a tiny model
        # Working: remember the small corpus creator in IMT repo

        # TODO: this temporarily uses the AmuNMT src-pe model, move to local tiny test model
        model_dir = '/media/1tb_drive/nematus_ape_experiments/amunmt_ape_pretrained/system/models/src-pe'
        model_file = os.path.join(model_dir, 'model.iter370000.npz')
        cls.model_file = model_file

        translate_config = {
            "return_alignment": False
        }

        # these are params which are output in the *.json file when a Nematus model is saved
        config = {
           "decoder": "gru_cond",
           "decay_c": 0.0,
           "patience": 10,
           "max_epochs": 5000,
           "dispFreq": 1000,
           "overwrite": False,
           "alpha_c": 0.0,
           "clip_c": 1.0,
           "n_words_src": 40000,
           "saveto": "model.npz",
           "valid_batch_size": 80,
           "n_words": 40000,
           "optimizer": "adadelta",
           "validFreq": 10000,
           "batch_size": 64,
           "encoder": "gru",
           "lrate": 0.0001,
           "valid_datasets": [
               "dev.bpe.en",
               "dev.bpe.de"
           ],
           "shuffle_each_epoch": True,
           "dim": 1024,
           "use_dropout": False,
           "datasets": [
               "corpus.bpe.en.shuf",
               "corpus.bpe.de.shuf"
           ],
           "dim_word": 500,
           "sampleFreq": 10000,
           "finetune": True,
           "dictionaries": [
               "vocab.src.json",
               "vocab.pe.json"
           ],
           "reload_": True,
           "maxlen": 50,
           "finish_after": 660000,
           "saveFreq": 30000
        }

        cls.config = dict(translate_config, **config)

    def test_initialization(self):
        test_tm = NematusTranslationModel([self.model_file], [self.config],
                                          model_weights=None)
        self.assertTrue(len(test_tm.fs_init) == len(test_tm.fs_next) == 1)

    def test_mapping_inputs(self):
        """Test that each of test_tm's models knows how to map inputs into internal representations"""

        pass



if __name__ == '__main__':
    unittest.main()
