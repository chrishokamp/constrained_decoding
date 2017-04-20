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

        # params which are inputs to `nematus/translate.py`, but aren't part of the persisted *.json config
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
           "shuffle_each_epoch": True,
           "dim": 1024,
           "use_dropout": False,
           "dim_word": 500,
           "sampleFreq": 10000,
           "finetune": True,
           "dictionaries": [
               os.path.join(model_dir, "vocab.src.json"),
               os.path.join(model_dir, "vocab.pe.json")
           ],
           "reload_": True,
           "maxlen": 50,
           "finish_after": 660000,
           "saveFreq": 30000
        }

        cls.config = dict(translate_config, **config)
        cls.test_tm = NematusTranslationModel([cls.model_file], [cls.config],
                                              model_weights=None)

    def test_initialization(self):
        self.assertTrue(len(self.test_tm.fs_init) == len(self.test_tm.fs_next) == 1)

    def test_load_dictionaries(self):
        dict_keys = ['input_dicts', 'input_idicts', 'output_dict', 'output_idict']
        for dict_map in self.test_tm.word_dicts:
            self.assertTrue(all(k in dict_map.keys() for k in dict_keys))
            self.assertTrue(type(dict_map['input_dicts'] is list))
            self.assertTrue(type(dict_map['input_idicts'] is list))
            self.assertTrue(type(dict_map['output_dict'] is dict))
            self.assertTrue(type(dict_map['output_idict'] is dict))

    def test_mapping_inputs(self):
        """Test that each of the tm's models knows how to map inputs into internal representations"""

        test_input = u'A sample English sentence'
        mapped_input = self.test_tm.map_inputs([test_input])

        self.assertTrue(len(mapped_input) == 1)
        self.assertTrue(mapped_input[0].shape == (1, 5, 1))

    def test_start_hypothesis(self):
        test_input = u'A sample English sentence'
        mapped_input = self.test_tm.map_inputs([test_input])
        constraint_1 = u'Ein langweiliger'.split()
        test_constraints = self.test_tm.map_constraints([constraint_1])

        start_hyp = self.test_tm.start_hypothesis(mapped_input, test_constraints)

        self.assertTrue('next_states' in start_hyp.payload.keys())
        self.assertTrue(len(start_hyp.payload['next_states']) == 1)

    def test_generate(self):
        test_input = u'a sample English sentence'
        mapped_input = self.test_tm.map_inputs([test_input])
        constraint_1 = u'Ein langweiliger'.split()
        test_constraints = self.test_tm.map_constraints([constraint_1])
        start_hyp = self.test_tm.start_hypothesis(mapped_input, test_constraints)

        nbest = 5
        next_hyps = self.test_tm.generate(start_hyp, nbest)
        self.assertTrue(len(next_hyps) == nbest)
        # assert each hyp generated a different symbol
        self.assertTrue(len(set(hyp.token for hyp in next_hyps)) == nbest)

    def test_generate_constrained(self):
        test_input = u'A sample English sentence'
        mapped_input = self.test_tm.map_inputs([test_input])
        constraint_1 = u'Ein langweiliger'.split()
        test_constraints = self.test_tm.map_constraints([constraint_1])
        start_hyp = self.test_tm.start_hypothesis(mapped_input, test_constraints)

        next_hyps = self.test_tm.generate_constrained(start_hyp)
        self.assertTrue(len(next_hyps) == 1)
        self.assertTrue(next_hyps[0].token == constraint_1[0])

    def test_continue_constrained(self):
        test_input = u'A sample English sentence'
        mapped_input = self.test_tm.map_inputs([test_input])
        constraint_1 = u'ein langer'.split()
        test_constraints = self.test_tm.map_constraints([constraint_1])
        start_hyp = self.test_tm.start_hypothesis(mapped_input, test_constraints)

        next_hyps = self.test_tm.generate_constrained(start_hyp)
        continued_hyp = self.test_tm.continue_constrained(next_hyps[0])
        self.assertTrue(continued_hyp.token == constraint_1[1])
        self.assertFalse(continued_hyp.unfinished_constraint)


if __name__ == '__main__':
    unittest.main()
