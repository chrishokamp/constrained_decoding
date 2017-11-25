import logging
import os
import cPickle

from six import add_metaclass
from abc import ABCMeta, abstractmethod

import ahocorasick

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@add_metaclass(ABCMeta)
class AbstractSpotter(object):

    @abstractmethod
    def get_spots(self, text, **kwargs):
        pass

    @abstractmethod
    def load(self, basepath):
        """
        Load the spotter

        Args:
          path: path to persisted ahocorasick Automaton
        """
        pass

    @abstractmethod
    def persist(self, basepath):
        """
        Save the spotter

        Args:
          path: where the model should be persisted
        """
        pass


class MatchSpotter(AbstractSpotter):

    def __init__(self, rules=None):
        # set up the ahocorasick spotter for this user
        # support adding new spots, and persisting the user's spot dictionary to a file
        self.spotter = ahocorasick.Automaton()
        self.automaton_created = False

        if rules is not None:
            for r in rules:
                self.add_rule(r)

            self._make_automaton()

    def persist(self, basepath):
        # Note we use a default name for each model type
        model_path = os.path.join(basepath, 'match_spotter.pkl')
        if not self.automaton_created:
            self.make_automaton()

        with open(model_path, 'w') as out:
            cPickle.dump(self.spotter, out)
            logger.info('Persisted MatchSpotter to: {}'.format(model_path))

    def load(self, basepath):
        # Note we use a default name for each model type
        model_path = os.path.join(basepath, 'match_spotter.pkl')
        with open(model_path) as inp:
            cPickle.load(inp)
            self.automaton_created = True
            logger.info('Loaded MatchSpotter from: {}'.format(model_path))

    def add(self, surface_form):
        """
        Adds a new rule with a dummy URI

        Args:
          surface_form: unicode string for surface form
        """
        assert type(surface_form) is unicode, 'Match Spotter only accepts unicode strings'
        rule_obj = {
            'surface_form': surface_form,
            'uri': u'http://dbpedia.org/resource/' + surface_form
        }
        self.add_rule(rule_obj)

    def add_rule(self, rule_obj):
        # if user passed a string dynamically create rule obj
        if type(rule_obj) is unicode:
            rule_obj = dict([('surface_form', rule_obj), ('uri', None)])

        sf = rule_obj['surface_form']
        # workaround for lack of unicode support in pyahocorasick for python 2
        unicode_len = len(sf)
        sf = sf.encode("utf-8")
        # note that pyahocorasick supports adding any python object as the value for a Trie entry
        self.spotter.add_word(sf, rule_obj)
        self.automaton_created = False

    def _make_automaton(self):
        self.spotter.make_automaton()
        self.automaton_created = True

    def make_automaton(self):
        self._make_automaton()

    def get_spots(self, text):
        if not self.automaton_created:
            self._make_automaton()

        text = text.encode('utf-8')
        all_matches = []
        for end_idx, spot_obj in self.spotter.iter(text):
            # if decoding is expensive, the next line will be slow
            _content = text[:end_idx+1].decode("utf-8")
            sf = spot_obj['surface_form']
            uri = spot_obj['uri']

            e = len(_content)
            s = e - len(sf)
            # all_matches.append((s, e, sf, uri))
            all_matches.append((s, e))

        # filter to only capitalized
        # all_matches = [m for m in all_matches if m[2][0].isupper()]
        # extract the longest matches in each interval
        best_matches = get_longest_matches(all_matches)

        return best_matches


def get_longest_matches(all_matches):
    # The following logic extracts the longest match in each interval
    all_matches.sort(key=lambda x: x[1])
    all_matches.sort(key=lambda x: x[0])

    longest_matches = []

    current_start = 0
    current_end = 0
    current_longest = 0
    for m in all_matches:
        if current_end == 0:
            current_start = m[0]
            current_end = m[1]
            current_longest = m[1] - m[0]
            longest_matches.append(m)
        elif current_start <= m[0] < current_end:
            if m[1] - m[0] > current_longest:
                current_start = m[0]
                current_end = m[1]
                current_longest = m[1] - m[0]
                longest_matches[-1] = m
        else:
            current_start = m[0]
            current_end = m[1]
            current_longest = m[1] - m[0]
            longest_matches.append(m)

    return longest_matches
