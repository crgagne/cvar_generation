import datasets
from pathlib import Path
import numpy as np

class GenerationsConfig(datasets.BuilderConfig):
    """BuilderConfig for SMHD."""

    def __init__(self, data_filepath, n_rewards, **kwargs):
        super(GenerationsConfig, self).__init__(**kwargs)
        self.data_filepath = Path(data_filepath)
        self.n_rewards = n_rewards

class GenerationsDataset(datasets.GeneratorBasedBuilder):
    """SMHD - Self-Reported Mental Health Diagnoses."""

    BUILDER_CONFIG_CLASS = GenerationsConfig
    VERSION = datasets.Version("1.1.0")

    def _info(self):
        features = datasets.Features(
            {
                'text': datasets.Value('string'),
                'rewards': datasets.Sequence(datasets.Value('float32')),
                'summed_reward': datasets.Value('float32'),
                'composite_reward': datasets.Value('float32')
            }
        )

        return datasets.DatasetInfo(
            description="",
            features=features,
            supervised_keys=None,
            citation="",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'filepath': self.config.data_filepath,
                    'split': 'train',
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    'filepath': self.config.data_filepath,
                    'split': 'test'
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    'filepath': self.config.data_filepath,
                    'split': 'validation'
                },
            )
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples as (key, example) tuples."""

        # load data
        with open(filepath,'r') as f:
            lines = f.readlines()

        #spl_pt1=0.8
        #spl_pt2=0.9
        spl_pt1=1.
        spl_pt2=1.

        n_lines = len(lines)
        if split=='train':
            lines_to_iterate = range(0,int(n_lines*spl_pt1))
        elif split=='validation':
            lines_to_iterate = range(int(n_lines*spl_pt1), int(n_lines*spl_pt2))
        elif split=='test':
            lines_to_iterate = range(int(n_lines*spl_pt2), int(n_lines))

        for l in lines_to_iterate:
            line = lines[l].replace('\n','')
            text = line.split('p=')[0].strip()
            if 'r_all=' in line:
                composite_reward = float(line.split('r_all=')[1])
                line = sentences = line.split('r_all=')[0]
            else:
                composite_reward = None

            rewards = list(np.array(line.split('r=')[-1].split(',')).astype('float'))
            summed_reward = np.sum(rewards)
            if composite_reward is None:
                composite_reward = summed_reward
            if len(rewards)==self.config.n_rewards:
                yield l, {'text': text,
                          'rewards': rewards,
                          'summed_reward': summed_reward,
                          'composite_reward': composite_reward}
