from src.dataset.dataset import Dataset


class MonoDataset(Dataset):

    @staticmethod
    def is_weakly_polyphonic(melody):
        melody['end_ticks'] = melody['ticks'] + melody['duration']

        return melody[
                   (melody['end_ticks'].shift(1) > melody['ticks']) |
                   (melody['end_ticks'] > melody['ticks'].shift(-1))
                   ][['ticks', 'end_ticks']].shape[0] > 0

    def remove_weak_polyphony(self, melody):
        mono_melody = melody.copy()

        overlap = (mono_melody['end_ticks'] - mono_melody['ticks'].shift(-1)).clip(0, None)

        mono_melody['duration'] -= overlap
        mono_melody['end_ticks'] -= overlap

        if self.is_weakly_polyphonic(mono_melody):
            raise Exception('Error!!! Weak polyphony not removed correctly')

        return mono_melody

    @staticmethod
    def is_strongly_polyphonic(melody):
        return melody['ticks'].shape[0] > melody['ticks'].nunique()

    def remove_strong_polyphony(self, melody):
        mono_melody = melody.copy()

        mono_melody.groupby('ticks').apply(lambda x: x['pitch'].max())

        if self.is_strongly_polyphonic(mono_melody):
            raise Exception('Error!!! Strong polyphony not removed correctly')

        return mono_melody
