from mido import MidiFile
import pandas as pd

'''
WAV <-> OGG
import soundfile as sf

data, samplerate = sf.read('guitar.ogg')
sf.write('cultofpersonalityguitar.wav', data, samplerate)

exit()
'''

'''
WAV -> Tensor
from docarray import Document
wav = Document(uri=r'cultofpersonalityguitar.wav').load_uri_to_audio_tensor()

print(wav.tensor.shape, wav.tensor.dtype)
print(wav.tensor)

exit()
'''

mid = MidiFile('cultofpersonality.mid', clip=True)

expertKeys = {
    '96': 'GREEN',
    '97': 'RED',
    '98': 'YELLOW',
    '99': 'BLUE',
    '100': 'ORANGE',
    '101': 'HAMMER PREV',
    '102': 'NORMAL PREV',
    '103': 'STARPOWER'
}

df = pd.DataFrame(columns=['Note', 'On', 'Time'])
for track in mid.tracks:
    temp = []
    if track.name == 'PART GUITAR':
        for msg in track:
            if msg.is_meta is False and 96 <= msg.note <= 103:
                temp += [[msg.note, msg.velocity == 100, msg.time]]
        df = pd.DataFrame(temp, columns=['Note', 'On', 'Time'])
print(df)

for index, row in df.iterrows():
    if row['On'] == True:
        print(expertKeys[str(row['Note'])])

'''

96: guitar note GREEN, expert (C)
97: guitar note RED, expert (C#)
98: guitar note YELLOW, expert (D)
99: guitar note BLUE, expert (D#)
100: guitar note ORANGE, expert (E)
103: star power group, expert (G)
'''

#for track in mid2.tracks:
#    print(track)

