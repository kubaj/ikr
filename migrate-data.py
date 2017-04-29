import glob
import os

GLOB_FACES = '*.png'
GLOB_SPEECH = '*.wav'
BASE_DIR = 'data/'
TRAIN_DIR = BASE_DIR + 'train/'
DEV_DIR = BASE_DIR + 'dev/'

mov = 0

for i in range(1, 32):
    filenames_new = [f for f in glob.glob('{}{}/{}'.format(DEV_DIR, i, GLOB_FACES)) if os.path.isfile(f)]
    victim = filenames_new[0].replace('.png', '')
    print('Moving', victim, victim.replace('dev', 'train'))
    os.rename(victim+'.png', victim.replace('dev', 'train')+'.png')
    os.rename(victim+'.wav', victim.replace('dev', 'train')+'.wav')
    mov += 2

print('Moved', mov, 'dev files to train')
