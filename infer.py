import argparse
import glob
import numpy as np
from keras.models import load_model
import ikr-data

CLASSES = 31


def load_model_or_none(from_arg, name):
    if from_arg:
        print("Loading {} model from {}...".format(name, from_arg))
        model = load_model(from_arg)
        print("OK")
        return model
    else:
        print("No {} model used.".format(name))
        return None


def get_inputs(evaldir):
    files = [f.replace('.png', '') for f in glob.glob('{}/*.png'.format(evaldir))]
    return files

parser = argparse.ArgumentParser()
parser.add_argument('evaldir', type=str, help='directory with evaluation data')
parser.add_argument('--model-speech', type=str, help='h5 model for speech net', dest='speech')
parser.add_argument('--model-face', type=str, help='h5 model for face net', dest='face')
parser.add_argument('-o', '--output', type=str, help='output file', dest='out')
parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', help='print detailed outputs on stdout')
args = parser.parse_args()

# Load models
model_face = load_model_or_none(args.face, 'facenet')
model_speech = load_model_or_none(args.speech, 'speechmap')

# report generation
report_file = None
if args.out:
    report_file = open(args.out, 'w')

# Process files
for file in get_inputs(args.evaldir):
    print('-'*80)
    print('Inferring input', file)
    facenet_result = np.ones([CLASSES])
    if model_face:
        # Process facenet
        imagedata = np.array([ikrdata.open_image(file+'.png')])
        imagedata /= 255
        facenet_result = model_face.predict(imagedata)[0]

        if args.verbose:
            print('Facenet:\n', facenet_result)

    speechnet_result = np.ones([CLASSES])
    if model_speech:
        # Process speechnet
        audiodata = np.array(ikrdata.open_audio(file+'.wav'))
        mapcount = len(audiodata)

        if args.verbose:
            print('Speechmap is predicting from {} maps'.format(mapcount))

        # get counts from predictions
        counts = np.zeros([CLASSES], np.int32)
        pred = model_speech.predict(audiodata)
        for p in pred:
            amax = np.argmax(p)
            counts[amax] += 1

        # get probabilites
        speechnet_result = counts / mapcount

        if args.verbose:
            print('Speechnet:\n', speechnet_result)

    result = facenet_result*speechnet_result
    hard = np.argmax(result) + 1

    if args.verbose:
        print("Total:\n", result)
        print('Hard decision:', hard)
        print()

    if report_file:
        file_name = file[file.rfind('/')+1:]
        report_file.write(file_name + ' ' + str(hard) + ' ')
        report_file.write(' '.join(str(v) for v in result))
        report_file.write('\n')

if report_file:
    report_file.close()
    print("Report written to", args.out)
