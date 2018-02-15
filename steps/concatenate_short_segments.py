import itertools
import sys

src = sys.argv[1]
dest = sys.argv[2]
silence_threshold = float(sys.argv[3])
segment_length_threshold = float(sys.argv[4])

utt2spk = {}
with open('%s/utt2spk' % src, 'r') as f:
    for line in f:
        (utt, spk) = line.strip().split()
        utt2spk[utt] = spk

utt2text = {}
with open('%s/text' % src, 'r') as f:
    for line in f:
        if " " in line.strip():
            (utt, text) = line.strip().split(" ", 1)
            utt2text[utt] = text
        else:
            utt2text[line.strip()] = text
            

with open('%s/segments' % src, 'r') as segments_in, \
    open('%s/segments' % dest, 'w') as segments_out, \
    open('%s/text' % dest, 'w') as text_out, \
    open('%s/utt2spk' % dest, 'w') as utt2spk_out:

    last_utt = None
    last_wav = None
    last_start = None
    last_end = None
    last_spk = None
    last_text = []

    for line in segments_in:
        (utt, wav, start, end) = line.strip().split()
        start = float(start)
        end = float(end)

        if last_wav != wav or start - last_end > silence_threshold:
            if last_end is not None and end - last_start >= segment_length_threshold:
                print >> segments_out, last_utt, last_wav, last_start, last_end
                print >> text_out, last_utt, " ".join(last_text)
                print >> utt2spk_out, last_utt, last_spk

            last_utt = utt
            last_wav = wav
            last_start = start
            last_end = end
            last_spk = utt2spk[utt]
            last_text = [utt2text[utt]]
        else:
            last_end = end
            last_text.append(utt2text[utt])

    if last_end is not None:
        print >> segments_out, last_utt, last_wav, last_start, last_end
        print >> text_out, last_utt, " ".join(last_text)
        print >> utt2spk_out, last_utt, last_spk
