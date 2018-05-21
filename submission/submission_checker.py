import os
import pandas as pd

# Modify the test directory where you put your test images
img_names = [x[:-4] for x in os.listdir('../test')]
# This is the submission csv file
sub = pd.read_csv('submission.csv')

correct_idx = []
print('Total test instance for submission: %d' % len(sub))
for i in range(len(sub)):
    # The correct flag indicates whether this row is correct, if set to False,
    # this row will simply be ignored from the modified submission
    correct_flag = True

    # check whether ImageId is correct
    img_n = sub.loc[i]['ImageId']
    if img_n not in img_names:
        print('Wrong im id : %s' % sub.loc[i]["ImageId"])
        correct_flag = False

    # check LabelIs is within the corpus
    label = sub.loc[i]["LabelId"]
    if label not in [33, 34, 35, 36, 38, 39, 40]:
        print("Wrong label for: %s" % sub.loc[i]["ImageId"])
        correct_flag = False

    # check your confidence is within the right range
    conf = sub.loc[i]['Confidence']
    if conf <= 0 or conf > 1:
        print("wrong confidence for: %s with conf: %.3f" % (sub.loc[i]["ImageId"], sub.loc[i]["Confidence"]))
        correct_flag = False

    # Check your pixel count is within the right range
    pc = sub.loc[i]['PixelCount']
    if pc <= 0 or pc >= 2710 * 3384:
        print("Wrong PC for: %s" % sub.loc[i]["ImageId"])
        correct_flag = False

    rle = sub.loc[i]['EncodedPixels']

    # This is the part where all my mistakes are made!!!! It could be the RLE encoding I used
    if rle[-1] != '|':
        print("Wrong RLE ending for: %s, with conf: %3.f, pixelCount: %d, last EP: %s" %
              (sub.loc[i]["ImageId"], sub.loc[i]['Confidence'], sub.loc[i]['PixelCount'], sub.loc[i]['EncodedPixels'][-10:]))
        correct_flag = False

    rle_pc = 0
    rle_split = rle.split('|')
    start_prev = -1
    for p in range(len(rle_split) - 1):
        start = int(rle_split[p].split(' ')[0])
        length = int(rle_split[p].split(' ')[1])
        if start <= start_prev:
            print("Wrong encoding for: %s" % sub.loc[i]["ImageId"])
            correct_flag = False

        start_prev = start + length

        if start < 0:
            print("Wrong start for: %s" % sub.loc[i]["ImageId"])
            correct_flag = False

        if start_prev >= 2710 * 3384:
            print("Wrong start_prev for: %s" % sub.loc[i]["ImageId"])
            correct_flag = False

        rle_pc += length

    if correct_flag:
        correct_idx.append(i)

    if i % 1000 == 0:
        print(i)
    if rle_pc != pc:
        print("Wrong: %d" % i)

sub_csv_new = sub.loc[correct_idx]
sub_csv_new.to_csv('correct_submission.csv', header=True, index=False)
