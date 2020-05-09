import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2

from object_detection.core import standard_fields
from object_detection.metrics import tf_example_parser
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vu
import matplotlib.ticker as ticker

import PIL
import IPython.display
from google.protobuf import text_format
import itertools
from object_detection.protos import string_int_label_map_pb2 as pb
from object_detection.data_decoders.tf_example_decoder import TfExampleDecoder as TfDecoder

import os
from object_detection.core import data_decoder
from object_detection.utils import visualization_utils as vis_util
from datetime import datetime

flags = tf.app.flags

flags.DEFINE_string('label_map', None, 'Path to the label map')
flags.DEFINE_string('detections_record', None, 'Path to the detections record file')
flags.DEFINE_string('output_path', None, 'Path to the output the results in a csv.')

FLAGS = flags.FLAGS

IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5

NUM_TO_SHOW = 100
FACTOR = 2
IMAGE_HEIGHT = 64*FACTOR
IMAGE_WIDTH = 128*FACTOR

category_index = 0

detectionDirName = datetime.now().strftime("DetectionImages_%Y-%m-%d_%H-%M")

def compute_iou(groundtruth_box, detection_box):
    g_ymin, g_xmin, g_ymax, g_xmax = tuple(groundtruth_box.tolist())
    d_ymin, d_xmin, d_ymax, d_xmax = tuple(detection_box.tolist())

    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

    boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
    boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

    return intersection / float(boxAArea + boxBArea - intersection)


def process_detections(detections_record, categories):
    record_iterator = tf.python_io.tf_record_iterator(path=detections_record)
    data_parser = tf_example_parser.TfExampleDetectionAndGTParser()


    confusion_matrix = np.zeros(shape=(len(categories) + 1, len(categories) + 1))

    num_shown = 0
    image_index = 0


    os.makedirs(detectionDirName, exist_ok=True)

    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        decoded_dict = data_parser.parse(example)


        image_index += 1

        if decoded_dict:
            groundtruth_boxes = decoded_dict[standard_fields.InputDataFields.groundtruth_boxes]
            groundtruth_classes = decoded_dict[standard_fields.InputDataFields.groundtruth_classes]

            detection_scores = decoded_dict[standard_fields.DetectionResultFields.detection_scores]
            detection_classes = decoded_dict[standard_fields.DetectionResultFields.detection_classes][detection_scores >= CONFIDENCE_THRESHOLD]
            detection_boxes = decoded_dict[standard_fields.DetectionResultFields.detection_boxes][detection_scores >= CONFIDENCE_THRESHOLD]

            if num_shown < NUM_TO_SHOW:

                # Convert encoded image in example TF Record to image
                testing = TfDecoder()
                features = testing.decode(string_record)
                image = features['image']
                with tf.Session() as sess:
                    image = image.eval()

                im = PIL.Image.fromarray(image)
                for box in groundtruth_boxes:
                    vis_util.draw_bounding_box_on_image(im, box[0]*IMAGE_HEIGHT, box[1]*IMAGE_WIDTH, box[2]*IMAGE_HEIGHT, box[3]*IMAGE_WIDTH, color='red', thickness=1, use_normalized_coordinates=False)
                for box in detection_boxes:
                    vis_util.draw_bounding_box_on_image(im, box[0]*IMAGE_HEIGHT, box[1]*IMAGE_WIDTH, box[2]*IMAGE_HEIGHT, box[3]*IMAGE_WIDTH, color='blue', thickness=1, use_normalized_coordinates=False)

                # UNCOMMENT TO DISPLAY IMAGES W/ BoundingBox
                #plt.imshow(np.asarray(im))
                #plt.show()

                # Code to create directory & save images w/ bounding boxes

                filename = decoded_dict['key']

                im.save(detectionDirName + "/" + filename)

                num_shown += 1


            matches = []


            if image_index % 100 == 0:
                print("Processed %d images" %(image_index))
            for i in range(len(groundtruth_boxes)):
                for j in range(len(detection_boxes)):
                    iou = compute_iou(groundtruth_boxes[i], detection_boxes[j])

                    if iou > IOU_THRESHOLD:
                        matches.append([i, j, iou])

            matches = np.array(matches)
            if matches.shape[0] > 0:
                # Sort list of matches by descending IOU so we can remove duplicate detections
                # while keeping the highest IOU entry.
                matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

                # Remove duplicate detections from the list.
                matches = matches[np.unique(matches[:,1], return_index=True)[1]]

                # Sort the list again by descending IOU. Removing duplicates doesn't preserve
                # our previous sort.
                matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

                # Remove duplicate ground truths from the list.
                matches = matches[np.unique(matches[:,0], return_index=True)[1]]

            for i in range(len(groundtruth_boxes)):
                if matches.shape[0] > 0 and matches[matches[:,0] == i].shape[0] == 1:
                    confusion_matrix[groundtruth_classes[i] - 1][detection_classes[int(matches[matches[:,0] == i, 1][0])] - 1] += 1
                else:
                    confusion_matrix[groundtruth_classes[i] - 1][confusion_matrix.shape[1] - 1] += 1

            for i in range(len(detection_boxes)):
                if matches.shape[0] > 0 and matches[matches[:,1] == i].shape[0] == 0:
                    confusion_matrix[confusion_matrix.shape[0] - 1][detection_classes[i] - 1] += 1
        else:
            print("Skipped image %d" % (image_index))

    print("Processed %d images" % (image_index))

    return confusion_matrix


def display(confusion_matrix, categories, output_path):
    print("\nConfusion Matrix:")
    print(confusion_matrix, "\n")
    results = []

    for i in range(len(categories)):
        id = categories[i]["id"] - 1
        name = categories[i]["name"]

        total_target = np.sum(confusion_matrix[id,:])
        total_predicted = np.sum(confusion_matrix[:,id])

        precision = float(confusion_matrix[id, id] / total_predicted)
        recall = float(confusion_matrix[id, id] / total_target)

        true_positive = confusion_matrix[id, id]

        # this is when it predicts the class but it is not the class
        false_positive = total_predicted - true_positive

        # this is when it should have predicted the class but dosen't
        false_negative = total_target - true_positive

        #print('precision_{}@{}IOU: {:.2f}'.format(name, IOU_THRESHOLD, precision))
        #print('recall_{}@{}IOU: {:.2f}'.format(name, IOU_THRESHOLD, recall))

        results.append({'category' : name, 'precision_@{}IOU'.format(IOU_THRESHOLD) : precision,'recall_@{}IOU'.format(IOU_THRESHOLD) : recall,'TP' : true_positive, 'FP' : false_positive, 'FN' : false_negative})

    df = pd.DataFrame(results)
    print(df)
    df.to_csv(output_path)


def main(argv):
    #matplotlib.use('tkagg')
    del argv
    required_flags = ['detections_record', 'label_map', 'output_path']
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise ValueError('Flag --{} is required'.format(flag_name))

    label_map = label_map_util.load_labelmap(FLAGS.label_map)

    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=100, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    confusion_matrix = process_detections(FLAGS.detections_record, categories)


    ids = range(len(categories))
    names = [0]*len(categories)
    for i in range(len(categories)):
        names[categories[i]["id"] -1 ]  = categories[i]["name"]

    fig = plt.figure(figsize=(18,16))
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_matrix, interpolation='nearest', cmap = plt.cm.Blues)
    plt.title("Confusion Matrix")
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    ax.set_xticklabels([' ']+names)
    ax.set_yticklabels([' ']+names)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig(detectionDirName + "/" +"ConfusionMatrix.png")

    display(confusion_matrix, categories, FLAGS.output_path)

if __name__ == '__main__':
    tf.app.run(main)
