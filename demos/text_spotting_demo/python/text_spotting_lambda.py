import sys
import boto3
import numpy as np
from openvino.runtime import Core, get_version
from PIL import Image
import json
import io

SOS_INDEX = 0
EOS_INDEX = 1
MAX_SEQ_LEN = 5


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def expand_box(box, scale):
    w_half = (box[2] - box[0]) * .5
    h_half = (box[3] - box[1]) * .5
    x_c = (box[2] + box[0]) * .5
    y_c = (box[3] + box[1]) * .5
    w_half *= scale
    h_half *= scale
    box_exp = np.zeros(box.shape)
    box_exp[0] = x_c - w_half
    box_exp[2] = x_c + w_half
    box_exp[1] = y_c - h_half
    box_exp[3] = y_c + h_half
    return box_exp


def segm_postprocess(box, raw_cls_mask, im_h, im_w):
    # Add zero border to prevent upsampling artifacts on segment borders.
    raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), 'constant', constant_values=0)
    extended_box = expand_box(box, raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0)).astype(int)
    w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
    x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
    x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])

    raw_cls_mask = Image.fromarray(raw_cls_mask)
    raw_cls_mask = raw_cls_mask.resize((w, h), resample=Image.BILINEAR)
    raw_cls_mask = np.array(raw_cls_mask) > 0.5
    mask = raw_cls_mask.astype(np.uint8)
    # Put an object mask in an image mask.
    im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
    im_mask[y0:y1, x0:x1] = mask[(y0 - extended_box[1]):(y1 - extended_box[1]),
                            (x0 - extended_box[0]):(x1 - extended_box[0])]
    return im_mask


# initialize everything we want to maintain across multiple calls
s3 = boto3.client('s3')
mask_rcnn_model = '/opt/text-spotting-0005-detector.xml'
text_enc_model = '/opt/text-spotting-0005-recognizer-encoder.xml'
text_dec_model = '/opt/text-spotting-0005-recognizer-decoder.xml'
device = 'CPU'
prob_threshold = 0.2
tr_threshold = 0.2
keep_aspect_ratio = True
limit = 1000

trd_input_prev_symbol = 'prev_symbol'
trd_input_prev_hidden = 'prev_hidden'
trd_input_encoder_outputs = 'encoder_outputs'
trd_output_symbols_distr = 'output'
trd_output_cur_hidden = 'hidden'
alphabet = '  abcdefghijklmnopqrstuvwxyz0123456789'

core = Core()

# Read IR
mask_rcnn_model = core.read_model(mask_rcnn_model)

input_tensor_name = 'image'

try:
    n, c, h, w = mask_rcnn_model.input(input_tensor_name).shape
    if n != 1:
        raise RuntimeError('Only batch 1 is supported by the demo application')
except RuntimeError:
    raise RuntimeError(
        'Demo supports only topologies with the following input tensor name: {}'.format(input_tensor_name))

required_output_names = {'boxes', 'labels', 'masks', 'text_features'}
for output_tensor_name in required_output_names:
    try:
        mask_rcnn_model.output(output_tensor_name)
    except RuntimeError:
        raise RuntimeError('Demo supports only topologies with the following output tensor names: {}'.format(
            ', '.join(required_output_names)))

text_enc_model = core.read_model(text_enc_model)

text_dec_model = core.read_model(text_dec_model)

mask_rcnn_compiled_model = core.compile_model(mask_rcnn_model, device_name=device)
mask_rcnn_infer_request = mask_rcnn_compiled_model.create_infer_request()

text_enc_compiled_model = core.compile_model(text_enc_model, device)
text_enc_output_tensor = text_enc_compiled_model.outputs[0]
text_enc_infer_request = text_enc_compiled_model.create_infer_request()

text_dec_compiled_model = core.compile_model(text_dec_model, device)
text_dec_infer_request = text_dec_compiled_model.create_infer_request()

hidden_shape = text_dec_model.input(trd_input_prev_hidden).shape
text_dec_output_names = {trd_output_symbols_distr, trd_output_cur_hidden}



def lambda_handler(event, context):
    s3Event = event['Records'][0]['s3']
    bucket = s3Event['bucket']['name']
    key = s3Event['object']['key']

    try:
        file_content = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
        image = Image.open(io.BytesIO(file_content))
    except Exception as e:
        print(f'could not get image from bucket: {e}')
        return {
            'statusCode': 400,
            'body': json.dumps(f'could not get image from bucket: {e}')
        }

    frame = image
    if not keep_aspect_ratio:
        # Resize the image to a target size.
        scale_x = w / frame.width
        scale_y = h / frame.height
        input_image = frame.resize((w, h))
    else:
        # Resize the image to keep the same aspect ratio and to fit it to a window of a target size.
        scale_x = scale_y = min(h / frame.height, w / frame.width)
        input_image = frame.resize((int(frame.width * scale_x), int(frame.height * scale_y)))

    input_image_size = (input_image.height, input_image.width)
    input_image = np.pad(input_image, ((0, h - input_image_size[0]),
                                       (0, w - input_image_size[1]),
                                       (0, 0)),
                         mode='constant', constant_values=0)
    # Change data layout from HWC to CHW.
    input_image = input_image.transpose((2, 0, 1))
    input_image = input_image.reshape((n, c, h, w)).astype(np.float32)

    # Run the MaskRCNN model.
    mask_rcnn_infer_request.infer({input_tensor_name: input_image})
    outputs = {name: mask_rcnn_infer_request.get_tensor(name).data[:] for name in required_output_names}

    # Parse detection results of the current request
    boxes = outputs['boxes'][:, :4]
    scores = outputs['boxes'][:, 4]
    classes = outputs['labels'].astype(np.uint32)
    raw_masks = outputs['masks']
    text_features = outputs['text_features']

    # Filter out detections with low confidence.
    detections_filter = scores > prob_threshold
    scores = scores[detections_filter]
    classes = classes[detections_filter]
    boxes = boxes[detections_filter]
    raw_masks = raw_masks[detections_filter]
    text_features = text_features[detections_filter]

    boxes[:, 0::2] /= scale_x
    boxes[:, 1::2] /= scale_y
    # masks = []
    # for box, cls, raw_mask in zip(boxes, classes, raw_masks):
    #     mask = segm_postprocess(box, raw_mask, frame.height, frame.width)
    #     masks.append(mask)

    texts = []
    for feature in text_features:
        input_data = {'input': np.expand_dims(feature, axis=0)}
        feature = text_enc_infer_request.infer(input_data)[text_enc_output_tensor]
        feature = np.reshape(feature, (feature.shape[0], feature.shape[1], -1))
        feature = np.transpose(feature, (0, 2, 1))

        hidden = np.zeros(hidden_shape)
        prev_symbol_index = np.ones((1,)) * SOS_INDEX

        text = ''
        text_confidence = 1.0
        for i in range(MAX_SEQ_LEN):
            text_dec_infer_request.infer({
                trd_input_prev_symbol: np.reshape(prev_symbol_index, (1,)),
                trd_input_prev_hidden: hidden,
                trd_input_encoder_outputs: feature})
            decoder_output = {name: text_dec_infer_request.get_tensor(name).data[:] for name in text_dec_output_names}
            symbols_distr = decoder_output[trd_output_symbols_distr]
            symbols_distr_softmaxed = softmax(symbols_distr, axis=1)[0]
            prev_symbol_index = int(np.argmax(symbols_distr, axis=1))
            text_confidence *= symbols_distr_softmaxed[prev_symbol_index]
            if prev_symbol_index == EOS_INDEX:
                break
            text += alphabet[prev_symbol_index]
            hidden = decoder_output[trd_output_cur_hidden]

        texts.append(text if text_confidence >= tr_threshold else '')

    detected_text = texts

    return {
        'statusCode': 200,
        'body': json.dumps(f'Finished Processing image, found these numbers: {detected_text}')
    }
