'use strict';Object.defineProperty(exports,'__esModule',{value:true});/* globals tf, Image */

const IMAGESIZE = 400;

const computeTargetSize = function (width, height) {
  let resizeRatio = IMAGESIZE / Math.max(width, height);

  return {
    width: Math.round(resizeRatio * width),
    height: Math.round(resizeRatio * height)
  }
};

const getImageData = function (imageInput) {
  {
    return Promise.resolve(imageInput)
  }
};

const imageToTensor = function (imageData) {
  return tf.tidy(() => {
    const imgTensor = tf.browser.fromPixels(imageData);
    const targetSize = computeTargetSize(imgTensor.shape[0], imgTensor.shape[1]);
    return imgTensor.resizeBilinear([targetSize.width, targetSize.height]).expandDims()
  })
};

/**
 * convert image to Tensor input required by the model
 *
 * @param {HTMLImageElement} imageInput - the image element
 */
const preprocess = function (imageInput) {
  return getImageData(imageInput)
    .then(imageToTensor)
    .then(inputTensor => {
      return Promise.resolve(inputTensor)
    })
    .catch(err => {
      console.error(err);
      return Promise.reject(err)
    })
};/* globals tf */

// should be URL to hosted model assets (e.g., COS)
const modelPath = '/model_web/model.json';

let model = null;
let warmed = false;

/**
 * load the droneaid model
 */
const load = function (initialize) {
  if (!model) {
    // console.log('loading model...')
    // console.time('model load')
    return tf.loadGraphModel(modelPath)
      .then(m => {
        // console.timeEnd('model load')
        model = m;
        if (istrue(initialize)) {
          warmup();
        }
        return Promise.resolve(model)
      })
      .catch(err => {
        // console.timeEnd('model load')
        console.error(err);
        return Promise.reject(err)
      })
  } else if (istrue(initialize) && !warmed) {
    warmup();
    return Promise.resolve(model)
  } else {
    return Promise.resolve(model)
  }
};

/**
 * run the model to get a prediction
 */
const run = async function (imageTensor) {
  if (!imageTensor) {
    console.error('no image provided');
    throw new Error('no image provided')
  } else if (!model) {
    console.error('model not available');
    throw new Error('model not available')
  } else {
    // console.log('running model...')
    // console.time('model inference')
    // https://js.tensorflow.org/api/latest/#tf.GraphModel.executeAsync
    // console.time('model.execute')
    const results = await model.executeAsync({
      'image_tensor': imageTensor
    });
    // console.timeEnd('model.execute')
    // console.timeEnd('model inference')
    warmed = true;
    return results
  }
};

/**
 * run inference on the TensorFlow.js model
 */
const inference = function (imageTensor) {
  return load(false).then(() => {
    try {
      return run(imageTensor)
    } catch (err) {
      return Promise.reject(err)
    }
  })
};

const warmup = function () {
  try {
    run(tf.ones([1, 1024, 1024, 3]));
  } catch (err) { }
};

const istrue = function (param) {
  return param === null ||
    typeof param === 'undefined' ||
    (typeof param === 'string' && param.toLowerCase() === 'true') ||
    (typeof param === 'boolean' && param)
};const labels = [
  'children',
  'ok',
  'water',
  'firstaid',
  'sos',
  'shelter',
  'elderly',
  'food'
];/* global tf */

const maxNumBoxes = 10;
const iouThreshold = 0.5;
const scoreThreshold = 0.3;

const calculateMaxScores = (scores, numBoxes, numClasses) => {
  const maxes = [];
  const classes = [];
  for (let i = 0; i < numBoxes; i++) {
    let max = Number.MIN_VALUE;
    let index = -1;
    for (let j = 0; j < numClasses; j++) {
      if (scores[i * numClasses + j] > max) {
        max = scores[i * numClasses + j];
        index = j;
      }
    }
    maxes[i] = max;
    classes[i] = index;
  }
  return [maxes, classes]
};

const formatResponse = (indexes, boxes, classes, maxScores) => {
  const objects = [];

  for (let i = 0; i < indexes.length; i++) {
    const idx = indexes[i];
    const bbox = boxes[idx][0];

    objects.push({
      'class': classes[idx],
      'score': maxScores[idx],
      'bbox': bbox.map(b => Math.max(0, +(b.toFixed(4)))),
      'label': labels[classes[idx]]
    });
  }

  return objects
};

/**
 * convert model Tensor output
 *
 * @param {Tensor} inferenceResults - the output from running the model
 */
const postprocess = function (inferenceResults, options) {
  return new Promise(async (resolve, reject) => {
    // console.time('postprocess')
    let minScore = scoreThreshold;
    let iou = iouThreshold;
    let maxBoxes = maxNumBoxes;

    if (options) {
      minScore = options.scoreThreshold || scoreThreshold;
      iou = options.iouThreshold || iouThreshold;
      maxBoxes = options.maxNumBoxes || maxNumBoxes;
    }

    const scores = await inferenceResults[0].data();
    const boxes = inferenceResults[1].unstack()[0];

    tf.dispose(inferenceResults);

    const [maxScores, classes] = calculateMaxScores(scores, inferenceResults[0].shape[1], inferenceResults[0].shape[2]);
    const boxes2 = boxes.reshape([boxes.shape[0], boxes.shape[2]]);
    const boxes3 = await boxes.array();

    boxes.dispose();

    const indexTensor = await tf.image.nonMaxSuppressionAsync(
      boxes2,
      maxScores,
      maxBoxes,
      iou,
      minScore
    );

    const indexes = await indexTensor.data();

    boxes2.dispose();
    indexTensor.dispose();

    // console.timeEnd('postprocess')
    resolve(formatResponse(indexes, boxes3, classes, maxScores));
  })
};const version="0.0.1";{
  global.tf = require('@tensorflow/tfjs-node');
}

const processInput = function (inputImage, options) {
  options ? options.mirrorImage : false;
  return preprocess(inputImage)
};

const loadModel = function (init) {
  return load(init)
};

const runInference = function (inputTensor) {
  return inference(inputTensor)
};

const processOutput = function (inferenceResults, options) {
  return postprocess(inferenceResults, options)
};

const predict = function (inputImage, options) {
  return processInput(inputImage, options)
    .then(runInference)
    .then(outputTensor => {
      return processOutput(outputTensor, options)
    })
    .catch(err => {
      console.error(err);
    })
};exports.labels=labels;exports.loadModel=loadModel;exports.predict=predict;exports.processInput=processInput;exports.processOutput=processOutput;exports.runInference=runInference;exports.version=version;