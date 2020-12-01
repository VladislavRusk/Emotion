/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package edu.ilab.cs663.classification.tflite;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.SystemClock;
import android.os.Trace;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import edu.ilab.cs663.classification.env.Logger;

import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/** A classifier specialized to label images using TensorFlow Lite.
 *
 *    NOTE 2: Classifier contains most of the complex logic for processing the camera input and running inference.
 *
 *       A subclasses of the file exist, in ClassifierFloatMobileNet.java (in other Tensorflowlite examples there is ClassifierQuantizedMobileNet.java), to demonstrate the use of
 *       floating point (and quantized) models.
 *
 *       The Classifier class implements a static method, create, which is used to instantiate the appropriate subclass based on the supplied model type (quantized vs floating point).
 *
 *
 */

public abstract class Classifier {
  private static final Logger LOGGER = new Logger();

  /** The runtime device type used for executing classification. */
  public enum Device {
    CPU,
    GPU
  }


  /** Number of results to show in the UI. */
  private static final int MAX_RESULTS = 2;

  /** The loaded TensorFlow Lite model. */
  private MappedByteBuffer featureExtractorModel;
  private MappedByteBuffer lstmModel;

  /** Image size along the x axis. */
  private final int imageSizeX;
  private final int imageSizeX2;

  /** Image size along the y axis. */
  private final int imageSizeY;
  private final int imageSizeY2;
  /** Optional GPU delegate for acceleration. */
  private GpuDelegate gpuDelegate = null;


  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  protected Interpreter featureTflite;
  protected Interpreter lstmTflite;


  /** Options for configuring the Interpreter. */
  private final Interpreter.Options tfliteOptions = new Interpreter.Options();

  /** Labels corresponding to the output of the vision model. */
  private List<String> labels;

  /** Input image TensorBuffer. */
  private TensorImage inputImageBuffer;
  public float[][][] lstmInput = new float[1][1][1024]; // LSTM INPUT VERY IMPORTANT
  public int frame = 0; // FRAME NUMBER
  /** Output probability TensorBuffer. */
  private final TensorBuffer outputProbabilityBuffer;
  private final TensorBuffer outputProbabilityBuffer2;

  /** Processer to apply post processing of the output probability. */
  private final TensorProcessor probabilityProcessor2;

  /**
   * Creates a classifier with the provided configuration.
   *
   * @param activity The current Activity.
   * @param device The device to use for classification.
   * @param numThreads The number of threads to use for classification.
   * @return A classifier with the desired configuration.
   */
  public static Classifier create(Activity activity, Device device, int numThreads)
      throws IOException {

    return new ClassifierFloatMobileNet(activity, device, numThreads);
  }

  /** An immutable result returned by a Classifier describing what was recognized. */
  public static class Recognition {
    /**
     * A unique identifier for what has been recognized. Specific to the class, not the instance of
     * the object.
     */
    private final String id;

    /** Display name for the recognition. */
    private final String title;

    /**
     * A sortable score for how good the recognition is relative to others. Higher should be better.
     */
    private final Float confidence;

    /** Optional location within the source image for the location of the recognized object. */
    private RectF location;

    public Recognition(
            final String id, final String title, final Float confidence, final RectF location) {
      this.id = id;
      this.title = title;
      this.confidence = confidence;
      this.location = location;
    }

    public String getId() {
      return id;
    }

    public String getTitle() {
      return title;
    }

    public Float getConfidence() {
      return confidence;
    }

    public RectF getLocation() {
      return new RectF(location);
    }

    public void setLocation(RectF location) {
      this.location = location;
    }

    @Override
    public String toString() {
      String resultString = "";
      if (id != null) {
        resultString += "[" + id + "] ";
      }

      if (title != null) {
        resultString += title + " ";
      }

      if (confidence != null) {
        resultString += String.format("(%.1f%%) ", confidence * 100.0f);
      }

      if (location != null) {
        resultString += location + " ";
      }

      return resultString.trim();
    }
  }

  /** Initializes a {@code Classifier}.
   *
   *
   * To perform inference, we need to load a model file and instantiate an Interpreter.
   * This happens in the constructor of the Classifier class, along with loading the list of class labels.
   * Information about the device type and number of threads is used to configure the Interpreter via the
   * Interpreter.Options instance passed into its constructor. Note how that in the case of a GPU being
   * available, a Delegate is created using GpuDelegateHelper.
   *
   * */
  protected Classifier(Activity activity, Device device, int numThreads) throws IOException {
    featureExtractorModel = FileUtil.loadMappedFile(activity, getModelPath());
    switch (device) {
      case GPU:
        //create a GPU delegate instance and add it to the interpreter options
        gpuDelegate = new GpuDelegate();
        tfliteOptions.addDelegate(gpuDelegate);

        break;
      case CPU:
        break;
    }
    tfliteOptions.setNumThreads(numThreads);

    // Create a TFLite interpreter instance
    featureTflite = new Interpreter(featureExtractorModel, tfliteOptions);

    // Loads labels out from the label file.
//    labels = FileUtil.loadLabels(activity, getLabelPath()); // Igonore Possibly?????

    // Reads type and shape of input and output tensors, respectively.
    //Determine the necessary input image size from the first layer imageShape.
    // get shape and data type of model lines 223-229
    int imageTensorIndex = 0;
    int[] imageShape = featureTflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3} get input tensor

    imageSizeY = imageShape[1];
    imageSizeX = imageShape[2];

    DataType imageDataType = featureTflite.getInputTensor(imageTensorIndex).dataType();


    int probabilityTensorIndex = 0;
    int[] probabilityShape =
            featureTflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, 2048} the shape of output
    DataType probabilityDataType = featureTflite.getOutputTensor(probabilityTensorIndex).dataType(); // datatype

    // Creates the input tensor.
    inputImageBuffer = new TensorImage(imageDataType);

    // Creates the output tensor and its processor.
    outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

    // Creates the post processor for the output probability.
//    probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();
//*************************************************************************************************
//*************************************************************************************************
// **********************************CREATE LSTM MODEL HERE ***************************************
//*************************************************************************************************
//*************************************************************************************************
//*************************************************************************************************
    lstmModel = FileUtil.loadMappedFile(activity, "LModel2.tflite");
    tfliteOptions.setNumThreads(numThreads);

    // Create a TFLite interpreter instance
    lstmTflite = new Interpreter(lstmModel, tfliteOptions);

    // Loads labels out from the label file.
    labels = FileUtil.loadLabels(activity, getLabelPath()); // Igonore Possibly?????

    // Reads type and shape of input and output tensors, respectively.
    //Determine the necessary input image size from the first layer imageShape.
    // get shape and data type of model lines 223-229
    int imageTensorIndex2 = 0;
    int[] imageShape2 = lstmTflite.getInputTensor(imageTensorIndex2).shape(); // {10, 1, width, 3} get input tensor

    imageSizeY2 = imageShape2[1];
    imageSizeX2 = imageShape2[2];

    DataType imageDataType2 = lstmTflite.getInputTensor(imageTensorIndex2).dataType();


    int probabilityTensorIndex2 = 0;
    int[] probabilityShape2 =
            lstmTflite.getOutputTensor(probabilityTensorIndex2).shape(); // {1, 1024} the shape of output
    DataType probabilityDataType2 = lstmTflite.getOutputTensor(probabilityTensorIndex2).dataType(); // datatype

    // Creates the input tensor.
//        inputImageBuffer = new float(imageDataType);

    // Creates the output tensor and its processor.
    outputProbabilityBuffer2 = TensorBuffer.createFixedSize(probabilityShape2, probabilityDataType2);

    // Creates the post processor for the output probability.
    probabilityProcessor2 = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();






    LOGGER.d("Created a Tensorflow Lite Image Classifier.");




  }

  public List<Recognition> recognizeImageLSTM() {
    // Logs this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("loadImage");
    long startTimeForLoadImage = SystemClock.uptimeMillis();
//        inputImageBuffer = loadImage(bitmap, sensorOrientation);
    long endTimeForLoadImage = SystemClock.uptimeMillis();
    Trace.endSection();
    LOGGER.v("Timecost to load the image: " + (endTimeForLoadImage - startTimeForLoadImage));

    // Runs the inference call.
    Trace.beginSection("runInference");
    long startTimeForReference = SystemClock.uptimeMillis();
    float[][] test = new float[1][7];
    // Run TFLite inference passing in the processed image.
    lstmTflite.run(lstmInput, test);
    long endTimeForReference = SystemClock.uptimeMillis();
    Trace.endSection();
    LOGGER.v("Timecost to run model inference: " + (endTimeForReference - startTimeForReference));

    // Gets the map of label and probability.
    // Use TensorLabel from TFLite Support Library to associate the probabilities
    //       with category labels
    //labeledProbability is the object that maps each label to its probability.
    //     The TensorFlow Lite Support Library provides a convenient utility to convert from the model
    //     output to a human-readable probability map. We later use the getTopKProbability(..) method to
    //     extract the top-K most probable labels from labeledProbability.
    Map<String, Float> labeledProbability = makeProb(test[0]);
//            new TensorLabel(labels, test).getMapWithFloatValue();

    Trace.endSection();

    // Gets top-k results.
    return getTopKProbability(labeledProbability);
  }

  public Map<String, Float> makeProb(float[] results) {
    Map<String,Float> temp = new HashMap<String, Float>();
    String[] theLabels = {"disgust", "fear", "happiness", "others", "repression", "sadness", "surprise"};
    for (int i = 0; i < 7; i++) {
      temp.put(theLabels[i], results[i]);
    }
    return temp;
  }



  /** Runs inference and returns the classification results. */
  public List<Recognition> recognizeImage(final Bitmap bitmap, int sensorOrientation) {
      // Logs this method so that it can be analyzed with systrace.
      Trace.beginSection("recognizeImage");

      Trace.beginSection("loadImage");
      long startTimeForLoadImage = SystemClock.uptimeMillis();
      inputImageBuffer = loadImage(bitmap, sensorOrientation);
      long endTimeForLoadImage = SystemClock.uptimeMillis();
      Trace.endSection();
      LOGGER.v("Timecost to load the image: " + (endTimeForLoadImage - startTimeForLoadImage));

      // Runs the inference call.
      Trace.beginSection("runInference");
      long startTimeForReference = SystemClock.uptimeMillis();
      //    float[][][] test = new float[1][1][1024];
      // Run TFLite inference passing in the processed image.
      featureTflite.run(inputImageBuffer.getBuffer(), lstmInput[0]);

      long endTimeForReference = SystemClock.uptimeMillis();
      Trace.endSection();
      LOGGER.v("Timecost to run model inference: " + (endTimeForReference - startTimeForReference));

      // Gets the map of label and probability.
      // Use TensorLabel from TFLite Support Library to associate the probabilities
      //       with category labels
      //labeledProbability is the object that maps each label to its probability.
      //     The TensorFlow Lite Support Library provides a convenient utility to convert from the model
      //     output to a human-readable probability map. We later use the getTopKProbability(..) method to
      //     extract the top-K most probable labels from labeledProbability.

      Trace.endSection();

      frame += 1;
      return recognizeImageLSTM();

    // Gets top-k results.
//    return test[0];
  }

  /** Closes the interpreter and model to release resources. */
  public void close() {
    if (featureTflite != null) {
      // Close the interpreter
      featureTflite.close();
      featureTflite = null;


    }
    if (lstmTflite != null) {
      lstmTflite.close();
      lstmTflite = null;

    }
    // Close the GPU delegate
    if (gpuDelegate != null) {
      gpuDelegate.close();
      gpuDelegate = null;
    }


    featureExtractorModel = null;
    lstmModel = null;
  }

  /** Get the image size along the x axis. */
  public int getImageSizeX() {
    return imageSizeX;
  }

  /** Get the image size along the y axis. */
  public int getImageSizeY() {
    return imageSizeY;
  }

  /** Loads input image, and applies preprocessing. */
  private TensorImage loadImage(final Bitmap bitmap, int sensorOrientation) {
    // Loads bitmap into a TensorImage.
    inputImageBuffer.load(bitmap);

    // Creates processor for the TensorImage.
    int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
    int numRoration = sensorOrientation / 90;
    // Define an ImageProcessor from TFLite Support Library to do preprocessing
    /* Basic image processor
    ImageProcessor imageProcessor =
            new ImageProcessor.Builder()




                .build();
    return imageProcessor.process(inputImageBuffer);*/

    //Image processor that resizes to cropSizeXcropSize or to imageSizeX X imageSizeY, and that can rotate 90 decrees, and
    // perform normalization on the image (basic filtering)
    //THIS LOOKS CORRECT - based on tests
    ImageProcessor imageProcessor =
            new ImageProcessor.Builder()
                    .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                    .add(new ResizeOp(imageSizeX, imageSizeY, ResizeMethod.NEAREST_NEIGHBOR))
                    .add(new Rot90Op(numRoration))
                    .add(getPreprocessNormalizeOp())
                    .build();
    return imageProcessor.process(inputImageBuffer);
  }

  /** Gets the top-k results. */
  static List<Recognition> getTopKProbability(Map<String, Float> labelProb) {
    // Find the best classifications.
    PriorityQueue<Recognition> pq =
        new PriorityQueue<>(
            MAX_RESULTS,
            new Comparator<Recognition>() {
              @Override
              public int compare(Recognition lhs, Recognition rhs) {
                // Intentionally reversed to put high confidence at the head of the queue.
                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
              }
            });

    for (Map.Entry<String, Float> entry : labelProb.entrySet()) {
      pq.add(new Recognition("" + entry.getKey(), entry.getKey(), entry.getValue(), null));
    }

    final ArrayList<Recognition> recognitions = new ArrayList<>();
    int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
    for (int i = 0; i <recognitionsSize; ++i) {
      recognitions.add(pq.poll());
    }
    return recognitions;
  }

  /** Gets the name of the model file stored in Assets. */
  protected abstract String getModelPath();

  /** Gets the name of the label file stored in Assets. */
  protected abstract String getLabelPath();

  /** Gets the TensorOperator to nomalize the input image in preprocessing. */
  protected abstract TensorOperator getPreprocessNormalizeOp();

  /**
   * Gets the TensorOperator to dequantize the output probability in post processing.
   *
   * <p>For quantized model, we need de-quantize the prediction with NormalizeOp (as they are all
   * essentially linear transformation). For float model, de-quantize is not required. But to
   * uniform the API, de-quantize is added to float model too. Mean and std are set to 0.0f and
   * 1.0f, respectively.
   */
  protected abstract TensorOperator getPostprocessNormalizeOp();
}