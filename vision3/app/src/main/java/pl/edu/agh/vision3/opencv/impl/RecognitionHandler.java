package pl.edu.agh.vision3.opencv.impl;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.nio.FloatBuffer;
import java.util.List;

import pl.edu.agh.vision3.tensorflow.TensorFlowUtils;
import pl.edu.agh.vision3.visual.IResultsComputedListener;
import pl.edu.agh.vision3.visual.VisualisationUtils;

import static pl.edu.agh.vision3.opencv.impl.ExtractionUtils.extractFace;
import static pl.edu.agh.vision3.opencv.impl.ExtractionUtils.extractMatFromInputFrame;
import static pl.edu.agh.vision3.opencv.impl.ExtractionUtils.filterOutEyes;
import static pl.edu.agh.vision3.tensorflow.TensorFlowUtils.INPUT_HEIGHT;
import static pl.edu.agh.vision3.tensorflow.TensorFlowUtils.INPUT_WIDTH;
import static pl.edu.agh.vision3.visual.VisualisationUtils.drawFaceMarker;

/**
 * Class for providing the recognition logic and executing the recognition flow.
 */
public class RecognitionHandler {

    /**
     * Runs recognition
     *
     * @param mFaceDetector face detection cascade
     * @param mEyeDetector eye detection cascade
     * @param inferenceInterface Tensorflow inference interface
     * @param resultsComputedListener listener for the partial results of recognition
     * @param inputFrame input frame with image to be processed
     *
     * @return input canvas with markers draw on it
     */
    public static Mat handle(
            CascadeClassifier mFaceDetector,
            CascadeClassifier mEyeDetector,
            TensorFlowInferenceInterface inferenceInterface,
            IResultsComputedListener resultsComputedListener,
            CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // extract face
        Mat matGray = extractMatFromInputFrame(inputFrame);
        Rect faceRec = detectMainFace(mFaceDetector, matGray);
        resultsComputedListener.onFaceRecognized(faceRec);

        if (faceRec != null) {
            drawFaceMarker(matGray, faceRec);
            Mat faceMat = extractFace(matGray, faceRec);

            // extract eyes and sight vectors
            handleSightRecognition(matGray, faceMat, faceRec, mEyeDetector, inferenceInterface, resultsComputedListener);
        }

        // rotate to match the output orientation
        Mat targetGray = new Mat();
        Core.rotate(matGray, targetGray, Core.ROTATE_180);
        return targetGray;
    }

    private static void handleSightRecognition(
            Mat matGray,
            Mat faceMat,
            Rect faceRec, CascadeClassifier mEyeDetector,
            TensorFlowInferenceInterface inferenceInterface,
            IResultsComputedListener resultsComputedListener) {


        // detect eyes
        final List<Rect> eyeRects = detectEyes(mEyeDetector, faceMat);

        FloatBuffer fb1 = null, fb2 = null;
        if (eyeRects != null) {
            // extract 2 biggest eyes from the recognized ones
            Rect[] eyes = filterOutEyes(faceRec, eyeRects);

            // recognize the vector for the first eye (biggest) and visualize it
            fb1 = recognizeVectorAndVisualize(
                    matGray, faceMat, faceRec, eyes[0], inferenceInterface);

            // recognize the vector for the second eye (second biggest) and visualize it
            if (eyes[1] != null) {
                fb2 = recognizeVectorAndVisualize(
                        matGray, faceMat, faceRec, eyes[1], inferenceInterface);
            }
        }

        // notify results listeners
        resultsComputedListener.onVectorsComputed(fb1, fb2);
    }

    private static List<Rect> detectEyes(CascadeClassifier mEyeDetector, Mat faceMat) {

        MatOfRect eyeMatOfRect = new MatOfRect();

        // detect eyes and save result to MatOfRect
        mEyeDetector.detectMultiScale(faceMat.t(), eyeMatOfRect);

        return eyeMatOfRect.toList();
    }

    private static Rect detectMainFace(CascadeClassifier mFaceDetector, Mat matGray) {
        MatOfRect matOfRect = new MatOfRect();

        // detect faces from prepared mat
        mFaceDetector.detectMultiScale(matGray.t(), matOfRect);
        List<Rect> rects = matOfRect.toList();

        // find the biggest one (dominating)
        return ExtractionUtils.filterOutMainFace(rects);
    }

    private static FloatBuffer recognizeVectorAndVisualize(Mat matGray, Mat faceMat, Rect faceRec, Rect eye, TensorFlowInferenceInterface inferenceInterface) {
        // run recognition
        FloatBuffer fb2 = runRecognition(faceMat, eye, inferenceInterface);

        // handle visualisation
        VisualisationUtils.drawEyeMarker(matGray, faceRec, eye);
        VisualisationUtils.drawVectors(matGray, faceRec, eye, fb2);

        return fb2;
    }

    private static FloatBuffer runRecognition(Mat inFaceMat, Rect eyeRec, TensorFlowInferenceInterface inferenceInterface) {

        // prepare mat with eye to match the expected orientation
        Mat faceMat = inFaceMat.t();
        Mat rawEyeMat = faceMat.submat(eyeRec);

        Mat eyeMat = new Mat();

        // resize to match the neural network input size
        Size size = new Size();
        size.width = INPUT_WIDTH;
        size.height = INPUT_HEIGHT;
        Imgproc.resize(rawEyeMat, eyeMat, size);

        // run recognition
        return TensorFlowUtils.runInference(inferenceInterface, eyeMat);
    }
}
