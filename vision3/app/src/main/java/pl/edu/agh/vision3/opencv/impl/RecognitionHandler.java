package pl.edu.agh.vision3.opencv.impl;

import android.util.Log;

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

public class RecognitionHandler {

    private static final String TAG = "Recognition Handler";

    public static Mat handle(
            CascadeClassifier mFaceDetector,
            CascadeClassifier mEyeDetector,
            TensorFlowInferenceInterface inferenceInterface,
            IResultsComputedListener resultsComputedListener,
            CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        Mat verticallyFlippedMatGray = inputFrame.gray();
        Mat matGray = new Mat();
        Core.rotate(verticallyFlippedMatGray, matGray, Core.ROTATE_180);

        try {
            MatOfRect matOfRect = new MatOfRect();
            mFaceDetector.detectMultiScale(matGray.t(), matOfRect);

            final List<Rect> rects = matOfRect.toList();
            Rect faceRec = ExtractionUtils.extractMainFace(rects);

            resultsComputedListener.onFacesRecognized(rects);

            if (faceRec != null) {
                VisualisationUtils.drawFaceMarker(matGray, faceRec);

                Rect normalizedFace = new Rect(faceRec.y, faceRec.x, faceRec.height, faceRec.width);
                Mat faceMat = matGray.submat(normalizedFace);

                MatOfRect eyeMatOfRect = new MatOfRect();
                mEyeDetector.detectMultiScale(faceMat.t(), eyeMatOfRect);

                final List<Rect> eyeRects = eyeMatOfRect.toList();

                FloatBuffer fb1 = null, fb2 = null;
                if (eyeRects != null && rects.size() > 0) {
                    Rect[] eyes = ExtractionUtils.extractEyes(faceRec, eyeRects);

                    fb1 = runRecognition(faceMat, faceRec, eyes[0], matGray, inferenceInterface);
                    VisualisationUtils.drawEyeMarker(matGray, faceRec, eyes[9]);

                    if (eyes[1] != null) {
                        fb2 = runRecognition(faceMat, faceRec, eyes[1], matGray, inferenceInterface);
                        VisualisationUtils.drawEyeMarker(matGray, faceRec, eyes[1]);
                    }
                }

                resultsComputedListener.onVectorsComputed(fb1, fb2);
            }

        } catch (Exception e) {
            Log.e(TAG, "Failed detection: " + e.getMessage(), e);
        }

        Mat targetGray = new Mat();
        Core.rotate(matGray, targetGray, Core.ROTATE_180);
        return targetGray;
    }

    private static FloatBuffer runRecognition(Mat inFaceMat, Rect faceRec, Rect eyeRec, Mat outputMatGray, TensorFlowInferenceInterface inferenceInterface) {
        Mat faceMat = inFaceMat.t();
        Mat rawEyeMat = faceMat.submat(eyeRec);
        Mat eyeMat = new Mat();
        Size size = new Size();
        size.width = 55;
        size.height = 35;
        Imgproc.resize(rawEyeMat, eyeMat, size);

        FloatBuffer fb = TensorFlowUtils.runInference(inferenceInterface, eyeMat);

        VisualisationUtils.drawVectors(outputMatGray, faceRec, eyeRec, fb);

        return fb;
    }
}
