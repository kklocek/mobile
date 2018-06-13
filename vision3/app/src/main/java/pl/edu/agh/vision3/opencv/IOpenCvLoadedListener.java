package pl.edu.agh.vision3.opencv;

import org.opencv.objdetect.CascadeClassifier;

/**
 * Listener for detection cascade loading results
 */
public interface IOpenCvLoadedListener {
    /**
     * Method to be invoked after eye detection cascade is loaded
     * @param cascadeClassifier eye detection cascade
     */
    void onEyeDetectionCascadeLoaded(CascadeClassifier cascadeClassifier);

    /**
     * Method to be invoked after face detection cascade is loaded
     * @param cascadeClassifier face detection cascade
     */
    void onFaceDetecitonCascadeLoaded(CascadeClassifier cascadeClassifier);
}
