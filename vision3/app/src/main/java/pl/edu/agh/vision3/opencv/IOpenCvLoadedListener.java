package pl.edu.agh.vision3.opencv;

import org.opencv.objdetect.CascadeClassifier;

public interface IOpenCvLoadedListener {
    void onEyeDetectionCascadeLoaded(CascadeClassifier cascadeClassifier);

    void onFaceDetecitonCascadeLoaded(CascadeClassifier cascadeClassifier);
}
