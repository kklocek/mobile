package pl.edu.agh.vision3.visual;

import org.opencv.core.Rect;

import java.nio.FloatBuffer;

/**
 * Definition of listener for detection results.
 */
public interface IResultsComputedListener {
    /**
     * Method to be called after the face gets recognized
     * @param rect face rectangle
     */
    void onFaceRecognized(Rect rect);

    /**
     * Method to be called after the sight vectors are recognized
     * @param vector1 vector for the 1st eye (greater in terms of rectangle area)
     * @param vector2 vector for the 2nd eye (smaller in terms of rectangle area)
     */
    void onVectorsComputed(FloatBuffer vector1, FloatBuffer vector2);
}
