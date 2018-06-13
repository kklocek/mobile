package pl.edu.agh.vision3.visual;

import org.opencv.core.Rect;

import java.nio.FloatBuffer;
import java.util.List;

public interface IResultsComputedListener {
    void onFacesRecognized(List<Rect> rects);

    void onVectorsComputed(FloatBuffer finalFb, FloatBuffer finalFb1);
}
