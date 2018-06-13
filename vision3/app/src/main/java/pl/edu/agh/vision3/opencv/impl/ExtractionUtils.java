package pl.edu.agh.vision3.opencv.impl;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.util.List;

/**
 * Utility class for filtering most important Mats from list of Mats and for cutting the subMats from Mats.
 */
public class ExtractionUtils {

    /**
     * Extracts the biggest rectangle
     *
     * @param rects list of rectangles
     *
     * @return the biggest rectangle
     */
    public static Rect filterOutMainFace(List<Rect> rects) {
        double max = 0;
        Rect faceRec = null;
        for (Rect r : rects) {
            if (r.area() > max) {
                max = r.area();
                faceRec = r;
            }
        }
        return faceRec;
    }

    /**
     * Extracts 2 biggest rectangles, but not bigger then face (filtering with scale factor)
     *
     * @param faceRect face rectangle
     * @param rects list of eye rectangles
     *
     * @return 2 biggest eye rectangles (but still acceptably big)
     */
    public static Rect[] filterOutEyes(Rect faceRect, List<Rect> rects) {
        Rect[] outRects = new Rect[2];
        double biggestSize = 0;
        double secondBiggestSize = 0;
        Rect biggestRec = null;
        Rect secondBiggestRec = null;
        for (Rect r : rects) {
            if (r.area() > 0.3 * faceRect.area()) {
                continue;
            }
            if (r.area() > biggestSize) {
                secondBiggestRec = biggestRec;
                biggestRec = r;
                secondBiggestSize = biggestSize;
                biggestSize = r.area();
            } else if (r.area() > secondBiggestSize) {
                secondBiggestRec = r;
                secondBiggestSize = r.area();
            }
        }

        outRects[0] = biggestRec;
        outRects[1] = secondBiggestRec;
        return outRects;
    }

    /**
     * Extracts Mat from input frame and applies transformations for the ability to be processed by cascade haar.
     *
     * @param inputFrame
     * @return
     */
    public static Mat extractMatFromInputFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat verticallyFlippedMatGray = inputFrame.gray();
        Mat matGray = new Mat();
        Core.rotate(verticallyFlippedMatGray, matGray, Core.ROTATE_180);
        return matGray;
    }

    public static Mat extractFace(Mat matGray, Rect faceRec) {
        Rect normalizedFace = new Rect(faceRec.y, faceRec.x, faceRec.height, faceRec.width);
        return matGray.submat(normalizedFace);
    }
}
