package pl.edu.agh.vision3.opencv.impl;

import org.opencv.core.Rect;

import java.util.List;

public class ExtractionUtils {
    public static Rect extractMainFace(List<Rect> rects) {
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

    public static Rect[] extractEyes(Rect faceRect, List<Rect> rects) {
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
}
