package pl.edu.agh.vision3.visual;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;

/**
 * Utility class holding methods for drawing markers on the image and highlight the recognized elements.
 */
public class VisualisationUtils {

    private static final int EYE_MARKER_THICKNESS = 2;
    private static final int FACE_MARKER_THICKNESS = 10;
    private static float VECTOR_SCALE = 5 * 50;

    private static final int THICKNESS = 3;
    private static final int LINE_TYPE = 8;
    private static final int SHIFT = 0;
    private static final double TIP_LENGTH = 0.1;

    private static final Scalar WHITE = new Scalar(255, 255, 255, 255);
    private static final Scalar BLACK = new Scalar(0, 255, 255, 255);

    /**
     * Draws arrowed vector on outputMatGray.
     * For arrow beginning takes faceRectangle and eyeRectangle.
     * For drawing the vector fb is used.
     *
     * @param outputMatGray target canvas for the method
     * @param faceRec face rectangle
     * @param eyeRec eye rectangle
     * @param fb sight vector
     */
    public static void drawVectors(Mat outputMatGray, Rect faceRec, Rect eyeRec, FloatBuffer fb) {
        Point point = computeCircleCenter(faceRec, eyeRec, fb);

        Point target = new Point();
        target.x = point.x + (fb.get(1) * VECTOR_SCALE);
        target.y = point.y + (fb.get(0) * VECTOR_SCALE);

        Imgproc.arrowedLine(outputMatGray,
                point,
                target,
                WHITE,
                THICKNESS,
                LINE_TYPE,
                SHIFT,
                TIP_LENGTH);
    }

    private static Point computeCircleCenter(Rect faceRec, Rect rec, FloatBuffer fb) {
        int bigCenterX = faceRec.y + rec.y + (rec.height / 2);
        int bigCenterY = faceRec.x + rec.x + (rec.width / 2);

        double radius = Math.pow((rec.height + rec.width) / 4, 2);
        double x = Math.sqrt(radius / (1 + Math.pow(fb.get(0) / fb.get(1), 2)));

        if (fb.get(1) < 0) {
            x *= -1;
        }

        double y = (int) (fb.get(0) / fb.get(1) * x);

        return new Point(bigCenterX + x, bigCenterY + y);
    }

    /**
     * Draws eye marker. (Circle)
     *
     * @param matGray target canvas for drawing
     *
     * @param faceRec face rectangle
     * @param eye eye rectangle
     */
    public static void drawEyeMarker(Mat matGray, Rect faceRec, Rect eye) {
        Imgproc.circle(matGray,
                new Point(faceRec.y + eye.y + (eye.height / 2), faceRec.x + eye.x + (eye.width / 2)),
                (eye.height + eye.width) / 4,
                BLACK,
                EYE_MARKER_THICKNESS);
    }

    /**
     * Draws face marker. (Rectangle)
     *
     * @param matGray target canvas for drawing
     * @param faceRec face rectangle
     */
    public static void drawFaceMarker(Mat matGray, Rect faceRec) {
        Imgproc.rectangle(matGray,
                new Point(faceRec.y, faceRec.x),
                new Point(faceRec.y + faceRec.height, faceRec.x + faceRec.width),
                BLACK,
                FACE_MARKER_THICKNESS);
    }
}
