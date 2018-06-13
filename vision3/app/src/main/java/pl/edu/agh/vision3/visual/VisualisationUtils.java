package pl.edu.agh.vision3.visual;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;

public class VisualisationUtils {

    public static void drawVectors(Mat outputMatGray, Rect faceRec, Rect eyeRec, FloatBuffer fb) {
        Point point = computeCircleCenter(faceRec, eyeRec, fb);

        Point target = new Point();
        target.x = point.x + (fb.get(1) * 5 * 50);
        target.y = point.y + (fb.get(0) * 5 * 50);

        int thickness = 3;
        int line_type = 8;
        int shift = 0;
        double tipLength = 0.1;

        Imgproc.arrowedLine(outputMatGray,
                point,
                target,
                white(),
                thickness,
                line_type,
                shift,
                tipLength);
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

    public static void drawEyeMarker(Mat matGray, Rect faceRec, Rect eye) {
        Imgproc.circle(matGray,
                new Point(faceRec.y + eye.y + (eye.height / 2), faceRec.x + eye.x + (eye.width / 2)),
                (eye.height + eye.width) / 4,
                black(),
                2);
    }

    public static void drawFaceMarker(Mat matGray, Rect faceRec) {
        Imgproc.rectangle(matGray,
                new Point(faceRec.y, faceRec.x),
                new Point(faceRec.y + faceRec.height, faceRec.x + faceRec.width),
                black(),
                10);
    }

    private static Scalar white() {
        return new Scalar(255, 255, 255, 255);
    }

    private static Scalar black() {
        return new Scalar(0, 255, 255, 255);
    }
}
