package pl.edu.agh.vision3.tensorflow;

import org.opencv.core.Mat;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.nio.FloatBuffer;

public class TensorFlowUtils {

    public static final String MODEL_FILE = "file:///android_asset/sight_vector_model.pb";
    public static final String INPUT_NODE = "input_2";
    public static final String OUTPUT_NODE = "output_node0";

    private static final double NORMALIZATION_FACTOR = 255.0;

    public static final int WIDTH = 55;
    public static final int HEIGHT = 35;

    public static final int[] INPUT_SIZE = {1, HEIGHT, WIDTH, 1};

    public static FloatBuffer runInference(TensorFlowInferenceInterface inferenceInterface, Mat eyeMat) {
        float[] inputFloats = new float[eyeMat.width() * eyeMat.height()];

        for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; j++) {
                inputFloats[i * WIDTH + j] = (float) (eyeMat.get(i, j)[0] / NORMALIZATION_FACTOR);
            }
        }

        inferenceInterface.fillNodeFloat(INPUT_NODE, INPUT_SIZE, inputFloats);
        inferenceInterface.runInference(new String[]{OUTPUT_NODE});
        FloatBuffer fb = FloatBuffer.allocate(2);
        inferenceInterface.readNodeIntoFloatBuffer(OUTPUT_NODE, fb);
        fb.put(1, -fb.get(1));
        return fb;
    }
}
