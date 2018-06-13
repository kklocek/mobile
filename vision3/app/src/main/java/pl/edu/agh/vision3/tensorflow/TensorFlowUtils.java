package pl.edu.agh.vision3.tensorflow;

import org.opencv.core.Mat;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.nio.FloatBuffer;

/**
 * Runner of Tensorflow-defined Neural Network inference algorithm.
 */
public class TensorFlowUtils {

    /**
     * Neural network width
     */
    public static final int INPUT_WIDTH = 55;
    /**
     * Neural network height
     */
    public static final int INPUT_HEIGHT = 35;

    private static final int[] INPUT_SIZE = {1, INPUT_HEIGHT, INPUT_WIDTH, 1};

    private static final String INPUT_NODE = "input_2";
    private static final String OUTPUT_NODE = "output_node0";

    private static final double NORMALIZATION_FACTOR = 255.0;


    /**
     * Inference algorithm runner
     *
     * @param inferenceInterface tensorflow inference interface
     * @param eyeMat input mat for image recognition (must be of input size)
     * @return recognized sight vector
     */
    public static FloatBuffer runInference(TensorFlowInferenceInterface inferenceInterface, Mat eyeMat) {
        // linearize image first channel (gray) and normalize it to [0;1]
        float[] inputFloats = new float[eyeMat.width() * eyeMat.height()];
        for (int i = 0; i < INPUT_HEIGHT; i++) {
            for (int j = 0; j < INPUT_WIDTH; j++) {
                inputFloats[i * INPUT_WIDTH + j] = (float) (eyeMat.get(i, j)[0] / NORMALIZATION_FACTOR);
            }
        }

        // fill input
        inferenceInterface.fillNodeFloat(INPUT_NODE, INPUT_SIZE, inputFloats);

        // triggering algorithm by definiton of last node to be executed
        // all dependent will be executed also
        inferenceInterface.runInference(new String[]{OUTPUT_NODE});

        // reading result
        FloatBuffer fb = FloatBuffer.allocate(2);
        inferenceInterface.readNodeIntoFloatBuffer(OUTPUT_NODE, fb);

        // normalization of result
        fb.put(1, -fb.get(1));
        return fb;
    }
}
