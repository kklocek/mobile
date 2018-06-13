package pl.edu.agh.vision3.opencv;

import org.opencv.android.CameraBridgeViewBase;

/**
 * Camera View Connector. For providing camera view by context.
 */
public interface ICameraViewConnector {

    /**
     * Accessor
     * @return camera view bridge
     */
    CameraBridgeViewBase getCameraBridgeView();
}
