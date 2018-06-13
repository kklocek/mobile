package pl.edu.agh.vision3.opencv.load;

import android.content.Context;
import android.util.Log;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

import pl.edu.agh.vision3.R;
import pl.edu.agh.vision3.opencv.ICameraViewConnector;
import pl.edu.agh.vision3.opencv.IOpenCvLoadedListener;

/**
 * Open CV loader callback. Main mathod is invoked after OpenCV is loaded.
 */
public class OpenCVBaseLoaderCallback extends BaseLoaderCallback {
    private static final String TAG = "OpenCvLoaderCallback";

    private static final String FRONT_FACE_CASCADE_FILE_NAME = "haarcascade_frontalface_default.xml";
    private static final String EYE_CASCADE_FILE_NAME = "haarcascade_eye.xml";
    private static final String CASCADE_DIR_NAME = "cascade";

    private final IOpenCvLoadedListener iOpenCvLoadedListener;
    private final ICameraViewConnector iCameraViewConnector;
    private int READ_BUFFER_SIZE = 4096;

    public OpenCVBaseLoaderCallback(Context AppContext,
                                    IOpenCvLoadedListener iOpenCvLoadedListener,
                                    ICameraViewConnector iCameraViewConnector) {
        super(AppContext);
        this.iOpenCvLoadedListener = iOpenCvLoadedListener;
        this.iCameraViewConnector = iCameraViewConnector;
    }

    /**
     * Loads native libraries and eye/face cascades. Enables camera view.
     *
     * @param status loading status
     */
    @Override
    public void onManagerConnected(int status) {
        switch (status) {
            case LoaderCallbackInterface.SUCCESS: {
                Log.i(TAG, "OpenCV loaded successfully");
                System.loadLibrary("native-lib");
                iOpenCvLoadedListener.onFaceDetecitonCascadeLoaded(loadFaceCascade());
                iOpenCvLoadedListener.onEyeDetectionCascadeLoaded(loadEyeCascade());
                iCameraViewConnector.getCameraBridgeView().enableView();
            }
            break;
            default: {
                super.onManagerConnected(status);
            }
        }
    }

    private CascadeClassifier loadFaceCascade() {
        return loadCascade(R.raw.haarcascade_frontalface_default, FRONT_FACE_CASCADE_FILE_NAME);
    }

    private CascadeClassifier loadEyeCascade() {
        return loadCascade(R.raw.haarcascade_eye, EYE_CASCADE_FILE_NAME);
    }

    private CascadeClassifier loadCascade(int resourceId, String cascadeFileName) {
        try {
            // load cascade file from application resources
            InputStream is = this.mAppContext.getResources().openRawResource(resourceId);
            File cascadeDir = this.mAppContext.getDir(CASCADE_DIR_NAME, Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, cascadeFileName);
            FileOutputStream os = new FileOutputStream(mCascadeFile);

            byte[] buffer = new byte[READ_BUFFER_SIZE];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            String path = mCascadeFile.getAbsolutePath();
            CascadeClassifier detector = new CascadeClassifier(path);
            if (detector.empty()) {
                Log.e(TAG, "Failed to load cascade classifier");
                throw new IllegalStateException("Detector " + cascadeFileName + " couldn't be loaded.");
            } else
                Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

            detector.load(path);

            cascadeDir.delete();

            return detector;
        } catch (Exception e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
        }
        return null;
    }
}
