package pl.edu.agh.vision3.opencv.load;

import android.content.Context;
import android.util.Log;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import pl.edu.agh.vision3.R;
import pl.edu.agh.vision3.opencv.IOpenCvLoadedListener;

public class OpenCVBaseLoaderCallback extends BaseLoaderCallback {
    private static final String TAG = "OpenCvLoaderCallback";
    private final CameraBridgeViewBase cameraBridgeViewBase;
    private final IOpenCvLoadedListener iOpenCvLoadedListener;

    public OpenCVBaseLoaderCallback(Context AppContext, CameraBridgeViewBase _cameraBridgeViewBase, IOpenCvLoadedListener iOpenCvLoadedListener) {
        super(AppContext);
        cameraBridgeViewBase = _cameraBridgeViewBase;
        this.iOpenCvLoadedListener = iOpenCvLoadedListener;
    }

    @Override
    public void onManagerConnected(int status) {
        switch (status) {
            case LoaderCallbackInterface.SUCCESS: {
                Log.i(TAG, "OpenCV loaded successfully");
                System.loadLibrary("native-lib");
                iOpenCvLoadedListener.onFaceDetecitonCascadeLoaded(loadFaceCascade());
                iOpenCvLoadedListener.onEyeDetectionCascadeLoaded(loadEyeCascade());
                cameraBridgeViewBase.enableView();
            }
            break;
            default: {
                super.onManagerConnected(status);
            }
        }
    }

    private CascadeClassifier loadFaceCascade() {
        try {
            // load cascade file from application resources
            InputStream is = this.mAppContext.getResources().openRawResource(R.raw.haarcascade_frontalface_default);
            File cascadeDir = this.mAppContext.getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_default.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            String path = mCascadeFile.getAbsolutePath();
            CascadeClassifier mFaceDetector = new CascadeClassifier(path);
            if (mFaceDetector.empty()) {
                Log.e(TAG, "Failed to load cascade classifier");
                mFaceDetector = null;
            } else
                Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

            mFaceDetector.load(path);

            cascadeDir.delete();

            return mFaceDetector;
        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
        }
        return null;

    }

    private CascadeClassifier loadEyeCascade() {
        // Load native library after(!) OpenCV initialization
        // System.loadLibrary("detection_based_tracker");

        try {
            // load cascade file from application resources
            InputStream is = this.mAppContext.getResources().openRawResource(R.raw.haarcascade_eye);
            File cascadeDir = this.mAppContext.getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "haarcascade_eye.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            String path = mCascadeFile.getAbsolutePath();
            CascadeClassifier mEyeDetector = new CascadeClassifier(path);
            if (mEyeDetector.empty()) {
                Log.e(TAG, "Failed to load cascade classifier");
                mEyeDetector = null;
            } else
                Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

            mEyeDetector.load(path);

            cascadeDir.delete();

            return mEyeDetector;
        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
        }
        return null;
    }
}
