package pl.edu.agh.vision3;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.nio.FloatBuffer;

import pl.edu.agh.vision3.opencv.ICameraViewConnector;
import pl.edu.agh.vision3.opencv.IOpenCvLoadedListener;
import pl.edu.agh.vision3.opencv.load.OpenCVBaseLoaderCallback;
import pl.edu.agh.vision3.opencv.impl.RecognitionHandler;
import pl.edu.agh.vision3.visual.IResultsComputedListener;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2,
        IOpenCvLoadedListener,
        IResultsComputedListener,
        ICameraViewConnector {
    private static final String MODEL_FILE = "file:///android_asset/sight_vector_model.pb";
    private static final String TAG = "MainAppActivity";
    private static final int FRONT_CAMERA_INDEX = 1;
    private CameraBridgeViewBase _cameraBridgeViewBase;
    private TextView mTextView;
    private TextView mBottomTextView;
    private TensorFlowInferenceInterface inferenceInterface;

    static {
        System.loadLibrary("tensorflow_inference");
    }

    private BaseLoaderCallback _baseLoaderCallback = new OpenCVBaseLoaderCallback(this, this, this);
    private CascadeClassifier mEyeDetector;
    private CascadeClassifier mFaceDetector;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        mTextView = findViewById(R.id.info);
        mBottomTextView = findViewById(R.id.orientation_text);
        // Permissions for Android 6+
        ActivityCompat.requestPermissions(MainActivity.this,
                new String[]{Manifest.permission.CAMERA},
                1);

        _cameraBridgeViewBase = (CameraBridgeViewBase) findViewById(R.id.main_surface);
        _cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        _cameraBridgeViewBase.setCameraIndex(FRONT_CAMERA_INDEX);
        _cameraBridgeViewBase.setCvCameraViewListener(this);

        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE);
    }

    @Override
    public void onPause() {
        super.onPause();
        disableCamera();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCV ::: Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, _baseLoaderCallback);
        } else {
            Log.e(TAG, "OpenCV ::: OpenCV library found inside package. Using it!");
            _baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        Log.e(TAG, "If you see this text. Loading possibly works.");
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults) {
        switch (requestCode) {
            case 1: {
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                } else {
                    Toast.makeText(MainActivity.this, "Permission denied to read your External storage", Toast.LENGTH_SHORT).show();
                }
            }
        }
    }

    public void onDestroy() {
        super.onDestroy();
        disableCamera();
    }

    public void disableCamera() {
        if (_cameraBridgeViewBase != null)
            _cameraBridgeViewBase.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        return RecognitionHandler.handle(
                mFaceDetector,
                mEyeDetector,
                inferenceInterface,
                this,
                inputFrame);
    }


    public native void salt(long matAddrGray, int nbrElem);

    @Override
    public void onEyeDetectionCascadeLoaded(CascadeClassifier eyeDetectionCascade) {
        this.mEyeDetector = eyeDetectionCascade;
    }

    @Override
    public void onFaceDetecitonCascadeLoaded(CascadeClassifier faceDetector) {
        this.mFaceDetector = faceDetector;
    }

    @Override
    public void onFaceRecognized(final Rect rect) {
        this.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (rect != null) {
                    String text = "Found face: ["
                            + rect.y + ","
                            + rect.x + ","
                            + rect.width + ","
                            + rect.height + "]";
                    mTextView.setText(text);
                } else {
                    mTextView.setText("No faces found, Sir.");
                }
            }
        });
    }

    @Override
    public void onVectorsComputed(final FloatBuffer vector1, final FloatBuffer vector2) {
        this.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                String text;
                if (vector1 != null || vector2 != null) {
                    text = "Eye vectors, Sir: \n";
                    if (vector1 != null) {
                        text += "[" + vector1.get(1) + ", " + vector1.get(0) + "] ";
                    }
                    text += "\n";
                    if (vector2 != null) {
                        text += "[" + vector2.get(1) + ", " + vector2.get(0) + "] ";
                    }
                } else {
                    text = "No vectors, Sir.\n\n";
                }
                mBottomTextView.setText(text);
            }
        });
    }

    @Override
    public CameraBridgeViewBase getCameraBridgeView() {
        return _cameraBridgeViewBase;
    }
}