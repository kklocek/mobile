package pl.edu.agh.vision3;

import android.Manifest;
import android.content.Context;
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
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "OCVSample::Activity";
    private static final int FRONT_CAMERA_INDEX = 1;
    private CameraBridgeViewBase _cameraBridgeViewBase;
    private TextView mTextView;
    private TextView mBottomTextView;
    private TensorFlowInferenceInterface inferenceInterface;

    static {
        System.loadLibrary("tensorflow_inference");
    }

    private static final String MODEL_FILE = "file:///android_asset/sight_vector_model.pb";
    private static final String INPUT_NODE = "input_2";
    private static final String OUTPUT_NODE = "output_node0";

    private static final int[] INPUT_SIZE = {1, 35, 55, 1};

    private BaseLoaderCallback _baseLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    // Load ndk built module, as specified in moduleName in build.gradle
                    // after opencv initialization
                    System.loadLibrary("native-lib");
                    loadFaceCascade();
                    loadEyeCascade();
                    _cameraBridgeViewBase.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
            }
        }
    };
    private CascadeClassifier mEyeDetector;

    private void loadEyeCascade() {
        // Load native library after(!) OpenCV initialization
        // System.loadLibrary("detection_based_tracker");

        try {
            // load cascade file from application resources
            InputStream is = getResources().openRawResource(R.raw.haarcascade_eye);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
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
            mEyeDetector = new CascadeClassifier(path);
            if (mEyeDetector.empty()) {
                Log.e(TAG, "Failed to load cascade classifier");
                mEyeDetector = null;
            } else
                Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

            mEyeDetector.load(path);

            cascadeDir.delete();

        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
        }
    }

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

    private void loadFaceCascade() {
        try {
            // load cascade file from application resources
            InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_default);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
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
            mFaceDetector = new CascadeClassifier(path);
            if (mFaceDetector.empty()) {
                Log.e(TAG, "Failed to load cascade classifier");
                mFaceDetector = null;
            } else
                Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

            mFaceDetector.load(path);

            cascadeDir.delete();

        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
        }

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
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, _baseLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            _baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
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
                return;
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
        Mat legsUpsideMatGray = inputFrame.gray();  // ;D
        Mat matGray = new Mat();
        Core.rotate(legsUpsideMatGray, matGray, Core.ROTATE_180);
        try {
            // salt(matGray.getNativeObjAddr(), 2000);
            MatOfRect matOfRect = new MatOfRect();
            mFaceDetector.detectMultiScale(matGray.t(), matOfRect);

            final List<Rect> rects = matOfRect.toList();
            double max = 0;
            Rect faceRec = null;
            for (Rect r : rects) {
                if (r.area() > max) {
                    max = r.area();
                    faceRec = r;
                }
            }

            this.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    if (!rects.isEmpty()) {
                        String text = "Found: " + rects.size() + " , 1st: ["
                                + rects.get(0).y + ","
                                + rects.get(0).x + ","
                                + rects.get(0).width + ","
                                + rects.get(0).height + "]";
                        mTextView.setText(text);
                    } else {
                        mTextView.setText("Shit, no faces.");
                    }
                }
            });

            if (faceRec != null) {
                Imgproc.rectangle(matGray,
                        new Point(faceRec.y, faceRec.x),
                        new Point(faceRec.y + faceRec.height, faceRec.x + faceRec.width),
                        new Scalar(0, 255, 0, 255),
                        10);

                Rect normalizedFace = new Rect(faceRec.y, faceRec.x, faceRec.height, faceRec.width);
                Mat faceMat = matGray.submat(normalizedFace);

                MatOfRect eyeMatOfRect = new MatOfRect();
                mEyeDetector.detectMultiScale(faceMat.t(), eyeMatOfRect);

                final List<Rect> eyeRects = eyeMatOfRect.toList();

                if (eyeRects != null && eyeRects.size() > 0) {
                    double biggestSize = 0;
                    double secondBiggestSize = 0;
                    Rect biggestRec = null;
                    Rect secondBiggestRec = null;
                    for (Rect r : eyeRects) {
                        if (r.area() > 0.3 * faceRec.area()) {
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

                    Imgproc.circle(matGray,
                            new Point(faceRec.y + biggestRec.y + (biggestRec.height/2), faceRec.x + biggestRec.x + (biggestRec.width/2)),
                            (biggestRec.height + biggestRec.width)/4,
                            new Scalar(0, 255, 255, 255),
                            2);
//                    Imgproc.rectangle(matGray,
//                            new Point(faceRec.y + biggestRec.y, faceRec.x + biggestRec.x),
//                            new Point(faceRec.y + biggestRec.y + biggestRec.height, faceRec.x + biggestRec.x + biggestRec.width),
//                            new Scalar(0, 255, 255, 255),
//                            2);

                    runRecognition(faceMat, faceRec, biggestRec, matGray);

                    if (secondBiggestRec != null) {
                        Imgproc.circle(matGray,
                                new Point(faceRec.y + secondBiggestRec.y + (secondBiggestRec.height/2), faceRec.x + secondBiggestRec.x + (secondBiggestRec.width/2)),
                                (secondBiggestRec.height + secondBiggestRec.width)/4,
                                new Scalar(0, 255, 255, 255),
                                2);
//                        Imgproc.rectangle(matGray,
//                                new Point(faceRec.y + secondBiggestRec.y, faceRec.x + secondBiggestRec.x),
//                                new Point(faceRec.y + secondBiggestRec.y + secondBiggestRec.height, faceRec.x + secondBiggestRec.x + secondBiggestRec.width),
//                                new Scalar(0, 255, 255, 255),
//                                1);
                        runRecognition(faceMat, faceRec, secondBiggestRec, matGray);
                    }
                }
            }


        } catch (Exception e) {
            Log.e(TAG, "Failed detection: " + e.getMessage(), e);
        }


        return matGray;
    }

    private void runRecognition(Mat inFaceMat, Rect faceRec, Rect eyeRec, Mat outputMatGray) {
        Mat faceMat = inFaceMat;//new Mat();
//        Core.normalize(faceMat, faceMat);
        // running recognition
        int width = 35;
        int height = 55;
        float[] inputFloats = new float[width * height];

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                inputFloats[i * width + j] = (float) (faceMat.get(eyeRec.x + i, eyeRec.y + j)[0]/255.0);// getting gray
            }
        }

        inferenceInterface.fillNodeFloat(INPUT_NODE, INPUT_SIZE, inputFloats);
//        inferenceInterface.fillNodeFloat(INPUT_NODE, INPUT_SIZE, inputFloats);
        inferenceInterface.runInference(new String[]{OUTPUT_NODE});
//                    final float[] resu = {0, 0};
        final FloatBuffer fb = FloatBuffer.allocate(2);
        inferenceInterface.readNodeIntoFloatBuffer(OUTPUT_NODE, fb);

        this.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                String text = "Direction: [" + fb.get(0) + ", " + fb.get(1) + "]";
                mBottomTextView.setText(text);
            }
        });

        Point point = computeCircleCenter(faceRec, eyeRec, fb);
        double promotion = (Math.abs(fb.get(1)) + Math.abs(fb.get(0)))/2;
        Imgproc.circle(outputMatGray,
                point,
                (int) (1 + (promotion * 4)),
                new Scalar(255, 255, 255, 255),
                (int) (1 + (promotion * 10)));
        Imgproc.circle(outputMatGray,
                point,
                10,
                new Scalar(255, 255, 255, 255),
                1);
    }

    private Point computeCircleCenter(Rect faceRec, Rect rec, FloatBuffer fb) {
        int bigCenterX = faceRec.y + rec.y + (rec.height/2);
        int bigCenterY = faceRec.x + rec.x + (rec.width/2);
        double radius = Math.pow((rec.height + rec.width)/4, 2);
        double x = Math.sqrt(radius/(1 + Math.pow(fb.get(1)/fb.get(0) , 2)));
        if (fb.get(0) < 0) {
            x*= -1;
        }
        double y = (int) (fb.get(1) / fb.get(0) * x);
        return new Point(bigCenterX + x, bigCenterY + y);
    }


    public native void salt(long matAddrGray, int nbrElem);
}