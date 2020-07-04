/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.ImageReader.OnImageAvailableListener;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.Handler;
import android.os.Message;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;

import androidx.core.app.ActivityCompat;

import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.StorageReference;
import com.google.firebase.storage.UploadTask;

import org.json.JSONObject;
import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.Classifier2;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.LinkedList;
import java.util.List;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
    public final static int REPEAT_DELAY = 5000;
    public final static int NOISE_REPEAT_DELAY = 10000;
    private static final Logger LOGGER = new Logger();
    // Configuration values for the prepackaged SSD model.
    private static final int TF_OD_API_INPUT_SIZE = 512;
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static final String TF_OD_API_MODEL_FILE = "face_detect.tflite";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/face_label.txt";
    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    private static final float MINIMUM_LIGHT_SENSOR_VALUE = 20.0f;
    private static final float MINIMUM_DB = 70.0f;
    private static final boolean MAINTAIN_ASPECT = false;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    public static double REFERENCE = 0.00002;
    public Handler handlerc = null;
    public Handler noiseHandler = null;
    OverlayView trackingOverlay;
    private int total_count = 0;
    private int crying_count = 0;
    private int not_count = 0;
    private String detect_image = "/detect.jpg";
    private Integer sensorOrientation;
    private Classifier detector;
    private Classifier2 classifier;
    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;
    private Bitmap storageBitmap = null;
    private Uri storageUri = null;
    private boolean computingDetection = false;
    private long timestamp = 0;
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private MultiBoxTracker tracker;
    private BorderedText borderedText;
    private float sensorValue;

    @SuppressLint("HandlerLeak")
    @Override
    public synchronized void onResume() {
        LOGGER.d("onResume " + this);
        super.onResume();
        handlerc = new Handler() {
            public void handleMessage(Message msg) {
                super.handleMessage(msg);
                LOGGER.d("Total Count : " + total_count);
                LOGGER.d("Crying Count : " + crying_count);
                if (total_count != 0 && total_count > 15) {
                    if (total_count - 1 <= crying_count) {
                        stoImage();
                    }
                    total_count = 0;
                    crying_count = 0;
                }
                this.sendEmptyMessageDelayed(0, REPEAT_DELAY);
            }
        };
        handlerc.sendEmptyMessage(0);

        noiseHandler = new Handler() {
            public void handleMessage(Message msg) {
                super.handleMessage(msg);
                fireNoise();
                this.sendEmptyMessageDelayed(0, NOISE_REPEAT_DELAY);
            }
        };
        noiseHandler.sendEmptyMessage(0);
    }

    @Override
    public synchronized void onPause() {
        LOGGER.d("onPause " + this);
        super.onPause();
        handlerc.removeMessages(0);
        noiseHandler.removeMessages(0);
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        int cropSize = TF_OD_API_INPUT_SIZE;

        try {
            detector =
                    TFLiteObjectDetectionAPIModel.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_INPUT_SIZE,
                            TF_OD_API_IS_QUANTIZED);
            cropSize = TF_OD_API_INPUT_SIZE;
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        recreateClassifier();
        if (classifier == null) {
            LOGGER.e("No classifier on preview!");
            return;
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                canvas -> {
                    tracker.draw(canvas);
                    if (isDebug()) {
                        tracker.drawDebug(canvas);
                    }
                });

        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    }

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(() -> {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            final Classifier.Recognition result = detector.recognizeImage(croppedBitmap);

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas1 = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;

            final List<Classifier.Recognition> mappedRecognitions =
                    new LinkedList<>();

            final RectF location = result.getLocation();
            if (location != null && result.getConfidence() >= minimumConfidence) {
                canvas1.drawRect(location, paint);

                cropToFrameTransform.mapRect(location);

                result.setLocation(location);
                mappedRecognitions.add(result);
                float left = result.getLocation().left;
                float top = result.getLocation().top;
                float right = result.getLocation().right;
                float bottom = result.getLocation().bottom;
                try {
                    Bitmap.createBitmap(rgbFrameBitmap, (int) left, (int) top, (int) (right - left), (int) (bottom - top));
                } catch (IllegalArgumentException e) {
                    e.printStackTrace();
                    LOGGER.e(e, "IllegalArgumentException");
                }
            }

            tracker.trackResults(mappedRecognitions, currTimestamp);
            trackingOverlay.postInvalidate();

            computingDetection = false;

            if (location != null && result.getConfidence() >= minimumConfidence && classifier != null) {
                storageBitmap = rgbFrameBitmap;
                Canvas canvas2 = new Canvas(rgbFrameBitmap);
                Paint paint2 = new Paint();
                ColorMatrix colorMatrix = new ColorMatrix();
                colorMatrix.setSaturation(0);
                ColorMatrixColorFilter colorMatrixFilter = new ColorMatrixColorFilter(colorMatrix);
                paint2.setColorFilter(colorMatrixFilter);
                canvas2.drawBitmap(rgbFrameBitmap, 0, 0, paint2);
                rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, 224, 224);
                final List<Classifier2.Recognition> results =
                        classifier.recognizeImage(rgbFrameBitmap, sensorOrientation);
                LOGGER.v("Detect: %s", results);

                not_count = 0;
                if (results.get(0).toString().contains("Cry")) {
                    crying_count++;
                    total_count++;
                } else {
                    total_count++;
                }

                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                LOGGER.d("Processing Time : " + lastProcessingTimeMs);

                if (noise) {
                    double db = getNoiseLevel();
                    runOnUiThread(() -> {
                        showResultsInBottomSheet(results);
                        showInference(lastProcessingTimeMs + "ms");
                        showDB((int) db + "dB");
                    });
                    if (db >= MINIMUM_DB) {
                        LOGGER.d("Noise >= " + MINIMUM_DB + "dB");
                        pushAll("BFER", "Noise appeared!!");
                    }
                } else {
                    runOnUiThread(() -> {
                        showResultsInBottomSheet(results);
                        showInference(lastProcessingTimeMs + "ms");
                        showDB("-dB");
                    });
                }
            } else {// 디텍팅이 안됐을 경우
                storageUri = null;
                LOGGER.d("Can't Detecting face, Sensor value : " + sensorValue);
                if (sensorValue <= MINIMUM_LIGHT_SENSOR_VALUE) {
                    double db = getNoiseLevel();
                    runOnUiThread(() -> {
                        showInference("-ms");
                        showDB((int) db + "dB");
                    });
                    // MINIMUM_DB 데시벨 보다 높은 값이 측정 됐을 때
                    if (db >= MINIMUM_DB) {
                        LOGGER.d("Noise >= " + MINIMUM_DB + "dB");
                        pushAll("BFER", "Noise appeared!!");
                    }
                } else {
                    // 뒷통수 및 얼굴감지 안된 거 푸쉬
                    not_count++;
                    LOGGER.d("Not Count : " + not_count);
                    if (not_count > 10) {
                        pushAll("BFER", "Can't find face");
                        not_count = 0;
                    }
                }
            }
        });
    }

    protected void stoImage() {
        FirebaseStorage storage = FirebaseStorage.getInstance(getString(R.string.fcm_image_url));
        StorageReference storageRef = storage.getReference();
        StorageReference mountainImagesRef = storageRef.child(id + detect_image);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        storageBitmap.compress(Bitmap.CompressFormat.JPEG, 100, baos);
        byte[] data = baos.toByteArray();

        UploadTask uploadTask = mountainImagesRef.putBytes(data);
        uploadTask.addOnFailureListener(exception -> {
            // Handle unsuccessful uploads
            storageUri = null;
        }).addOnSuccessListener(taskSnapshot -> {
            // taskSnapshot.getMetadata() contains file metadata such as size, content-type, etc.
            // ...
            storageRef.child(id + detect_image).getDownloadUrl().addOnSuccessListener(uri -> {
                // Got the download URL for 'users/me/profile.png'
                storageUri = uri;
                pushAll("BFER", "Baby is Crying!!");
            }).addOnFailureListener(exception -> {
                // Handle any errors
                storageUri = null;
            });

            LOGGER.d("Storage URI : " + storageUri);
        });
    }

    protected void sendPostToFCM(final String token, final String title, final String message) {
        db.collection("users")
                .document(id).addSnapshotListener((documentSnapshot, e) -> new Thread(() -> {
            try {
                // FCM 메시지 생성 start
                JSONObject root = new JSONObject();
                JSONObject notification = new JSONObject();
                notification.put("body", message);
                notification.put("title", title);
                notification.put("image", storageUri);
                root.put("notification", notification);
                root.put("to", token);
                // FCM 메시지 생성 end
                LOGGER.d("JSON request : " + root.toString());

                URL Url = new URL(FCM_MESSAGE_URL);
                HttpURLConnection conn = (HttpURLConnection) Url.openConnection();
                conn.setRequestMethod("POST");
                conn.setDoOutput(true);
                conn.setDoInput(true);
                conn.addRequestProperty("Authorization", "key=" + SERVER_KEY);
                conn.setRequestProperty("Accept", "application/json");
                conn.setRequestProperty("Content-type", "application/json");
                OutputStream os = conn.getOutputStream();
                os.write(root.toString().getBytes(StandardCharsets.UTF_8));
                os.flush();
                conn.getResponseCode();
            } catch (Exception e1) {
                e1.printStackTrace();
            }
        }).start());
    }

    public double getNoiseLevel() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, 1);
        }

        LOGGER.d("Noise : start new recording process");
        int bufferSize = AudioRecord.getMinBufferSize(44100, AudioFormat.CHANNEL_IN_DEFAULT, AudioFormat.ENCODING_PCM_16BIT);

        bufferSize = bufferSize * 4;
        AudioRecord recorder = new AudioRecord(MediaRecorder.AudioSource.MIC,
                44100, AudioFormat.CHANNEL_IN_DEFAULT, AudioFormat.ENCODING_PCM_16BIT, bufferSize);

        short[] data = new short[bufferSize];
        double average = 0.0;
        recorder.startRecording();
        //recording data;
        recorder.read(data, 0, bufferSize);

        recorder.stop();
        LOGGER.d("Noise : stop");
        for (short s : data) {
            if (s > 0) {
                average += Math.abs(s);
            } else {
                bufferSize--;
            }
        }
        //x=max;
        double x = average / bufferSize;
        LOGGER.d("Noisd : " + x);
        recorder.release();
        double db = 0;
        if (x == 0) {
            LOGGER.e("Noise : error x = 0");
            return db;
        }
        // calculating the pascal pressure based on the idea that the max amplitude (between 0 and 32767) is
        // relative to the pressure
        double pressure = x / 51805.5336; //the value 51805.5336 can be derived from asuming that x=32767=0.6325 Pa and x=1 = 0.00002 Pa (the reference value)
        LOGGER.d("Noise : x=" + pressure + "pa");
        db = (20 * Math.log10(pressure / REFERENCE));
        LOGGER.d("Noise : db = " + db);
        if (db > 0) {
            LOGGER.e("Noise : error x=0");
        }
        return db;
    }

    @Override
    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    private void recreateClassifier() {
        if (classifier != null) {
            LOGGER.d("Closing classifier.");
            classifier.close();
            classifier = null;
        }
        try {
            classifier = Classifier2.create(this, Classifier2.Model.FLOAT_MOBILENET, Classifier2.Device.GPU, 1);
        } catch (IOException e) {
            LOGGER.e(e, "Failed to create classifier.");
        }
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_LIGHT) {
            sensorValue = event.values[0];
            LOGGER.d("SENSOR VALUE : " + sensorValue);
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }
}
