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
import android.app.Fragment;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.hardware.Sensor;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Trace;
import android.util.Size;
import android.view.Surface;
import android.view.View;
import android.view.ViewTreeObserver;
import android.view.WindowManager;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.UiThread;
import androidx.appcompat.app.AppCompatActivity;

import com.google.android.gms.auth.api.signin.GoogleSignIn;
import com.google.android.gms.auth.api.signin.GoogleSignInAccount;
import com.google.android.gms.auth.api.signin.GoogleSignInClient;
import com.google.android.gms.auth.api.signin.GoogleSignInOptions;
import com.google.android.gms.common.SignInButton;
import com.google.android.gms.common.api.ApiException;
import com.google.android.gms.tasks.Task;
import com.google.android.material.bottomsheet.BottomSheetBehavior;
import com.google.firebase.auth.AuthCredential;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.auth.GoogleAuthProvider;
import com.google.firebase.firestore.DocumentReference;
import com.google.firebase.firestore.DocumentSnapshot;
import com.google.firebase.firestore.EventListener;
import com.google.firebase.firestore.FirebaseFirestore;
import com.google.firebase.firestore.FirebaseFirestoreException;
import com.google.firebase.iid.FirebaseInstanceId;

import org.json.JSONObject;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier2;

import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public abstract class CameraActivity extends AppCompatActivity
        implements OnImageAvailableListener,
        Camera.PreviewCallback,
        CompoundButton.OnCheckedChangeListener,
        SensorEventListener,
        View.OnClickListener {
    protected static final int RC_SIGN_IN = 9001;
    private static final Logger LOGGER = new Logger();
    private static final int PERMISSIONS_REQUEST = 1;
    private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
    protected String FCM_MESSAGE_URL = null;
    protected String SERVER_KEY = null;
    protected FirebaseAuth mAuth = null;
    protected FirebaseFirestore db = null;
    protected String id = null;
    protected Map<String, Object> deviceMap = null;
    protected GoogleSignInClient mGoogleSignInClient;
    protected int previewWidth = 0;
    protected int previewHeight = 0;
    protected TextView recognitionTextView,
            recognition1TextView,
            recognition2TextView,
            recognitionValueTextView,
            recognition1ValueTextView,
            recognition2ValueTextView;
    protected LinearLayout ll_result, ll_email, ll_noise, ll_chk_noise;
    protected RelativeLayout rl_signIn;
    protected SignInButton btn_signIn;
    protected ImageButton btn_logout;
    protected TextView tv_email;
    protected TextView inferenceTimeTextView, dbTextView;
    protected CheckBox cb_noise;
    protected ImageView bottomSheetArrowImageView;
    protected String token = null;
    protected boolean noise = false;
    private boolean debug = false;
    private Handler handler;
    private HandlerThread handlerThread;
    private boolean useCamera2API;
    private boolean isProcessingFrame = false;
    private byte[][] yuvBytes = new byte[3][];
    private int[] rgbBytes = null;
    private int yRowStride;
    private Runnable postInferenceCallback;
    private Runnable imageConverter;
    private LinearLayout bottomSheetLayout;
    private LinearLayout gestureLayout;
    private BottomSheetBehavior<LinearLayout> sheetBehavior;
    private SensorManager sensorManager;
    private Sensor lightSensor;

    private static boolean allPermissionsGranted(final int[] grantResults) {
        for (int result : grantResults) {
            if (result != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        LOGGER.d("onCreate " + this);
        super.onCreate(null);

        setContentView(R.layout.tfe_od_activity_camera);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        deviceMap = new HashMap<>();
        FCM_MESSAGE_URL = getString(R.string.fcm_message_url);
        SERVER_KEY = getString(R.string.server_key);
        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        lightSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT);

        if (hasPermission()) {
            setFragment();
        } else {
            requestPermission();
        }

        bottomSheetLayout = findViewById(R.id.bottom_sheet_layout);
        gestureLayout = findViewById(R.id.gesture_layout);
        sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout);
        bottomSheetArrowImageView = findViewById(R.id.bottom_sheet_arrow);

        ViewTreeObserver vto = gestureLayout.getViewTreeObserver();
        vto.addOnGlobalLayoutListener(
                new ViewTreeObserver.OnGlobalLayoutListener() {
                    @Override
                    public void onGlobalLayout() {
                        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.JELLY_BEAN) {
                            gestureLayout.getViewTreeObserver().removeGlobalOnLayoutListener(this);
                        } else {
                            gestureLayout.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                        }
                        //                int width = bottomSheetLayout.getMeasuredWidth();
                        int height = gestureLayout.getMeasuredHeight();

                        sheetBehavior.setPeekHeight(height);
                    }
                });
        sheetBehavior.setHideable(false);

        sheetBehavior.setBottomSheetCallback(
                new BottomSheetBehavior.BottomSheetCallback() {
                    @Override
                    public void onStateChanged(@NonNull View bottomSheet, int newState) {
                        switch (newState) {
                            case BottomSheetBehavior.STATE_HIDDEN:
                                break;
                            case BottomSheetBehavior.STATE_EXPANDED: {
                                bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_down);
                            }
                            break;
                            case BottomSheetBehavior.STATE_COLLAPSED: {
                                bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                            }
                            break;
                            case BottomSheetBehavior.STATE_DRAGGING:
                                break;
                            case BottomSheetBehavior.STATE_SETTLING:
                                bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                                break;
                        }
                    }

                    @Override
                    public void onSlide(@NonNull View bottomSheet, float slideOffset) {
                    }
                });


        recognitionTextView = findViewById(R.id.detected_item);
        recognitionValueTextView = findViewById(R.id.detected_item_value);
        recognition1TextView = findViewById(R.id.detected_item1);
        recognition1ValueTextView = findViewById(R.id.detected_item1_value);
        recognition2TextView = findViewById(R.id.detected_item2);
        recognition2ValueTextView = findViewById(R.id.detected_item2_value);

        ll_result = findViewById(R.id.ll_result);
        rl_signIn = findViewById(R.id.rl_signIn);
        btn_signIn = findViewById(R.id.btn_signIn);
        ll_email = findViewById(R.id.ll_email);
        tv_email = findViewById(R.id.tv_email);
        btn_logout = findViewById(R.id.btn_logout);
        ll_noise = findViewById(R.id.ll_noise);
        ll_chk_noise = findViewById(R.id.ll_chk_noise);
        cb_noise = findViewById(R.id.cb_noise);

        mAuth = FirebaseAuth.getInstance();
        db = FirebaseFirestore.getInstance();
        if (mAuth.getCurrentUser() != null) {
            ll_result.setVisibility(View.VISIBLE);
            rl_signIn.setVisibility(View.GONE);
            ll_email.setVisibility(View.VISIBLE);
            ll_noise.setVisibility(View.VISIBLE);
            id = mAuth.getCurrentUser().getEmail();
            tv_email.setText(id);
            getToken();
            cb_noise.setChecked(noise);
        } else {
            ll_result.setVisibility(View.GONE);
            rl_signIn.setVisibility(View.VISIBLE);
            ll_email.setVisibility(View.GONE);
            ll_noise.setVisibility(View.GONE);
            noise = false;
            cb_noise.setChecked(noise);
            token = null;
        }

        btn_signIn.setOnClickListener(view -> signIn());

        btn_logout.setOnClickListener(view -> logOut());

        ll_chk_noise.setOnClickListener(view -> chkNoise());

        inferenceTimeTextView = findViewById(R.id.inference_info);
        dbTextView = findViewById(R.id.db_info);
    }

    protected void signIn() {
        LOGGER.d("SignIn");
        GoogleSignInOptions gso = new GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN)
                .requestIdToken(getString(R.string.default_client_id))
                .requestEmail()
                .build();
        mGoogleSignInClient = GoogleSignIn.getClient(this, gso);
        Intent signInIntent = mGoogleSignInClient.getSignInIntent();
        startActivityForResult(signInIntent, RC_SIGN_IN);
    }

    protected void logOut() {
        LOGGER.d("SignOut");
        FirebaseAuth.getInstance().signOut();
        token = null;
        updateUI(null);
    }

    protected void chkNoise() {
        noise = !noise;
        deviceMap.put("noise", noise);
        db.collection("users").document(id).update("noise", noise);
        cb_noise.setChecked(noise);
    }

    protected void getToken() {
        FirebaseInstanceId.getInstance().getInstanceId().addOnCompleteListener(task -> {
            if (!task.isSuccessful()) {
                LOGGER.w("FCM getInstanceID Failed" + task.getException());
                return;
            }
            id = id.substring(0, id.indexOf("@"));
            token = task.getResult().getToken();
            LOGGER.d("ID : " + id);
            LOGGER.d("FCM Token : " + token);
            deviceList();
        });
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        // Result returned from launching the Intent from GoogleSignInApi.getSignInIntent(...);
        LOGGER.d("Request Code = " + requestCode);
        LOGGER.d("Result Code = " + resultCode);
        LOGGER.d("Data = " + data);
        if (requestCode == RC_SIGN_IN) {
            Task<GoogleSignInAccount> task = GoogleSignIn.getSignedInAccountFromIntent(data);
            try {
                // Google Sign In was successful, authenticate with Firebase
                GoogleSignInAccount account = task.getResult(ApiException.class);
                LOGGER.d("Account = " + account);
                firebaseAuthWithGoogle(account);
            } catch (ApiException e) {
                LOGGER.d("ApiException = " + e);
            }
        }
    }

    protected void firebaseAuthWithGoogle(GoogleSignInAccount acct) {
        AuthCredential credential = GoogleAuthProvider.getCredential(acct.getIdToken(), null);
        mAuth.signInWithCredential(credential)
                .addOnCompleteListener(this, task -> {
                    if (task.isSuccessful()) {
                        // Sign in success, update UI with the signed-in user's information
                        Toast.makeText(CameraActivity.this, "Authentication Successed.", Toast.LENGTH_SHORT).show();
                        FirebaseUser user = mAuth.getCurrentUser();
                        updateUI(user);
                    } else {
                        // If sign in fails, display a message to the user.
                        Toast.makeText(CameraActivity.this, "Authentication Failed.", Toast.LENGTH_SHORT).show();
                        updateUI(null);
                    }
                });
    }

    protected void deviceList() {
        DocumentReference docRef = db.collection("users").document(id);
        docRef.get().addOnCompleteListener(task -> {
            if (task.isSuccessful()) {
                DocumentSnapshot document = task.getResult();
                if (document.exists()) {
                    deviceMap = document.getData();
                    LOGGER.d("FireStore Document : " + deviceMap);
                    if (!deviceMap.containsKey(token)) {
                        deviceMap.put(token, true);
                        deviceMap.put("noise", noise);
                        db.collection("users").document(id).set(deviceMap);
                    }
                } else {
                    LOGGER.d("FireStore : Not exists Document");
                    deviceMap.put(token, true);
                    deviceMap.put("noise", noise);
                    LOGGER.d("FireStore Document : " + deviceMap);
                    db.collection("users").document(id).set(deviceMap);
                }
            }
        });
    }


    protected void sendPostToFCM(final String token, final String title, final String message) {
        db.collection("users")
                .document(id).addSnapshotListener((documentSnapshot, e) -> new Thread(() -> {
            try {
                // FMC 메시지 생성 start
                JSONObject root = new JSONObject();
                JSONObject notification = new JSONObject();
                notification.put("body", message);
                notification.put("title", title);
                root.put("notification", notification);
                root.put("to", token);
                // FMC 메시지 생성 end

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

    protected void pushAll(final String title, final String message) {
        Set<String> keys = deviceMap.keySet();
        for (String key : keys) {
            LOGGER.d("FCM TOKEN : " + key);
            if (!key.contains("noise"))
                sendPostToFCM(key, title, message);
        }
    }

    protected void fireNoise() {
        final DocumentReference docRef = db.collection("users").document(id);
        docRef.addSnapshotListener(new EventListener<DocumentSnapshot>() {
            @Override
            public void onEvent(@Nullable DocumentSnapshot snapshot,
                                @Nullable FirebaseFirestoreException e) {
                if (e != null) {
                    LOGGER.w("FireNoise : Listen failed.", e);
                    return;
                }

                if (snapshot != null && snapshot.exists()) {
                    LOGGER.d("FireNoise : Current data: " + snapshot.getData());
                    deviceMap = snapshot.getData();
                    if (deviceMap.containsKey("noise"))
                        noise = (boolean) deviceMap.get("noise");
                    cb_noise.setChecked(noise);
                } else {
                    LOGGER.d("FireNoise : Current data: null");
                }
            }
        });
    }

    protected void updateUI(FirebaseUser user) { //update ui code here
        LOGGER.d("User = " + user);
        if (user != null) {
            ll_result.setVisibility(View.VISIBLE);
            rl_signIn.setVisibility(View.GONE);
            ll_email.setVisibility(View.VISIBLE);
            ll_noise.setVisibility(View.VISIBLE);
            cb_noise.setChecked(noise);
            id = mAuth.getCurrentUser().getEmail();
            tv_email.setText(id);
            getToken();
        } else {
            ll_result.setVisibility(View.GONE);
            rl_signIn.setVisibility(View.VISIBLE);
            ll_email.setVisibility(View.GONE);
            ll_noise.setVisibility(View.GONE);
            noise = false;
            cb_noise.setChecked(noise);
        }
    }

    protected int[] getRgbBytes() {
        imageConverter.run();
        return rgbBytes;
    }

    protected int getLuminanceStride() {
        return yRowStride;
    }

    protected byte[] getLuminance() {
        return yuvBytes[0];
    }

    /**
     * Callback for android.hardware.Camera API
     */
    @Override
    public void onPreviewFrame(final byte[] bytes, final Camera camera) {
        if (isProcessingFrame) {
            LOGGER.w("Dropping frame!");
            return;
        }

        try {
            // Initialize the storage bitmaps once when the resolution is known.
            if (rgbBytes == null) {
                Camera.Size previewSize = camera.getParameters().getPreviewSize();
                previewHeight = previewSize.height;
                previewWidth = previewSize.width;
                rgbBytes = new int[previewWidth * previewHeight];
                onPreviewSizeChosen(new Size(previewSize.width, previewSize.height), 90);
            }
        } catch (final Exception e) {
            LOGGER.e(e, "Exception!");
            return;
        }

        isProcessingFrame = true;
        yuvBytes[0] = bytes;
        yRowStride = previewWidth;

        imageConverter = () -> ImageUtils.convertYUV420SPToARGB8888(bytes, previewWidth, previewHeight, rgbBytes);

        postInferenceCallback = () -> {
            camera.addCallbackBuffer(bytes);
            isProcessingFrame = false;
        };
        processImage();
    }

    /**
     * Callback for Camera2 API
     */
    @Override
    public void onImageAvailable(final ImageReader reader) {
        // We need wait until we have some size from onPreviewSizeChosen
        if (previewWidth == 0 || previewHeight == 0) {
            return;
        }
        if (rgbBytes == null) {
            rgbBytes = new int[previewWidth * previewHeight];
        }
        try {
            final Image image = reader.acquireLatestImage();

            if (image == null) {
                return;
            }

            if (isProcessingFrame) {
                image.close();
                return;
            }
            isProcessingFrame = true;
            Trace.beginSection("imageAvailable");
            final Plane[] planes = image.getPlanes();
            fillBytes(planes, yuvBytes);
            yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();

            imageConverter = () -> ImageUtils.convertYUV420ToARGB8888(
                    yuvBytes[0],
                    yuvBytes[1],
                    yuvBytes[2],
                    previewWidth,
                    previewHeight,
                    yRowStride,
                    uvRowStride,
                    uvPixelStride,
                    rgbBytes);

            postInferenceCallback = () -> {
                image.close();
                isProcessingFrame = false;
            };

            processImage();
        } catch (final Exception e) {
            LOGGER.e(e, "Exception!");
            Trace.endSection();
            return;
        }
        Trace.endSection();
    }

    @Override
    public synchronized void onStart() {
        LOGGER.d("onStart " + this);
        super.onStart();
    }

    @Override
    public synchronized void onResume() {
        LOGGER.d("onResume " + this);
        super.onResume();

        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
        sensorManager.registerListener(this, lightSensor, SensorManager.SENSOR_DELAY_NORMAL);
    }

    @Override
    public synchronized void onPause() {
        LOGGER.d("onPause " + this);

        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (final InterruptedException e) {
            LOGGER.e(e, "Exception!");
        }

        super.onPause();
        sensorManager.unregisterListener(this);
    }

    @Override
    public synchronized void onStop() {
        LOGGER.d("onStop " + this);
        super.onStop();
    }

    @Override
    public synchronized void onDestroy() {
        LOGGER.d("onDestroy " + this);
        super.onDestroy();
    }

    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

    @Override
    public void onRequestPermissionsResult(
            final int requestCode, final String[] permissions, final int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSIONS_REQUEST) {
            if (allPermissionsGranted(grantResults)) {
                setFragment();
            } else {
                requestPermission();
            }
        }
    }

    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
                Toast.makeText(
                        CameraActivity.this,
                        "Camera permission is required for this demo",
                        Toast.LENGTH_LONG)
                        .show();
            }
            requestPermissions(new String[]{PERMISSION_CAMERA}, PERMISSIONS_REQUEST);
        }
    }

    // Returns true if the device supports the required hardware level, or better.
    private boolean isHardwareLevelSupported(
            CameraCharacteristics characteristics, int requiredLevel) {
        int deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
        if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
            return requiredLevel == deviceLevel;
        }
        // deviceLevel is not LEGACY, can use numerical sort
        return requiredLevel <= deviceLevel;
    }

    private String chooseCamera() {
        final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            for (final String cameraId : manager.getCameraIdList()) {
                final CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);

                // We don't use a front facing camera in this sample.
                final Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    continue;
                }

                final StreamConfigurationMap map =
                        characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

                if (map == null) {
                    continue;
                }

                // Fallback to camera1 API for internal cameras that don't have full support.
                // This should help with legacy situations where using the camera2 API causes
                // distorted or otherwise broken previews.
                useCamera2API =
                        (facing == CameraCharacteristics.LENS_FACING_EXTERNAL)
                                || isHardwareLevelSupported(
                                characteristics, CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL);
                LOGGER.i("Camera API lv2?: %s", useCamera2API);
                return cameraId;
            }
        } catch (CameraAccessException e) {
            LOGGER.e(e, "Not allowed to access camera");
        }

        return null;
    }

    protected void setFragment() {
        String cameraId = chooseCamera();

        Fragment fragment;
        if (useCamera2API) {
            CameraConnectionFragment camera2Fragment =
                    CameraConnectionFragment.newInstance((size, rotation) -> {
                                previewHeight = size.getHeight();
                                previewWidth = size.getWidth();
                                CameraActivity.this.onPreviewSizeChosen(size, rotation);
                            },
                            this,
                            getLayoutId(),
                            getDesiredPreviewFrameSize());

            camera2Fragment.setCamera(cameraId);
            fragment = camera2Fragment;
        } else {
            fragment =
                    new LegacyCameraConnectionFragment(this, getLayoutId(), getDesiredPreviewFrameSize());
        }

        getFragmentManager().beginTransaction().replace(R.id.container, fragment).commit();
    }

    protected void fillBytes(final Plane[] planes, final byte[][] yuvBytes) {
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (int i = 0; i < planes.length; ++i) {
            final ByteBuffer buffer = planes[i].getBuffer();
            if (yuvBytes[i] == null) {
                LOGGER.d("Initializing buffer %d at size %d", i, buffer.capacity());
                yuvBytes[i] = new byte[buffer.capacity()];
            }
            buffer.get(yuvBytes[i]);
        }
    }

    public boolean isDebug() {
        return debug;
    }

    protected void readyForNextImage() {
        if (postInferenceCallback != null) {
            postInferenceCallback.run();
        }
    }

    protected int getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }

    @UiThread
    protected void showResultsInBottomSheet(List<Classifier2.Recognition> results) {
        if (results != null && results.size() >= 3) {
            String title = null;
            Classifier2.Recognition recognition = results.get(0);
            if (recognition != null) {
                title = recognition.getTitle();
                if (title != null && title.contains("Crying"))
                    recognitionTextView.setText(title);
                else if (title != null)
                    recognitionTextView.setText("Not Crying");

                if (recognition.getConfidence() != null)
                    recognitionValueTextView.setText(
                            String.format("%.2f", (100 * recognition.getConfidence())) + "%");
            }

            Classifier2.Recognition recognition1 = results.get(1);
            Classifier2.Recognition recognition2 = results.get(2);
            if (recognition1 != null && recognition2 != null) {
                title = recognition1.getTitle();
                if (title != null && title.contains("Crying"))
                    recognition1TextView.setText(recognition1.getTitle());
                else if (title != null && recognition.getTitle().contains("Crying")) {
                    recognition1TextView.setText("Not Crying");
                    if (recognition1.getConfidence() != null)
                        recognition1ValueTextView.setText(
                                String.format("%.2f", (100 * recognition1.getConfidence())) + "%");
                } else {
                    title = recognition2.getTitle();
                    recognition1TextView.setText(title);
                    if (recognition2.getConfidence() != null)
                        recognition1ValueTextView.setText(
                                String.format("%.2f", (100 * recognition2.getConfidence())) + "%");
                }
            }
        }
    }

    @Override
    public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
    }

    @Override
    public void onClick(View v) {
    }

    protected void showInference(String inferenceTime) {
        inferenceTimeTextView.setText(inferenceTime);
    }

    protected void showDB(String db) {
        dbTextView.setText(db);
    }

    protected abstract void processImage();

    protected abstract void onPreviewSizeChosen(final Size size, final int rotation);

    protected abstract int getLayoutId();

    protected abstract Size getDesiredPreviewFrameSize();
}
