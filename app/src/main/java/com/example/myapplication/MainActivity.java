package com.example.myapplication;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.example.myapplication.ml.Model;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.DecimalFormat;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    Button camera, gallery;
    ImageView imageView;
    TextView result;
    int imageSize = 128;
    Yolov5TFLiteDetector detector;
    Paint boxPaint= new Paint();
    Paint textPaint= new Paint();
    Bitmap bitmap;
    private final String[] classes = {"Spoiled", "Half-Fresh", "Fresh"};
    private final int[] colors= {Color.RED, Color.YELLOW, Color.GREEN};



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera= findViewById(R.id.button);
        gallery= findViewById(R.id.button2);

        imageView= findViewById(R.id.imageView);

        detector= new Yolov5TFLiteDetector();
        detector.setModelFile("yolov5.tflite");
        detector.initialModel(this);

        boxPaint.setStrokeWidth(5);
        boxPaint.setStyle(Paint.Style.STROKE);

        textPaint.setTextSize(32);
        textPaint.setStyle(Paint.Style.FILL);

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent= new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 1);
            }
        });

    }
    public void predict() {
        ArrayList<Recognition> recognitions= detector.detect(bitmap);
        Bitmap mutable= bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas= new Canvas(mutable);
        for (Recognition recognition: recognitions) {
            if (recognition.getConfidence() > 0.4 && recognition.getLabelId() == 1) {
                RectF location = recognition.getLocation();
                int left = Math.round(location.left);
                int top = Math.round(location.top);
                int right = Math.round(location.right);
                int bottom = Math.round(location.bottom);

                // Ensure coordinates are within bounds of the image
                left = Math.max(left, 0);
                top = Math.max(top, 0);
                right = Math.min(right, bitmap.getWidth());
                bottom = Math.min(bottom, bitmap.getHeight());

                // Calculate width and height of the cropped region
                int width = right - left;
                int height = bottom - top;

                // Crop the image using the calculated coordinates
                Bitmap croppedImage = Bitmap.createBitmap(bitmap, left, top, width, height);
                croppedImage = Bitmap.createScaledBitmap(croppedImage, imageSize, imageSize, false);
                int n= classifyImage(croppedImage);
                boxPaint.setColor(colors[n]);
                textPaint.setColor(colors[n]);
                canvas.drawRect(location,boxPaint);
            }
        }
        imageView.setImageBitmap(mutable);
    }


    public int classifyImage(Bitmap image) {
        int maxPos = 0;
        try {
            Model model = Model.newInstance(getApplicationContext());
            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 128, 128, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for(int i = 0; i < imageSize; i ++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
        return maxPos;
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        Log.d("ActivityResultCheck", "onActivityResult is being called");
        if (resultCode == RESULT_OK) {
            Uri dat = data.getData();
            if (requestCode == 3) {
                bitmap = (Bitmap) data.getExtras().get("data");
            } else {
                bitmap = null;
                try {
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            predict();
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}