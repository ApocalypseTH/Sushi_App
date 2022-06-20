package com.example.sushi;

import static org.opencv.imgproc.Imgproc.FONT_HERSHEY_SIMPLEX;

import androidx.appcompat.app.AppCompatActivity;

import android.net.Uri;
import android.os.Environment;
//import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Toast;
import android.content.Context;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import org.opencv.dnn.Dnn;
import org.opencv.utils.Converters;


import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;
    boolean startYolo = false;
    boolean firstTimeYolo = false;
//    Net tinyYolo;
    Net yolo;
    Context context;
    List<String> labels = Arrays.asList("ebi maki", "ebi nigiri", "unagi nigiri", "maguro maki", "maguro nigiri", "sake maki", "sake nigiri", "suzuki nigiri", "tako nigiri", "edamame", "wakame", "gyoza", "shao mai", "tempura", "temaki");

    int INPUT_WIDTH = 640;
    int INPUT_HEIGHT = 640;
    float SCORE_THRESHOLD = 0.2f;
    float NMS_THRESHOLD = 0.2f;
    float CONFIDENCE_THRESHOLD = 0.4f;


    public MatOfByte getYolo(){
        InputStream inputStream = null;
        MatOfByte weights = new MatOfByte();

        try {
            inputStream = new BufferedInputStream(this.getAssets().open("yolov5s.onnx"));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            weights.fromArray(data);
        } catch (IOException e) {
            e.printStackTrace();
        }

        return weights;
    }

    public void YOLO(View Button){

        if (startYolo == false){




            startYolo = true;

            if (firstTimeYolo == false){


                firstTimeYolo = true;
//                String p = context.getFilesDir().getAbsolutePath() + "/raw/";
//
//                String yoloPath = p + "yolov5s.onnx";
                yolo = Dnn.readNetFromONNX(getYolo());


            }



        }

        else{

            startYolo = false;


        }




    }






    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        context = getApplicationContext();


        cameraBridgeViewBase = (JavaCameraView)findViewById(R.id.CameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);


        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);

                switch(status){

                    case BaseLoaderCallback.SUCCESS:
                        cameraBridgeViewBase.enableView();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }


            }

        };




    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        Mat frame = inputFrame.rgba();
        System.out.println("camera frame");

        if (startYolo == true) {

            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
//          QUI preprocessing immagine


            Mat imageBlob = Dnn.blobFromImage(frame, 1/255.0, new Size(INPUT_WIDTH, INPUT_HEIGHT), new Scalar(0, 0, 0),/*swapRB*/false, /*crop*/false);


            yolo.setInput(imageBlob);



            java.util.List<Mat> result = new java.util.ArrayList<Mat>(2);

//            List<String> outBlobNames = new java.util.ArrayList<>();
//            outBlobNames.add(0, "yolo_16");
//            outBlobNames.add(1, "yolo_23");
//
//            yolo.forward(result,outBlobNames)
            yolo.forward(result, yolo.getUnconnectedOutLayersNames());

//            System.out.println(result);
//            float confThreshold = 0.3f;



            List<Integer> clsIds = new ArrayList<>();
            List<Float> confs = new ArrayList<>();
            List<Rect2d> rects = new ArrayList<>();


            for (int i = 0; i < result.size(); ++i)
            {

                Mat level = result.get(i);

                for (int j = 0; j < level.rows(); ++j)
                {
                    Mat row = level.row(j);
                    Mat scores = row.colRange(5, level.cols());

                    Core.MinMaxLocResult mm = Core.minMaxLoc(scores);




                    float confidence = (float)mm.maxVal;


                    Point classIdPoint = mm.maxLoc;



                    if (confidence > CONFIDENCE_THRESHOLD)
                    {
                        int centerX = (int)(row.get(0,0)[0] * frame.cols());
                        int centerY = (int)(row.get(0,1)[0] * frame.rows());
                        int width   = (int)(row.get(0,2)[0] * frame.cols());
                        int height  = (int)(row.get(0,3)[0] * frame.rows());


                        int left    = centerX - width  / 2;
                        int top     = centerY - height / 2;

                        clsIds.add((int)classIdPoint.x);
                        confs.add((float)confidence);




                        rects.add(new Rect2d(left, top, width, height));
                    }
                }
            }
            int ArrayLength = confs.size();

            if (ArrayLength>=1) {
                // Apply non-maximum suppression procedure.
//                float nmsThresh = 0.2f;




                MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));


                Rect2d[] boxesArray = rects.toArray(new Rect2d[0]);

                MatOfRect2d boxes = new MatOfRect2d(boxesArray);

                MatOfInt indices = new MatOfInt();



                Dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);


                // Draw result boxes:
                int[] ind = indices.toArray();
                for (int i = 0; i < ind.length; ++i) {

                    int idx = ind[i];
                    Rect2d box = boxesArray[idx];

                    int idGuy = clsIds.get(idx);

                    float conf = confs.get(idx);


                    int intConf = (int) (conf * 100);



                    Imgproc.putText(frame,labels.get(idGuy) + " " + intConf + "%", box.tl(), FONT_HERSHEY_SIMPLEX, 2, new Scalar(255,255,0),2);



                    Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(255, 0, 0), 2);





                }
            }









        }



        return frame;
    }


    @Override
    public void onCameraViewStarted(int width, int height) {


        if (startYolo == true){

//            String tinyYoloCfg = Environment.getExternalStorageDirectory() + "/dnns/yolov3-tiny.cfg" ;
//            String tinyYoloWeights = Environment.getExternalStorageDirectory() + "/dnns/yolov3-tiny.weights";

//            tinyYolo = Dnn.readNetFromDarknet(tinyYoloCfg, tinyYoloWeights);

//            String yoloPath = "file:///android_asset/" + "yolov5s.onnx";
            yolo = Dnn.readNetFromONNX(getYolo());
        }



    }


    @Override
    public void onCameraViewStopped() {

    }


    @Override
    protected void onResume() {
        super.onResume();

        if (!OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(),"There's a problem, yo!", Toast.LENGTH_SHORT).show();
        }

        else
        {
            baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
        }



    }

    @Override
    protected void onPause() {
        super.onPause();
        if(cameraBridgeViewBase!=null){

            cameraBridgeViewBase.disableView();
        }

    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }
}
