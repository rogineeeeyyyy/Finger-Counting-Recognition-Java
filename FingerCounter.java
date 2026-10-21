import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;

public class FingerCounter {

public static void main(String[] args) {

    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

    VideoCapture camera = new VideoCapture(0);

    if (!camera.isOpened()) {

        System.out.println("Unable to open webcam.");
        return;
    }

    Mat frame = new Mat();
    Mat hsv = new Mat();
    Mat mask = new Mat();

    System.out.println("Finger Counting Recognition Started");
    System.out.println("Press ESC to close the camera.");

    while (camera.read(frame)) {

        if (frame.empty()) {
            break;
        }

        // Convert the webcam image to HSV
        Imgproc.cvtColor(
            frame,
            hsv,
            Imgproc.COLOR_BGR2HSV
        );

        // Approximate skin-color detection
        Core.inRange(
            hsv,
            new Scalar(0, 20, 70),
            new Scalar(20, 255, 255),
            mask
        );

        // Find contours
        java.util.List<MatOfPoint> contours =
            new java.util.ArrayList<>();

        Imgproc.findContours(
            mask,
            contours,
            new Mat(),
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        );

        double largestArea = 0;
        MatOfPoint largestContour = null;

        for (MatOfPoint contour : contours) {

            double area = Imgproc.contourArea(contour);

            if (area > largestArea) {
                largestArea = area;
                largestContour = contour;
            }
        }

        if (largestContour != null && largestArea > 3000) {

            Rect box = Imgproc.boundingRect(largestContour);

            Imgproc.rectangle(
                frame,
                box.tl(),
                box.br(),
                new Scalar(0, 255, 0),
                2
            );

            Imgproc.putText(
                frame,
                "Hand Region Detected",
                new Point(box.x, Math.max(30, box.y - 10)),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.7,
                new Scalar(0, 255, 0),
                2
            );
        }

        HighGui.imshow("Finger Counting Recognition", frame);

        if (HighGui.waitKey(30) == 27) {
            break;
        }
    }

    camera.release();
    frame.release();
    hsv.release();
    mask.release();

    HighGui.destroyAllWindows();

    System.out.println("Application Closed");
}

}
