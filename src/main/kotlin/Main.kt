import nu.pattern.OpenCV
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.highgui.HighGui
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.awt.Dimension
import java.awt.Toolkit
import kotlin.system.exitProcess

const val WINDOW_NAME_ORIGINAL = "Image (original)"
const val WINDOW_NAME_PROCESSED = "Image (GaussianBlur)"

fun main() {
    OpenCV.loadLocally()

    val screenDimension = Toolkit.getDefaultToolkit().screenSize
    initWindows(screenDimension)

    val img = Imgcodecs.imread("./Pictures/dog.jpg", Imgcodecs.IMREAD_COLOR)
    processImage(img, screenDimension)

    HighGui.waitKey()
    HighGui.destroyAllWindows()

    exitProcess(0)
}

fun initWindows(
    screenDimension: Dimension
) {
    HighGui.namedWindow(WINDOW_NAME_ORIGINAL, HighGui.WINDOW_NORMAL)
    HighGui.resizeWindow(WINDOW_NAME_ORIGINAL, screenDimension.width / 2, screenDimension.height / 2)
    HighGui.moveWindow(WINDOW_NAME_ORIGINAL, 0, 0)

    HighGui.namedWindow(WINDOW_NAME_PROCESSED, HighGui.WINDOW_NORMAL)
    HighGui.resizeWindow(WINDOW_NAME_PROCESSED, screenDimension.width / 2, screenDimension.height / 2)
    HighGui.moveWindow(WINDOW_NAME_PROCESSED, 0, screenDimension.height / 2)
}

fun processImage(
    src: Mat,
    screenDimension: Dimension
) {
    resizeImage(src, screenDimension.width / 2, screenDimension.height / 2)
    HighGui.imshow(WINDOW_NAME_ORIGINAL, src)

    val blurImg = gaussianBlur(src)
    HighGui.imshow(WINDOW_NAME_PROCESSED, blurImg)
}

fun resizeImage(
    src: Mat,
    maxWidth: Int,
    maxHeight: Int
) {
    val scale = (maxWidth / src.width().toDouble())
        .coerceAtMost(maxHeight / src.height().toDouble())
        .coerceAtMost(1.0)
    Imgproc.resize(src, src, Size(0.0, 0.0), scale, scale)
}

fun gaussianBlur(
    src: Mat
): Mat {
    val dst = Mat()

    Imgproc.GaussianBlur(src, dst, Size(3.0, 3.0), 0.0)
    return dst
}