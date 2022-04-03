import nu.pattern.OpenCV
import org.opencv.core.*
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

    val kernel = gaussianKernel()
    kernel.print()

    val img = Imgcodecs.imread("./Pictures/skate.webp", Imgcodecs.IMREAD_COLOR)
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
    // az eredeti kép szürkeskálássá alakítása
    val graySrc = Mat()
    Imgproc.cvtColor(src, graySrc, Imgproc.COLOR_BGR2GRAY)

    // a transzformációhoz optimális képméret meghatározása (a gyorsaság miatt szükséges)
    val paddedSrc = Mat()
    val m = Core.getOptimalDFTSize(graySrc.rows())
    val n = Core.getOptimalDFTSize(graySrc.cols())
    // a szegélyen lévő új elemek kitöltése 0 értékekkel
    Core.copyMakeBorder(
        graySrc,
        paddedSrc,
        0,
        m - graySrc.rows(),
        0,
        n - graySrc.cols(),
        Core.BORDER_CONSTANT,
        Scalar.all(0.0)
    )

    // komplex komponensek létrehozása
    val planes: MutableList<Mat> = ArrayList()
    paddedSrc.convertTo(paddedSrc, CvType.CV_32F)
    planes.add(paddedSrc)
    planes.add(Mat.zeros(paddedSrc.size(), CvType.CV_32F))

    // komplex kép összeállítása
    val complexSrc = Mat()
    Core.merge(planes, complexSrc)

    // Fourier transzformáció
    Core.dft(complexSrc, complexSrc)

    // **** SZŰRÉS ****

    // szűrő kép létrehozása (ugyanolyan méretű és típusú kell legyen, mint a complexSrc)
    // HIÁNYZÓ RÉSZ: csupa 1 érték helyett a megfelelő Gauss értékek legyenek a szűrő képben

    // Próba 1: csupa 1 érték, az végső kép változatlan
    val filterImg = Mat.ones(complexSrc.size(), complexSrc.type())

    // Próba 2: csupa 0 érték, az végső kép teljesen üres (fekete)
    //val filterImg = Mat.ones(complexSrc.size(), complexSrc.type())

    // Próba 3: csupa 1 érték középen, csupa 0 érték azon kívül
//    val filterImg = Mat.zeros(complexSrc.size(), complexSrc.type())
//    for (i in 1..filterImg.rows())
//        for (j in 1..filterImg.cols())
//            if (i > floor(filterImg.rows() * 0.25) && i < floor(filterImg.rows() * 0.75))
//                if (j > floor(filterImg.cols() * 0.25) && j < floor(filterImg.cols() * 0.75))
//                    filterImg.put(i, j, 1.0, 0.0)

    // a transzformált kép és a szűrő kép elemenként történő összeszorzása
    Core.mulSpectrums(complexSrc, filterImg, complexSrc, 0)

    // **** SZŰRÉS vége ****

    // inverz Fourier transzformáció
    Core.idft(complexSrc, complexSrc)

    // komplex komponensek kiemelése
    Core.split(complexSrc, planes)

    // valós értékek normalizálása
    var restoredSrc = Mat()
    Core.normalize(planes[0], restoredSrc, 0.0, 255.0, Core.NORM_MINMAX)

    // szegélyek eldobása
    restoredSrc = restoredSrc.submat(Rect(0, 0, src.cols(), src.rows()))

    // a normalizált kép visszalakaítása szürkeskálássá
    restoredSrc.convertTo(restoredSrc, CvType.CV_8U)

    return restoredSrc
}

fun gaussianKernel(): Mat {
    val kernel = Imgproc.getGaussianKernel(3, 0.0)
    val kernelT = Mat()
    Core.transpose(kernel, kernelT)

    val product = Mat()
    Core.gemm(kernel, kernelT, 1.0, Mat(), 0.0, product, 0)
    return product
}

fun Mat.print() {
    for (i in 0 until rows()) {
        for (j in 0 until cols()) {
            val value = row(i).get(0, j).first()
            print("$value${if (j == cols() - 1) "" else ", "}")
        }
        println()
    }
}