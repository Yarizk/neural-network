package data

import java.io.File
import java.io.DataInputStream

object MNISTLoader {
    fun loadMNISTData(imagesFile: String, labelsFile: String): Pair<List<DoubleArray>, List<DoubleArray>> {
        val images = loadMNISTImages(imagesFile)
        val labels = loadMNISTLabels(labelsFile)

        val normalizedImages = images.map { it.map { pixel -> pixel.toDouble() / 255.0 }.toDoubleArray() }
        val encodedLabels = labels.map { label ->
            DoubleArray(10) { i -> if (i == label) 1.0 else 0.0 }
        }

        return Pair(normalizedImages, encodedLabels)
    }

    private fun loadMNISTImages(filename: String): List<ByteArray> {
        DataInputStream(File(filename).inputStream()).use { inputStream ->
            val magicNumber = inputStream.readInt()
            val numberOfImages = inputStream.readInt()
            val numberOfRows = inputStream.readInt()
            val numberOfColumns = inputStream.readInt()

            return (1..numberOfImages).map {
                ByteArray(numberOfRows * numberOfColumns).also { imageData ->
                    inputStream.readFully(imageData)
                }
            }
        }
    }

    private fun loadMNISTLabels(filename: String): List<Int> {
        DataInputStream(File(filename).inputStream()).use { inputStream ->
            val magicNumber = inputStream.readInt()
            val numberOfLabels = inputStream.readInt()

            return (1..numberOfLabels).map {
                inputStream.readUnsignedByte()
            }
        }
    }
}