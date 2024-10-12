import data.MNISTLoader
import model.Layer
import model.NeuralNetwork
import utils.ActivationFunctions
import java.io.File

fun main() {
    // 784 -> 128 -> 10)
    val network = NeuralNetwork(listOf(
        Layer(784, 128, ActivationFunctions::relu),
        Layer(128, 10, ActivationFunctions::sigmoid)
    ))

//    File(".").listFiles()?.forEach {
//        println(it)
//    }
    val (trainInputs, trainTargets) = MNISTLoader.loadMNISTData("src/train-images.idx3-ubyte", "src/train-labels.idx1-ubyte")
    val (testInputs, testTargets) = MNISTLoader.loadMNISTData("src/t10k-images.idx3-ubyte", "src/t10k-labels.idx1-ubyte")

    // Train
    network.trainWithBatches(trainInputs, trainTargets, epochs = 10, batchSize = 32, learningRate = 0.001)

    // Evaluate
    var correct = 0
    for (i in testInputs.indices) {
        val prediction = network.predict(testInputs[i])
        val actualLabel = when (val target = testTargets[i]) {
            else -> target.indexOfFirst { it > 0.5 }
        }
        println("Prediction: $prediction, Actual: $actualLabel")
        if (prediction == actualLabel) correct++
    }
    val predictionCounts = IntArray(10) { 0 }
    for (input in testInputs) {
        val prediction = network.predict(input)
        predictionCounts[prediction]++
    }
    println("Prediction distribution: ${predictionCounts.joinToString()}")
    println("Accuracy: ${correct.toDouble() / testInputs.size}")
}