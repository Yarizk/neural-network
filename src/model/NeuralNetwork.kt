package model

import utils.ProgressTracker

class NeuralNetwork(private val layers: List<Layer>) {
    fun forward(input: DoubleArray): DoubleArray {
        var currentInput = input
        for (layer in layers) {
            layer.input = currentInput
            layer.output = DoubleArray(layer.outputSize) { 0.0 }
            for (i in 0 until layer.outputSize) {
                var sum = layer.biases[i]
                for (j in currentInput.indices) {
                    sum += currentInput[j] * layer.weights[i][j]
                }
                layer.output[i] = layer.activation(sum)
            }
            currentInput = layer.output
        }
        return currentInput
    }

    fun backward(input: DoubleArray, target: DoubleArray, learningRate: Double) {
        val output = forward(input)
        var delta = output.zip(target) { a, b -> a - b }.toDoubleArray()

        for (i in layers.indices.reversed()) {
            val layer = layers[i]
            val newDelta = DoubleArray(layer.inputSize) { 0.0 }

            for (j in 0 until layer.outputSize) {
                layer.biases[j] -= learningRate * delta[j]
                for (k in 0 until layer.inputSize) {
                    layer.weights[j][k] -= learningRate * delta[j] * layer.input[k]
                    newDelta[k] += delta[j] * layer.weights[j][k]
                }
            }

            if (i > 0) {
                delta = newDelta.zip(layers[i-1].output) { a, b ->
                    a * if (b > 0) 1.0 else 0.0 // Derivative of ReLU
                }.toDoubleArray()
            }
        }
    }

    fun trainWithBatches(inputs: List<DoubleArray>, targets: List<DoubleArray>, epochs: Int, batchSize: Int, learningRate: Double) {
        val numBatches = inputs.size / batchSize
        val progressTracker = ProgressTracker(epochs, numBatches)

        for (epoch in 0 until epochs) {
            var totalLoss = 0.0

            val shuffled = inputs.zip(targets).shuffled()

            for (batchStart in 0 until inputs.size step batchSize) {
                val batchEnd = minOf(batchStart + batchSize, inputs.size)
                val batchInputs = shuffled.subList(batchStart, batchEnd).map { it.first }
                val batchTargets = shuffled.subList(batchStart, batchEnd).map { it.second }

                var batchLoss = 0.0
                batchInputs.zip(batchTargets).forEach { (input, target) ->
                    val output = forward(input)
                    backward(input, target, learningRate)
                    batchLoss += calculateLoss(output, target)
                }

                totalLoss += batchLoss
                progressTracker.updateBatch(batchStart / batchSize)
            }

            val averageLoss = totalLoss / inputs.size
            progressTracker.updateEpoch(epoch + 1, averageLoss)
        }
    }

    fun predict(input: DoubleArray): Int {
        val output = forward(input)
        return output.indices.maxByOrNull { output[it] } ?: -1
    }

    private fun calculateLoss(output: DoubleArray, target: DoubleArray): Double {
        return output.zip(target) { a, b -> (a - b) * (a - b) }.sum() / output.size
    }
}
