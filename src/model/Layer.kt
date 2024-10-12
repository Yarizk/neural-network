package model

import kotlin.random.Random

data class Layer(val inputSize: Int, val outputSize: Int, val activation: (Double) -> Double) {
    var weights: Array<DoubleArray> = Array(outputSize) { DoubleArray(inputSize) { Random.nextDouble(-1.0, 1.0) } }
    var biases: DoubleArray = DoubleArray(outputSize) { Random.nextDouble(-1.0, 1.0) }
    lateinit var output: DoubleArray
    lateinit var input: DoubleArray
}