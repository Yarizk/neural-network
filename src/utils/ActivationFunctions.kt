package utils

import kotlin.math.exp

object ActivationFunctions {
    fun relu(x: Double): Double = if (x > 0) x else 0.0
    fun sigmoid(x: Double): Double = 1 / (1 + exp(-x))
    fun softmax(x: DoubleArray): DoubleArray {
        val expValues = x.map { exp(it) }
        val sum = expValues.sum()
        return expValues.map { it / sum }.toDoubleArray()
    }
}