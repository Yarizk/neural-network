package utils

class ProgressTracker(private val totalEpochs: Int, private val totalBatches: Int) {
    private var currentEpoch = 0
    private var currentBatch = 0
    private var startTime = System.currentTimeMillis()

    fun updateEpoch(epoch: Int, loss: Double) {
        currentEpoch = epoch
        currentBatch = 0
        val elapsedTime = (System.currentTimeMillis() - startTime) / 1000
        println("Epoch $currentEpoch/$totalEpochs completed. Loss: $loss. Time elapsed: $elapsedTime seconds")
    }

    fun updateBatch(batch: Int) {
        currentBatch = batch
        if (currentBatch % (totalBatches / 10) == 0) {
            val progress = (currentEpoch * totalBatches + currentBatch).toDouble() / (totalEpochs * totalBatches) * 100
            println("Progress: ${"%.2f".format(progress)}% (Epoch $currentEpoch, Batch $currentBatch/$totalBatches)")
        }
    }
}