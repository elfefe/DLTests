package utils.extensions


import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.callback.Callback
import org.jetbrains.kotlinx.dl.api.core.history.EpochTrainingEvent
import org.jetbrains.kotlinx.dl.api.core.history.TrainingHistory
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.summary.ModelSummary
import org.jetbrains.kotlinx.dl.api.core.summary.format

class ModelCallback(val onEpochEnd: (Int, Double) -> Unit) : Callback() {
    override fun onEpochEnd(epoch: Int, event: EpochTrainingEvent, logs: TrainingHistory) {
        super.onEpochEnd(epoch, event, logs)
        onEpochEnd(epoch, event.lossValue)
    }
}

fun Sequential.configure(
    onSummary: (String) -> Unit,
    onCallback: (Int, Double) -> Unit
) {
    compile(
        optimizer = Adam(),
        loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
        metric = Metrics.ACCURACY,
        callback = ModelCallback { epoch, loss ->
            onCallback(epoch, loss)
        }
    )

    onSummary(summary().text())
}

fun ModelSummary.text(): String = format().run {
    var text = ""
    forEach {
        if (!it.contains("="))
            text +="$it\n"
    }
    text
}