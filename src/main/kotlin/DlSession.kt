import androidx.compose.runtime.mutableStateListOf
import common.configure
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import model.Epoch
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import java.io.File

class DlSession {
    val scope = CoroutineScope(Dispatchers.Default)

    val epochHistory = mutableListOf<Epoch>()
    var summary = ""

    var model = configuration()

    var epoch = 0

    val stringLabels = mapOf(
        0 to "T-shirt/top",
        1 to "Trousers",
        2 to "Pullover",
        3 to "Dress",
        4 to "Coat",
        5 to "Sandals",
        6 to "Shirt",
        7 to "Sneakers",
        8 to "Bag",
        9 to "Ankle boots"
    )

    init {

    }

    fun configuration() = Sequential.of(
        Input(
            28, 28, 1
        ),
        Flatten(),
        Dense(outputSize = 300, activation = Activations.Relu),
        Dense(100, activation = Activations.Relu),
        Dense(10, activation = Activations.Sigmoid)
    ).apply {
        var lastTime = timestamp
        configure(
            onSummary = {
                summary = it
            },
            onCallback = { epoch, loss ->
                val time = timestamp
                epochHistory.add(Epoch(epoch, loss, (time - lastTime) / 1000f))
                lastTime = time
            }
        )
    }

    fun train(train: OnHeapDataset, test: OnHeapDataset) {
        scope.launch {
            model.use {
//            trainButtonEnabled = true
                epochHistory.clear()

                it.fit(
                    dataset = train,
                    epochs = epoch,
                    batchSize = 100
                )

//            trainButtonText = TrainStatus.saving

                val accuracy =
                    it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
                val fileIndex = modelsDir.list()?.size

                withContext(Dispatchers.IO) {
                    val file =
                        File(
                            "${
                                modelsDir.path
                            }/${
                                it.summary().type
                            }_${
                                fileIndex ?: 0
                            }_${
                                accuracy.toString().apply { removeRange(5, length - 1) }
                            }".trim()
                        )
                    it.save(file)

//                modelItems = modelsDir.listFiles()!!.toMutableList()
                }
            }
        }
    }
}