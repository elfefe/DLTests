package server

import io.ktor.routing.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import models.Epoch
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import utils.TrainStatus
import utils.extensions.configure
import utils.extensions.modelsDir
import utils.extensions.timestamp
import java.io.File

class NeuralNetworkServer(val scope: CoroutineScope) {
    val fashionMnist = org.jetbrains.kotlinx.dl.dataset.fashionMnist()
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

    private var epochs = 20

    private val epochHistory = mutableListOf<Epoch>()
    private var summary = ""

    init {
        embeddedServer(Netty, port = 9310) {
            routing {
                get("/") {
                    
                }
            }
        }.start(wait = true)
    }

    suspend fun defaultModel() = Sequential.of(
        Input(
            28, 28, 1
        ),
        Flatten(),
        Dense(300),
        Dense(100),
        Dense(10)
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

    suspend fun trainModel() =
        defaultModel().use {
            val (train, test) = fashionMnist

            epochHistory.clear()

            it.fit(
                dataset = train,
                epochs = epochs,
                batchSize = 100
            )

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
            }
        }
}