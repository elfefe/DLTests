// Copyright 2000-2021 JetBrains s.r.o. and contributors. Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE file.
import androidx.compose.desktop.DesktopMaterialTheme
import androidx.compose.desktop.ui.tooling.preview.Preview
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material.Button
import androidx.compose.material.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.ExperimentalUnitApi
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.TextUnitType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.callback.Callback
import org.jetbrains.kotlinx.dl.api.core.history.EpochTrainingEvent
import org.jetbrains.kotlinx.dl.api.core.history.TrainingHistory
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.summary.format
import org.jetbrains.kotlinx.dl.dataset.fashionMnist
import java.awt.Dimension
import java.io.File
import java.util.*

val timestamp: Long
    get() = Date().time

val modelsDir: File = File("models")

object ButtonStatus {
    const val train = "train"
    const val stop = "stop"
    const val configuring = "configuring ..."
    const val saving = "saving ..."
}

fun main() = application {
    Window(onCloseRequest = ::exitApplication) {
        App(window.size)
    }
}

@OptIn(ExperimentalUnitApi::class)
@Composable
@Preview
fun App(size: Dimension) {

    val scope = rememberCoroutineScope()

    var text by remember { mutableStateOf(ButtonStatus.configuring) }
    var buttonText by remember { mutableStateOf(ButtonStatus.train) }
    val epochHistory = mutableStateListOf<Epoch>()

    val epochState = rememberLazyListState(0)

    val epoch = 20

    var isTraining = false

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

    val (train, test) = fashionMnist()
    fun defaultModel() = Sequential.of(
        Input(28, 28, 1),
        Flatten(),
        Dense(300),
        Dense(100),
        Dense(10)
    ).apply {
        var lastTime = timestamp
        configure(
            onSummary = {
                text = it
            },
            onCallback = { epoch, loss ->
                val time = timestamp
                epochHistory.add(Epoch(epoch, loss, (time - lastTime) / 1000f))
                lastTime = time
            }
        )
    }

    var model = defaultModel()

    DesktopMaterialTheme {
        Row(
            Modifier.fillMaxSize()
        ) {

            Column(
                Modifier
                    .padding(10.dp)
                    .fillMaxHeight()
            ) {
                Row(
                    horizontalArrangement = Arrangement.Start
                ) {
                    Text(
                        modifier = Modifier,
                        text = text,
                        fontSize = TextUnit(10f, TextUnitType.Sp),
                        fontWeight = FontWeight.Bold
                    )
                }
                LazyColumn(
                    Modifier
                        .fillMaxHeight()
                        .padding(10.dp),
                    state = epochState
                ) {
                    scope.launch {
                        if (epochState.layoutInfo.visibleItemsInfo.size != epochState.layoutInfo.totalItemsCount) {
                            epochState.scrollToItem(epochState.layoutInfo.totalItemsCount - 1)
                        }
                    }
                    items(epochHistory) {
                        Text("${it.epoch}: ${it.loss}  ${it.time}s")
                    }
                }
            }
            Column(
                Modifier
                    .fillMaxHeight()
                    .padding(10.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Canvas(
                    Modifier
                        .fillMaxWidth()
                        .fillMaxSize(.4f)
                ) {
                    drawRect(Color.Black, Offset.Zero, this.size)
                    println("----------------------")
                    for (i in epochHistory.indices) {
                        val x = (i / (epoch - 1f)) * this.size.width
                        val y = this.size.height - (this.size.height * epochHistory[i].loss.toFloat())
                        val destX =
                            if (epochHistory.indices.last == 0) 0f
                            else if (i < epochHistory.indices.last) ((i + 1) / (epoch - 1f)) * this.size.width
                            else x
                        val destY = if (i < epochHistory.indices.last)
                            this.size.height - (this.size.height * epochHistory[i + 1].loss.toFloat()) else y

                        if (i == 0) drawCircle(Color(255, 174, 0), 1f, Offset(destX, destY))
                        drawLine(Color(255, 174, 0), Offset(x, y), Offset(destX, destY))

                        println("$i - ${Offset(destX, destY)} (${epoch - 1}) / ${this.size.width}*${this.size.height}")
                    }
                }

                Spacer(Modifier.height(10.dp))

                Button(
                    onClick = {
                        if (!isTraining) {
                            if (model.stopTraining) {
                                model.stopTraining = false
                                epochHistory.clear()
                                buttonText = ButtonStatus.configuring
                                model = defaultModel()
                            }
                            buttonText = ButtonStatus.stop
                            scope.launch(Dispatchers.Default) {
                                model.use {

                                    it.fit(
                                        dataset = train,
                                        epochs = epoch,
                                        batchSize = 100
                                    )

                                    buttonText = ButtonStatus.saving

                                    val accuracy =
                                        it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
                                    val fileIndex = modelsDir.list()?.size

                                    withContext(Dispatchers.IO) {
                                        val file =
                                            File("${modelsDir.path}/${it.summary().type}_${fileIndex ?: 0}_$accuracy")
                                        it.save(file)
                                    }
                                }

                                buttonText = ButtonStatus.train
                            }
                        } else {
                            model.stopTraining = true
                            buttonText = ButtonStatus.train
                        }
                        isTraining = !isTraining
                    },
                    enabled = buttonText == ButtonStatus.train || buttonText == ButtonStatus.stop
                ) {
                    Text(buttonText)
                }
            }
        }
    }
}

data class Epoch(
    val epoch: Int,
    val loss: Double,
    val time: Float
)

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
    var text = ""
    summary().format().forEach {
        if (!it.contains("="))
            text += "$it\n"
    }
    onSummary(text)
}