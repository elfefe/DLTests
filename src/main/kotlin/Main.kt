// Copyright 2000-2021 JetBrains s.r.o. and contributors. Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE file.
import androidx.compose.desktop.DesktopMaterialTheme
import androidx.compose.desktop.ui.tooling.preview.Preview
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.*
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.ExperimentalUnitApi
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.TextUnitType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.AwtWindow
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import common.configure
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import model.Epoch
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.callback.Callback
import org.jetbrains.kotlinx.dl.api.core.history.EpochTrainingEvent
import org.jetbrains.kotlinx.dl.api.core.history.TrainingHistory
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.summary.ModelSummary
import org.jetbrains.kotlinx.dl.api.core.summary.format
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.jetbrains.kotlinx.dl.dataset.fashionMnist
import java.awt.Dimension
import java.awt.FileDialog
import java.awt.Frame
import java.awt.image.BufferedImage
import java.awt.image.ColorConvertOp
import java.io.File
import java.util.*
import javax.imageio.ImageIO

val timestamp: Long
    get() = Date().time

val modelsDir: File = File("models").apply { createNewFile() }

object TrainStatus {
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

    var summary by remember { mutableStateOf(TrainStatus.configuring) }

    var trainButtonText by remember { mutableStateOf(TrainStatus.train) }
    var trainButtonEnabled by remember { mutableStateOf(true) }

    var predictButtonText by remember { mutableStateOf("Predict") }
    var predictButtonClicked by remember { mutableStateOf(false) }

    var epoch by remember { mutableStateOf(20) }
    val epochHistory = mutableStateListOf<Epoch>()

    val epochState = rememberLazyListState(0)

    var modelExpanded by remember { mutableStateOf(false) }
    var modelItems by remember { mutableStateOf(modelsDir.listFiles()!!.toMutableList()) }

    var selectedModel by remember { mutableStateOf(modelItems.first()) }

    var imageLabel by remember { mutableStateOf("") }
    var toImage by remember { mutableStateOf(ImageBitmap(28, 28, ImageBitmapConfig.F16)) }

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
                        text = summary,
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
                    }
                }

                Spacer(Modifier.height(10.dp))

                Column(
                    Modifier
                        .fillMaxSize(),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Top
                ) {
                    Row(
                        Modifier
                            .fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceEvenly
                    ) {
                        Button(
                            onClick = {
                                if (!isTraining) {
                                    isTraining = true
                                    trainButtonEnabled = false
                                    if (model.stopTraining) {
                                        model.stopTraining = false
                                        trainButtonText = TrainStatus.configuring
                                        model = defaultModel()
                                    }
                                    trainButtonText = TrainStatus.stop
                                    scope.launch(Dispatchers.Default) {
                                        model.use {
                                            trainButtonEnabled = true
                                            epochHistory.clear()

                                            it.fit(
                                                dataset = train,
                                                epochs = epoch,
                                                batchSize = 100
                                            )

                                            trainButtonText = TrainStatus.saving

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

                                                modelItems = modelsDir.listFiles()!!.toMutableList()
                                            }
                                        }

                                        trainButtonText = TrainStatus.train
                                        isTraining = false
                                        model.stopTraining = true
                                    }
                                } else {
                                    model.stopTraining = true
                                    trainButtonText = TrainStatus.train
                                    isTraining = false
                                }
                            },
                            enabled = trainButtonEnabled
                        ) {
                            Text(trainButtonText)
                        }

                        Button(
                            onClick = {
                                predictButtonClicked = true
                            }
                        ) {
                            Text(predictButtonText)
                        }
                    }

                    Spacer(Modifier.height(10.dp))

                    Row(
                        Modifier
                            .fillMaxWidth()
                    ) {
                        TextField(
                            value = "$epoch",
                            onValueChange = {
                                it.trim().toIntOrNull()?.let { value ->
                                    epoch = value
                                }
                            },
                            Modifier
                                .fillMaxWidth(1f),
                            label = {
                                Text("Epoch: ")
                            },
                            singleLine = true
                        )
                    }
                    Spacer(Modifier.height(10.dp))

                    Column(
                        Modifier
                            .fillMaxSize(),
                        verticalArrangement = Arrangement.SpaceEvenly,
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {

                        Column(
                            Modifier
                                .fillMaxWidth(),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            var dropdownSize by remember { mutableStateOf(0f) }

                            Button(
                                onClick = {
                                    modelExpanded = true
                                },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .onSizeChanged { dropdownSize = it.width.toFloat() }
                            ) {
                                Text(selectedModel.name, fontSize = TextUnit(10f, TextUnitType.Sp))
                            }

                            DropdownMenu(
                                expanded = modelExpanded,
                                onDismissRequest = { modelExpanded = false },
                                modifier = Modifier.width(dropdownSize.dp)
                            ) {
                                modelItems.forEach {
                                    DropdownMenuItem(
                                        onClick = {
                                            selectedModel = it
                                            modelExpanded = false
                                        },
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .wrapContentHeight()
                                    ) {
                                        Text(it.name, fontSize = TextUnit(12f, TextUnitType.Sp))
                                        Spacer(Modifier.width(20.dp))
                                        IconButton(onClick = {
                                            modelExpanded = false
                                            it.deleteRecursively()
                                            modelItems.remove(it)
                                            selectedModel = modelItems.last()
                                        }) {
                                            Icon(Icons.Default.Delete, "delete", tint = Color.Gray)
                                        }
                                    }
                                }
                            }
                        }

                        Spacer(Modifier.height(10.dp))

                        Column(
                            Modifier
                                .fillMaxSize(),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Text(imageLabel)
                            Spacer(Modifier.height(10.dp))
                            Image(toImage, "to", Modifier.size(64.dp))
                        }
                    }
                }
            }
        }

        if (predictButtonClicked) {
            FileDialog {
                it?.let { fileName ->

                    val predictFile = File(fileName)
                    val fromImageBuffer = ImageIO.read(predictFile)
                    val toImageBuffer = BufferedImage(
                        fromImageBuffer.width, fromImageBuffer.height,
                        BufferedImage.TYPE_BYTE_INDEXED
                    )

                    ColorConvertOp(
                        fromImageBuffer.colorModel.colorSpace,
                        toImageBuffer.colorModel.colorSpace,
                        null
                    ).filter(fromImageBuffer, toImageBuffer)

                    toImage = toImageBuffer.toComposeBitmap()

//                    println("from: ${fromImageBuffer.raster.dataBuffer.size}, to: ${toImageBuffer.raster.dataBuffer.size}")
//                    println("to type: ${toImageBuffer.raster.dataBuffer.dataType}")

                    val predictFloatArray = FloatArray(toImageBuffer.width * toImageBuffer.height)
                    for (i in predictFloatArray.indices) {
                        predictFloatArray[i] = fromImageBuffer.raster.dataBuffer.getElemFloat(i) / 255
                    }

//                    println("predict: ${selectedModel.name}: \n")
                    TensorFlowInferenceModel.load(selectedModel).use { inference ->
                        inference.reshape(28, 28, 1)
                        val prediction = inference.predict(predictFloatArray)

//                        println("Predicted label is: $prediction. This corresponds to class ${stringLabels[prediction]}.")
//                        println("Actual label is: 8.")

                        imageLabel = "It's a ${stringLabels[prediction]} (͠≖ ͜ʖ͠≖)\uD83D\uDC4C"
                    }
                }
                predictButtonClicked = false
            }
        }
    }
}

@OptIn(ExperimentalComposeUiApi::class)
@Composable
private fun FileDialog(
    parent: Frame? = null,
    onCloseRequest: (result: String?) -> Unit
) = AwtWindow(
    create = {
        object : FileDialog(parent, "Choose a file", LOAD) {
            override fun setVisible(value: Boolean) {
                super.setVisible(value)
                if (value) {
                    apply {
                        onCloseRequest(directory + file)
                    }
                }
            }
        }
    },
    dispose = FileDialog::dispose
)
