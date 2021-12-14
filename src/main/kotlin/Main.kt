// Copyright 2000-2021 JetBrains s.r.o. and contributors. Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE file.
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import compose.App
import io.ktor.routing.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch
import server.NeuralNetworkServer

val coroutineScope = CoroutineScope(Dispatchers.IO)

fun main() {
    Thread.setDefaultUncaughtExceptionHandler { _, exception ->
        println(exception.message)
        coroutineScope.cancel()
    }

    coroutineScope.launch {
        NeuralNetworkServer(this)
    }

    application {
        Window(onCloseRequest = {
            coroutineScope.cancel()
            exitApplication()
        }) {
            App(window.size)
        }
    }
}