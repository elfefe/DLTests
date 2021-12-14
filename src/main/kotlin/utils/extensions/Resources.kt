package utils.extensions

import java.io.File
import java.util.*

val timestamp: Long
    get() = Date().time

val modelsDir: File = File("models").apply { createNewFile() }