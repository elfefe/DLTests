package utils.extensions

import androidx.compose.runtime.Composable
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.window.AwtWindow
import java.awt.FileDialog
import java.awt.Frame


@OptIn(ExperimentalComposeUiApi::class)
@Composable
fun FileDialog(
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