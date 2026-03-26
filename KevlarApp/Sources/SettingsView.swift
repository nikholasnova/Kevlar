import SwiftUI

struct SettingsView: View {
    @ObservedObject var manager: ServerManager
    @State private var portText: String = ""
    @State private var haikuModelText: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Settings")
                .font(.headline)

            LabeledContent("Port") {
                TextField("8080", text: $portText)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 80)
                    .onAppear { portText = String(manager.port) }
                    .onSubmit {
                        if let p = Int(portText), (1024...65535).contains(p) {
                            manager.port = p
                        }
                    }
            }

            LabeledContent("Haiku Model") {
                TextField("model ID", text: $haikuModelText)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 200)
                    .onAppear { haikuModelText = manager.haikuModel }
                    .onSubmit { manager.haikuModel = haikuModelText }
            }

            LabeledContent("Cache Dir") {
                Text(manager.cacheDir)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }

            LabeledContent("SSD Cache") {
                Text(manager.cacheSizeString)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Divider()

            Text("Changes take effect on next server start.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }
}
