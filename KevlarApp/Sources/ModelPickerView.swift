import SwiftUI

struct ModelPickerView: View {
    @ObservedObject var manager: ServerManager
    @State private var newModelID = "mlx-community/"

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Manage Models")
                .font(.headline)

            // Model list
            List {
                ForEach(manager.models, id: \.self) { model in
                    HStack {
                        VStack(alignment: .leading) {
                            Text(manager.shortName(model))
                                .fontWeight(model == manager.currentModel ? .medium : .regular)
                            Text(model)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                        Spacer()
                        if model == manager.currentModel {
                            Text("active")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(.secondary.opacity(0.1))
                                .clipShape(Capsule())
                        } else {
                            Button(role: .destructive) {
                                manager.removeModel(model)
                            } label: {
                                Image(systemName: "xmark")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }
            .frame(height: 160)

            // Add model
            HStack {
                TextField("HuggingFace model ID", text: $newModelID)
                    .textFieldStyle(.roundedBorder)
                    .font(.callout)
                Button("Add") {
                    let trimmed = newModelID.trimmingCharacters(in: .whitespaces)
                    if !trimmed.isEmpty && trimmed != "mlx-community/" {
                        manager.addModel(trimmed)
                        newModelID = "mlx-community/"
                    }
                }
                .disabled(newModelID.trimmingCharacters(in: .whitespaces).isEmpty
                          || newModelID == "mlx-community/")
            }

            Text("Models from [mlx-community](https://huggingface.co/mlx-community) are recommended. They download automatically on first use.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }
}
