import SwiftUI

struct StatusView: View {
    @ObservedObject var manager: ServerManager

    @State private var showModelPicker = false
    @State private var showSettings = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                Text("Kevlar")
                    .font(.headline)
                Spacer()
                Button(action: { showSettings.toggle() }) {
                    Image(systemName: "gearshape")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .popover(isPresented: $showSettings) {
                    SettingsView(manager: manager)
                        .padding()
                        .frame(width: 300)
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 12)
            .padding(.bottom, 8)

            Divider()

            // Status section
            VStack(alignment: .leading, spacing: 10) {
                // Main server
                HStack(spacing: 8) {
                    Circle()
                        .fill(statusColor)
                        .frame(width: 8, height: 8)
                    if manager.isLoadingModel {
                        Text("Loading model...")
                            .foregroundStyle(.secondary)
                    } else if manager.isStarting {
                        Text("Starting...")
                            .foregroundStyle(.secondary)
                    } else if let status = manager.mainStatus {
                        if let model = status.model {
                            Text(manager.shortName(model))
                                .fontWeight(.medium)
                        } else {
                            Text("No model loaded")
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Text(status.uptimeFormatted)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else {
                        Text("Stopped")
                            .foregroundStyle(.secondary)
                    }
                }

                // Haiku server
                HStack(spacing: 8) {
                    let haikuModelLoaded = manager.haikuStatus?.model != nil
                    Circle()
                        .fill(haikuModelLoaded ? .green : manager.haikuRunning ? .orange : .secondary.opacity(0.3))
                        .frame(width: 6, height: 6)
                    Text("Haiku")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    if let hs = manager.haikuStatus, let hm = hs.model {
                        Text(manager.shortName(hm))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else if manager.haikuRunning {
                        Text("No model")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else if manager.isStarting {
                        Text("Starting...")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                // Cache stats
                if let status = manager.mainStatus {
                    HStack {
                        Label {
                            Text("\(status.cache.memoryEntries) entries")
                        } icon: {
                            Image(systemName: "memorychip")
                        }
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        Spacer()
                        Text(String(format: "%.0f MB", status.cacheMB))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            Divider()

            // Controls
            VStack(spacing: 6) {
                // Server start / stop
                if manager.isRunning || manager.isStarting {
                    HStack(spacing: 6) {
                        Button(action: { manager.stopServers() }) {
                            Label("Stop Server", systemImage: "stop.fill")
                                .frame(maxWidth: .infinity)
                        }
                        .controlSize(.regular)

                        // Unload model (free memory, keep server up)
                        Button(action: { manager.unloadModel() }) {
                            Label("Unload", systemImage: "arrow.down.to.line")
                                .frame(maxWidth: .infinity)
                        }
                        .controlSize(.regular)
                        .disabled(!manager.modelLoaded || manager.isLoadingModel)
                    }
                } else {
                    Button(action: { manager.startServers() }) {
                        Label("Start Server", systemImage: "play.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .controlSize(.regular)
                    .tint(.blue)
                }

                // Launch Claude Code
                Button(action: { manager.launchClaudeCode() }) {
                    Label("Launch Claude Code", systemImage: "terminal")
                        .frame(maxWidth: .infinity)
                }
                .controlSize(.regular)
                .disabled(!manager.modelLoaded)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            Divider()

            // Model picker
            VStack(alignment: .leading, spacing: 6) {
                Text("Models")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                let modelList = manager.models
                ForEach(modelList, id: \.self) { (model: String) in
                    Button {
                        manager.switchModel(model)
                    } label: {
                        HStack {
                            if model == manager.currentModel && manager.modelLoaded {
                                Image(systemName: "checkmark")
                                    .font(.caption2)
                                    .foregroundStyle(.blue)
                                    .frame(width: 14)
                            } else {
                                Spacer().frame(width: 14)
                            }
                            Text(manager.shortName(model))
                                .font(.callout)
                                .lineLimit(1)
                                .opacity(model == manager.currentModel && manager.modelLoaded ? 1 : 0.7)
                            Spacer()
                            if manager.isRunning && !(model == manager.currentModel && manager.modelLoaded) {
                                Text("load")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .padding(.vertical, 2)
                    .disabled(manager.isLoadingModel)
                }

                Button(action: { showModelPicker.toggle() }) {
                    Label("Manage Models...", systemImage: "plus.circle")
                        .font(.caption)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .popover(isPresented: $showModelPicker) {
                    ModelPickerView(manager: manager)
                        .padding()
                        .frame(width: 320)
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            Divider()

            // Footer actions
            HStack {
                Button("Copy URL") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(manager.mainURL, forType: .string)
                }
                .font(.caption)
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)

                Spacer()

                Button("Clear Cache") {
                    manager.clearCache()
                }
                .font(.caption)
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)

                Spacer()

                Button("Quit") {
                    manager.stopServers()
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        NSApplication.shared.terminate(nil)
                    }
                }
                .font(.caption)
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
        }
        .frame(width: 300)
    }

    private var statusColor: Color {
        if manager.isLoadingModel { return .orange }
        if manager.isStarting { return .orange }
        if manager.isRunning { return .green }
        return .secondary.opacity(0.3)
    }
}
