import Foundation
import SwiftUI

@MainActor
final class ServerManager: ObservableObject {
    @Published var mainStatus: ServerStatus?
    @Published var haikuStatus: ServerStatus?
    @Published var isStarting = false
    @Published var isLoadingModel = false
    @Published var currentModel: String
    @Published var haikuModel: String
    @Published var models: [String]
    @Published var port: Int = 8080
    @Published var cacheDir: String

    private var mainProcess: Process?
    private var haikuProcess: Process?
    private var pollTimer: Timer?

    private let modelsFile: URL
    private let defaultModels = [
        "mlx-community/Qwen3.5-122B-A10B-4bit",
        "mlx-community/Qwen3-Coder-Next-8bit",
    ]

    var isRunning: Bool { mainStatus != nil }
    var haikuRunning: Bool { haikuStatus != nil }
    var haikuPort: Int { port + 1 }

    var mainURL: String { "http://127.0.0.1:\(port)" }
    var haikuURL: String { "http://127.0.0.1:\(haikuPort)" }

    var modelLoaded: Bool { mainStatus?.model != nil }

    init() {
        let kevlarDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".kevlar")
        modelsFile = kevlarDir.appendingPathComponent("models.json")
        cacheDir = kevlarDir.appendingPathComponent("cache").path

        var loaded: [String] = []
        if let data = try? Data(contentsOf: modelsFile),
           let list = try? JSONDecoder().decode([String].self, from: data),
           !list.isEmpty {
            loaded = list
        } else {
            loaded = defaultModels
        }
        models = loaded
        currentModel = loaded.first ?? defaultModels[0]
        haikuModel = "mlx-community/Qwen3-8B-4bit"

        startPolling()
    }

    // MARK: - Server lifecycle

    func startServers() {
        guard mainProcess == nil else { return }
        isStarting = true

        let python = findPython()

        mainProcess = launchProcess(
            python: python,
            args: ["serve", "--model", currentModel, "--host", "127.0.0.1",
                   "--port", String(port), "--haiku-port", String(haikuPort),
                   "--cache-dir", cacheDir]
        )

        haikuProcess = launchProcess(
            python: python,
            args: ["serve", "--model", haikuModel, "--host", "127.0.0.1",
                   "--port", String(haikuPort), "--cache-dir", cacheDir,
                   "--max-cache-gb", "2", "--max-tokens", "8192"]
        )
    }

    func stopServers() {
        let mainPort = port
        let haiku = haikuPort
        mainProcess = nil
        haikuProcess = nil
        mainStatus = nil
        haikuStatus = nil
        isStarting = false
        isLoadingModel = false
        DispatchQueue.global().async {
            self.killServerOnPort(mainPort)
            self.killServerOnPort(haiku)
        }
    }

    // MARK: - Hot-swap model via API

    func switchModel(_ model: String) {
        if let loaded = mainStatus?.model, loaded == model { return }
        currentModel = model

        if isRunning {
            Task {
                await loadAllModels(mainModel: model)
            }
        }
    }

    func unloadModel() {
        guard isRunning else { return }
        Task {
            await postJSON(port: port, path: "/v1/model/unload", body: [:])
            await postJSON(port: haikuPort, path: "/v1/model/unload", body: [:])
        }
    }

    private func loadAllModels(mainModel: String) async {
        isLoadingModel = true
        async let mainLoad = postJSON(port: port, path: "/v1/model/load", body: ["model": mainModel])
        async let haikuLoad = postJSON(port: haikuPort, path: "/v1/model/load", body: ["model": haikuModel])
        let _ = await (mainLoad, haikuLoad)
        isLoadingModel = false
    }

    private func postJSON(port: Int, path: String, body: [String: String]) async -> Bool {
        guard let url = URL(string: "http://127.0.0.1:\(port)\(path)") else { return false }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 300
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            return (response as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }

    // MARK: - Model management

    func addModel(_ id: String) {
        guard !id.isEmpty, !models.contains(id) else { return }
        models.append(id)
        saveModels()
    }

    func removeModel(_ id: String) {
        guard models.count > 1, id != currentModel else { return }
        models.removeAll { $0 == id }
        saveModels()
    }

    func shortName(_ id: String) -> String {
        id.split(separator: "/").last.map(String.init) ?? id
    }

    // MARK: - Cache

    func clearCache() {
        let python = findPython()
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: python)
        proc.arguments = ["-m", "kevlar.main", "cache", "clear", "-f"]
        proc.environment = processEnv()
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        try? proc.run()
        proc.waitUntilExit()
    }

    var cacheSizeString: String {
        let url = URL(fileURLWithPath: cacheDir)
        guard let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else {
            return "0 MB"
        }
        var total: Int64 = 0
        for case let file as URL in enumerator {
            if let size = try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                total += Int64(size)
            }
        }
        let mb = Double(total) / 1_000_000
        if mb > 1000 { return String(format: "%.1f GB", mb / 1000) }
        return String(format: "%.0f MB", mb)
    }

    // MARK: - Launch Claude Code

    func launchClaudeCode() {
        let script = """
        tell application "Terminal"
            activate
            do script "env -u ANTHROPIC_AUTH_TOKEN -u ANTHROPIC_API_KEY ANTHROPIC_BASE_URL=\(mainURL) claude --model \(currentModel)"
        end tell
        """
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        proc.arguments = ["-e", script]
        try? proc.run()
    }

    // MARK: - Polling

    private func startPolling() {
        pollTimer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                await self?.poll()
            }
        }
    }

    private func poll() async {
        mainStatus = await fetchStatus(port: port)
        haikuStatus = await fetchStatus(port: haikuPort)

        if isStarting && mainStatus != nil {
            isStarting = false
        }

        // Sync currentModel from server if it changed via API
        if let serverModel = mainStatus?.model, serverModel != currentModel {
            currentModel = serverModel
        }

        // Detect crash
        if mainProcess != nil && !mainProcess!.isRunning && mainStatus == nil {
            mainProcess = nil
            isStarting = false
        }
        if haikuProcess != nil && !haikuProcess!.isRunning && haikuStatus == nil {
            haikuProcess = nil
        }
    }

    private func fetchStatus(port: Int) async -> ServerStatus? {
        guard let url = URL(string: "http://127.0.0.1:\(port)/v1/status") else { return nil }
        var request = URLRequest(url: url)
        request.timeoutInterval = 3
        do {
            let (data, _) = try await URLSession.shared.data(for: request)
            return try JSONDecoder().decode(ServerStatus.self, from: data)
        } catch {
            return nil
        }
    }

    // MARK: - Process helpers

    private func findPython() -> String {
        let venvPython = findKevlarRoot()
            .appendingPathComponent(".venv/bin/python").path
        if FileManager.default.fileExists(atPath: venvPython) {
            return venvPython
        }
        return "/usr/bin/python3"
    }

    private func findKevlarRoot() -> URL {
        let candidates = [
            Bundle.main.bundleURL.deletingLastPathComponent()
                .deletingLastPathComponent(),
            FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Desktop/repos/Kevlar"),
        ]
        for c in candidates {
            if FileManager.default.fileExists(
                atPath: c.appendingPathComponent("kevlar/main.py").path) {
                return c
            }
        }
        return candidates.last!
    }

    private func processEnv() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        let root = findKevlarRoot()
        let venvBin = root.appendingPathComponent(".venv/bin").path
        if let path = env["PATH"] {
            env["PATH"] = "\(venvBin):\(path)"
        }
        env["PYTHONPATH"] = root.path
        return env
    }

    private func launchProcess(python: String, args: [String]) -> Process {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: python)
        proc.arguments = ["-m", "kevlar.main"] + args
        proc.environment = processEnv()
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        proc.qualityOfService = .userInitiated
        try? proc.run()
        return proc
    }

    nonisolated private func killServerOnPort(_ port: Int) {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/bin/bash")
        proc.arguments = ["-c", "lsof -ti :\(port) -sTCP:LISTEN | xargs kill -9 2>/dev/null"]
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        try? proc.run()
        proc.waitUntilExit()
    }

    private func saveModels() {
        try? FileManager.default.createDirectory(
            at: modelsFile.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        if let data = try? JSONEncoder().encode(models) {
            try? data.write(to: modelsFile)
        }
    }
}
