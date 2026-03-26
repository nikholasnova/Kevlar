import Foundation

struct CacheStatus: Codable, Sendable {
    let memoryEntries: Int
    let memoryBytes: Int
    let ssdDir: String
    let ssdEntries: Int

    enum CodingKeys: String, CodingKey {
        case memoryEntries = "memory_entries"
        case memoryBytes = "memory_bytes"
        case ssdDir = "ssd_dir"
        case ssdEntries = "ssd_entries"
    }
}

struct ServerStatus: Codable, Sendable {
    let status: String
    let model: String?
    let modelLoaded: Bool?
    let uptimeS: Double
    let cache: CacheStatus

    enum CodingKeys: String, CodingKey {
        case status
        case model
        case modelLoaded = "model_loaded"
        case uptimeS = "uptime_s"
        case cache
    }

    var uptimeFormatted: String {
        let h = Int(uptimeS) / 3600
        let m = (Int(uptimeS) % 3600) / 60
        if h > 0 { return "\(h)h \(m)m" }
        if m > 0 { return "\(m)m" }
        return "\(Int(uptimeS))s"
    }

    var cacheMB: Double {
        Double(cache.memoryBytes) / 1_000_000
    }
}
