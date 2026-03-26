import SwiftUI

@main
struct KevlarMenuBarApp: App {
    @StateObject private var manager = ServerManager()

    var body: some Scene {
        MenuBarExtra {
            StatusView(manager: manager)
        } label: {
            Text("K")
                .fontWeight(.bold)
        }
        .menuBarExtraStyle(.window)
    }
}
