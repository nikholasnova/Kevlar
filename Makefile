APP_NAME = Kevlar
BUNDLE_ID = com.kevlar.menubar
BUILD_DIR = KevlarApp/.build/release
APP_DIR = $(APP_NAME).app
INSTALL_DIR = /Applications

.PHONY: app install clean

app:
	@echo "Building $(APP_NAME)..."
	cd KevlarApp && swift build -c release
	@echo "Creating app bundle..."
	rm -rf $(APP_DIR)
	mkdir -p $(APP_DIR)/Contents/MacOS
	mkdir -p $(APP_DIR)/Contents/Resources
	cp $(BUILD_DIR)/KevlarApp $(APP_DIR)/Contents/MacOS/$(APP_NAME)
	cp KevlarApp/Sources/Resources/AppIcon.icns $(APP_DIR)/Contents/Resources/AppIcon.icns
	@echo '<?xml version="1.0" encoding="UTF-8"?>' > $(APP_DIR)/Contents/Info.plist
	@echo '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">' >> $(APP_DIR)/Contents/Info.plist
	@echo '<plist version="1.0">' >> $(APP_DIR)/Contents/Info.plist
	@echo '<dict>' >> $(APP_DIR)/Contents/Info.plist
	@echo '  <key>CFBundleExecutable</key><string>$(APP_NAME)</string>' >> $(APP_DIR)/Contents/Info.plist
	@echo '  <key>CFBundleIdentifier</key><string>$(BUNDLE_ID)</string>' >> $(APP_DIR)/Contents/Info.plist
	@echo '  <key>CFBundleName</key><string>$(APP_NAME)</string>' >> $(APP_DIR)/Contents/Info.plist
	@echo '  <key>CFBundlePackageType</key><string>APPL</string>' >> $(APP_DIR)/Contents/Info.plist
	@echo '  <key>CFBundleIconFile</key><string>AppIcon</string>' >> $(APP_DIR)/Contents/Info.plist
	@echo '  <key>CFBundleVersion</key><string>1.0</string>' >> $(APP_DIR)/Contents/Info.plist
	@echo '  <key>CFBundleShortVersionString</key><string>1.0</string>' >> $(APP_DIR)/Contents/Info.plist
	@echo '  <key>LSUIElement</key><true/>' >> $(APP_DIR)/Contents/Info.plist
	@echo '  <key>LSMinimumSystemVersion</key><string>14.0</string>' >> $(APP_DIR)/Contents/Info.plist
	@echo '</dict>' >> $(APP_DIR)/Contents/Info.plist
	@echo '</plist>' >> $(APP_DIR)/Contents/Info.plist
	@echo "Built $(APP_DIR)"

install: app
	@echo "Installing to $(INSTALL_DIR)/$(APP_DIR)..."
	cp -R $(APP_DIR) $(INSTALL_DIR)/
	@echo "Done. Open '$(APP_NAME)' from Spotlight or Launchpad."

clean:
	rm -rf $(APP_DIR)
	cd KevlarApp && swift package clean
