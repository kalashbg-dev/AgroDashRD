# AgroDashRD Mobile App 📱

This is the mobile application for AgroDashRD, built with **Flutter**.

## Structure

*   `lib/`: Main source code.
    *   `main.dart`: Entry point.
    *   `services/`: API communication (`api_service.dart`).
    *   `screens/`: UI Screens (Login, Home, etc).
    *   `models/`: Data models (User, Price, etc).
*   `pubspec.yaml`: Flutter dependencies.

## Setup

1.  **Install Flutter SDK**: [https://flutter.dev/docs/get-started/install](https://flutter.dev/docs/get-started/install)
2.  **Install Dependencies**:
    ```bash
    flutter pub get
    ```
3.  **Run the App**:
    ```bash
    flutter run
    ```
    *   For Android Emulator: Ensure backend is running. `api_service.dart` defaults to `http://10.0.2.2:8000`.
    *   For iOS Simulator / Web / Linux: Update `api_service.dart` to use `http://localhost:8000`.

## Features (Skeleton)

*   **Authentication**: Login and Register screens connected to `ApiService`.
*   **Dashboard**: Home screen listing latest prices.
*   **Reporting**: Screen to report new prices.

*Note: This is an MVP skeleton. Business logic integration is pending detailed implementation.*
