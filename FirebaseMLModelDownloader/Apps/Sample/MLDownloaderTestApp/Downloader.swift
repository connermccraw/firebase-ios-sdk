// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation
import TensorFlowLite
import FirebaseMLModelDownloader

class Downloader: ObservableObject {
  @Published var downloadProgress: Float = 0.0
  @Published var modelName = ""
  @Published var filePath = ""
  @Published var error = ""
  @Published var isDownloaded = false
  @Published var isDeleted = false
  @Published var isError = false
  @Published var modelNames = [String]()

  private func resetState() {
    isDownloaded = false
    isDeleted = false
    downloadProgress = 0.0
    filePath = ""
    error = ""
    isError = false
    modelNames = []
  }

  func downloadModelHelper(downloadType: ModelDownloadType) -> () -> Void {
    return {
      self.resetState()
      self.downloadModel(downloadType: downloadType)
    }
  }

  func downloadModel(downloadType: ModelDownloadType) {
    let modelDownloader = ModelDownloader.modelDownloader()
    let conditions = ModelDownloadConditions()
    modelDownloader.getModel(
      name: modelName,
      downloadType: downloadType,
      conditions: conditions,
      progressHandler: { progress in
        self.downloadProgress = progress
      }
    ) { result in
      switch result {
      case let .success(model):
        self.isDownloaded = true
        self.filePath = model.path
        let fileURL = URL(fileURLWithPath: self.filePath)
        do {
          _ = try fileURL.checkResourceIsReachable()
          let attr = try FileManager.default.attributesOfItem(atPath: self.filePath)
          if let size = attr[FileAttributeKey.size] {
            print("File size: \(size)")
          } else {
            print("Error - could not get file size.")
          }
        } catch {
          print("File access error - \(error)")
        }
        do {
          let interpreter = try Interpreter(modelPath: self.filePath)
            guard let sourceURL =  Bundle.main.url(forResource: "Voltages", withExtension: "json") else {
                fatalError("Could not find Voltages.json")
            }
            guard let data = try? Data(contentsOf: sourceURL) else {
                fatalError("Could not convert data")
            }
            try interpreter.allocateTensors()
            try interpreter.copy(data, toInputAt: 0)
            try interpreter.invoke()
            
//            let prob = UnsafeMutableBufferPointer<Float64>.allocate(capacity: 1000)
//            let output = try interpreter.output(at: 0).data.copyBytes(to: prob)
//            print(prob)
            let output = try interpreter.output(at: 0)
            print(output)
            let probabilities = self.dataToFloatArray(output.data)
            print(probabilities)
            
        } catch {
          print("Tensorflow error - \(error)")
        }
      case let .failure(error):
        self.isDownloaded = false
        self.isError = true
        self.error = "Model download failed with error: \(error)"
      }
    }
  }

  func deleteModelHelper() -> () -> Void {
    return {
      self.resetState()
      self.deleteModel()
    }
  }

  func deleteModel() {
    let modelDownloader = ModelDownloader.modelDownloader()
    modelDownloader.deleteDownloadedModel(name: modelName) { result in
      switch result {
      case .success:
        self.isDeleted = true
        self.isDownloaded = false
        self.filePath = ""
      case let .failure(error):
        self.isDeleted = false
        self.isError = true
        self.error = "Model deletion failed with error: \(error)"
      }
    }
  }

  func listModelHelper() -> () -> Void {
    return {
      self.resetState()
      self.listModel()
    }
  }

  func listModel() {
    let modelDownloader = ModelDownloader.modelDownloader()
    modelDownloader.listDownloadedModels { result in
      switch result {
      case let .success(models):
        if models.count == 0 {
          self.isError = true
          self.error = "No models found on device."
        } else {
          for model in models {
            self.modelNames.append(model.name)
          }
        }
      case let .failure(error):
        self.isError = true
        self.error = "Listing models failed with error: \(error)"
      }
    }
  }
    private func dataToFloatArray(_ data: Data) -> [Float]? {
        guard data.count % MemoryLayout<Float>.stride == 0 else { return nil }

        #if swift(>=5.0)
        return data.withUnsafeBytes { .init($0.bindMemory(to: Float.self)) }
        #else
        return data.withUnsafeBytes {
          .init(UnsafeBufferPointer<Float>(
            start: $0,
            count: unsafeData.count / MemoryLayout<Element>.stride
          ))
        }
        #endif // swift(>=5.0)
      }
}
