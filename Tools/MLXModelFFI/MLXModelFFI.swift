//
//  MLXModelFFI.swift
//  MLXModelFFI
//
//  Created by molly on 2025/11/4.
//

import Foundation
import MLXLMCommon
import MLXLLM
import MLX
import Metal

private let rwkvMeasuredDesktopFFI: Bool = {
    #if os(macOS)
    return MTLCopyAllDevices().first?.name == "Apple M4 Pro"
    #else
    return false
    #endif
}()
private func rwkvDesktopOption(_ name: String) -> Bool {
    #if os(macOS)
    return ProcessInfo.processInfo.environment[name].map { $0 == "1" } ?? rwkvMeasuredDesktopFFI
    #else
    return false
    #endif
}
private let rwkvLazyClearEnabled = rwkvDesktopOption("RWKV_MLX_LAZY_CLEAR")
private let rwkvLazyClearTrace = ProcessInfo.processInfo.environment["RWKV_MLX_TRACE_LAZY_CLEAR"] == "1"

private final class ModelContext {
    let modelDirectory: URL
    let model: any LanguageModel
    let config: FFIModelConfig
    var cache: [KVCache]
    var greedyLogits: MLXArray?
    var greedyToken: MLXArray?
    var greedyRollbackState: [[MLXArray]]?
    var greedyRollbackLogits: MLXArray?
    var deferredEvalPending = false

    func invalidateGreedy() {
        greedyLogits = nil
        greedyToken = nil
        greedyRollbackState = nil
        greedyRollbackLogits = nil
        deferredEvalPending = false
    }

    init(modelDirectory: URL, model: any LanguageModel, config: FFIModelConfig) {
        self.modelDirectory = modelDirectory
        self.model = model
        self.config = config
        self.cache = model.newCache(parameters: nil)
    }
}

private struct FFIModelConfig {
    let vocabSize: Int
    let hiddenSize: Int
    let headDim: Int
    let numLayers: Int

    var numHeads: Int {
        hiddenSize / headDim
    }
}

private func intValue(_ value: Any?) -> Int? {
    guard let value else { return nil }
    if let intValue = value as? Int {
        return intValue
    }
    if let int64Value = value as? Int64 {
        return Int(int64Value)
    }
    if let doubleValue = value as? Double {
        return Int(doubleValue)
    }
    return nil
}

private func readFFIModelConfig(from configurationURL: URL) throws -> FFIModelConfig {
    let configData = try Data(contentsOf: configurationURL)
    guard let configDict = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
        throw NSError(domain: "MLXModelFFI", code: 1, userInfo: [NSLocalizedDescriptionKey: "config.json is not a valid JSON object"])
    }

    guard let vocabSize = intValue(configDict["vocab_size"]), vocabSize > 0 else {
        throw NSError(domain: "MLXModelFFI", code: 2, userInfo: [NSLocalizedDescriptionKey: "could not determine vocab_size from config"])
    }
    guard let hiddenSize = intValue(configDict["hidden_size"]), hiddenSize > 0 else {
        throw NSError(domain: "MLXModelFFI", code: 3, userInfo: [NSLocalizedDescriptionKey: "could not determine hidden_size from config"])
    }
    let headDim = intValue(configDict["head_dim"]) ??
        intValue(configDict["num_attention_heads"]).map { $0 > 0 ? hiddenSize / $0 : 0 } ?? 0
    guard headDim > 0 else {
        throw NSError(domain: "MLXModelFFI", code: 4, userInfo: [NSLocalizedDescriptionKey: "could not determine head_dim from config"])
    }
    guard let numLayers = intValue(configDict["num_hidden_layers"]), numLayers > 0 else {
        throw NSError(domain: "MLXModelFFI", code: 5, userInfo: [NSLocalizedDescriptionKey: "could not determine num_hidden_layers from config"])
    }

    return FFIModelConfig(vocabSize: vocabSize, hiddenSize: hiddenSize, headDim: headDim, numLayers: numLayers)
}

// MARK: - Error handling

nonisolated(unsafe) private var errorMessageBuffer: UnsafeMutablePointer<CChar>? = nil
private let errorQueue = DispatchQueue(label: "mlx_model_ffi.error.queue")

@inline(__always)
private func setFFIError(_ message: String) {
    errorQueue.sync {
        if let old = errorMessageBuffer {
            free(old)
            errorMessageBuffer = nil
        }
        errorMessageBuffer = strdup(message)
    }
}

@_cdecl("mlx_last_error_message")
public func mlx_last_error_message() -> UnsafePointer<CChar>? {
    return UnsafePointer(errorMessageBuffer)
}

// MARK: - FFI API

@_cdecl("mlx_initialize")
public func mlx_initialize() -> Int32 {
    // Force initialization by creating a dummy array and evaluating it
    // This triggers MLX's internal initialization and loads the metallib
    let _ = MLXArray(0.0).eval()
    return 0 // Success
}

@_cdecl("mlx_model_load")
public func mlx_model_load(_ cPath: UnsafePointer<CChar>?) -> UnsafeMutableRawPointer? {
    guard let cPath else {
        setFFIError("path is null")
        return nil
    }

    let path = String(cString: cPath)

    // Resolve model directory
    let modelDirectory = URL(filePath: path)
    guard FileManager.default.fileExists(atPath: modelDirectory.path) else {
        setFFIError("file or directory not found: \(path)")
        return nil
    }

    do {
        // 1) Read base config to get model type and quantization
        let configurationURL = modelDirectory.appending(component: "config.json")
        let baseConfig = try JSONDecoder().decode(
            BaseConfiguration.self, from: Data(contentsOf: configurationURL))
        let ffiConfig = try readFFIModelConfig(from: configurationURL)

        // 2) Instantiate model from registry by model type
        let model = try LLMTypeRegistry.shared.createModel(
            configuration: configurationURL, modelType: baseConfig.modelType)

        // 3) Load and apply weights (handles optional per-layer quantization)
        try loadWeights(
            modelDirectory: modelDirectory,
            model: model,
            perLayerQuantization: baseConfig.perLayerQuantization)

        // 4) Return opaque handle retaining the context
        let context = ModelContext(modelDirectory: modelDirectory, model: model, config: ffiConfig)
        return Unmanaged.passRetained(context).toOpaque()

    } catch {
        setFFIError("load failed: \(error)")
        return nil
    }
}

@_cdecl("mlx_model_release")
public func mlx_model_release(_ handle: UnsafeMutableRawPointer?) {
    guard let handle else { return }
    Unmanaged<ModelContext>.fromOpaque(handle).release()
    if rwkvMeasuredDesktopFFI { eval(MLXRandom.globalState) }
    GPU.clearCache()
}

// MARK: - Configuration API

@_cdecl("mlx_model_get_config")
public func mlx_model_get_config(
    _ handle: UnsafeMutableRawPointer?,
    _ vocabSize: UnsafeMutablePointer<Int32>?,
    _ hiddenSize: UnsafeMutablePointer<Int32>?,
    _ headDim: UnsafeMutablePointer<Int32>?,
    _ numLayers: UnsafeMutablePointer<Int32>?
) -> Int32 {
    guard let handle else {
        setFFIError("handle is null")
        return -1
    }

    guard let vocabSize, let hiddenSize, let headDim, let numLayers else {
        setFFIError("output pointers are null")
        return -1
    }

    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()

    vocabSize.pointee = Int32(context.config.vocabSize)
    hiddenSize.pointee = Int32(context.config.hiddenSize)
    headDim.pointee = Int32(context.config.headDim)
    numLayers.pointee = Int32(context.config.numLayers)

    return 0 // Success
}

// MARK: - Eval API

private func resizeArrayBatch(_ array: MLXArray, batchSize: Int) -> MLXArray {
    let currentBatchSize = array.dim(0)
    if currentBatchSize == batchSize {
        return array
    }
    if currentBatchSize > batchSize {
        return array[0 ..< batchSize, 0...]
    }

    var extraShape = array.shape
    extraShape[0] = batchSize - currentBatchSize
    let zeros = MLXArray.zeros(extraShape, dtype: array.dtype)
    return concatenated([array, zeros], axis: 0)
}

private func ensureCacheBatchSize(_ context: ModelContext, batchSize: Int) -> Bool {
    guard batchSize > 0 else {
        setFFIError("batch size must be greater than 0")
        return false
    }

    for cacheIndex in 0..<context.cache.count {
        let state = context.cache[cacheIndex].state
        if state.isEmpty || state.allSatisfy({ $0.dim(0) == batchSize }) {
            continue
        }
        context.cache[cacheIndex].state = state.map { resizeArrayBatch($0, batchSize: batchSize) }
    }
    return true
}

private func evaluateModel(_ context: ModelContext, input: LMInput) throws -> MLXArray {
    let prepared = try context.model.prepare(input, cache: context.cache, windowSize: nil)
    switch prepared {
        case .tokens(let tokens):
            let modelInput: LMInput.Text
            if tokens.tokens.ndim == 1 {
                modelInput = tokens[text: .newAxis]
            } else {
                modelInput = tokens
            }
            if let model = context.model as? Rwkv7Model {
                return model.lastLogits(modelInput.tokens, cache: context.cache.isEmpty ? nil : context.cache)
            }
            let out = context.model(modelInput, cache: context.cache.isEmpty ? nil : context.cache, state: nil)
            return out.logits
        case .logits(let out):
            return out.logits
    }
}

private func copyLastLogits(_ result: MLXArray, expectedBatchSize: Int, to logits: UnsafeMutablePointer<Float>) -> Int32 {
    let lastLogits: MLXArray
    if result.ndim == 3 {
        let batchSize = result.dim(0)
        let seqLen = result.dim(1)
        guard batchSize == expectedBatchSize, seqLen > 0 else {
            setFFIError("invalid logits shape: \(result.shape), expected batch \(expectedBatchSize)")
            return -1
        }
        if expectedBatchSize == 1 {
            lastLogits = result[0, -1, 0...]
        } else {
            lastLogits = result[0..., -1, 0...]
        }
    } else if result.ndim == 2 && expectedBatchSize == 1 {
        let seqLen = result.dim(0)
        guard seqLen > 0 else {
            setFFIError("invalid logits shape: \(result.shape)")
            return -1
        }
        lastLogits = result[-1, 0...]
    } else {
        setFFIError("unsupported logits shape: \(result.shape)")
        return -1
    }

    let logitsArray = lastLogits.dtype == .float32 ? lastLogits : lastLogits.asType(.float32)
    let logitsCount = logitsArray.size
    guard logitsCount > 0 else {
        setFFIError("invalid logits size: \(logitsCount)")
        return -1
    }

    withExtendedLifetime(logitsArray) {
        let logitsData = logitsArray.asData(access: .copy)
        logitsData.data.withUnsafeBytes { bytes in
            let floatBytes = bytes.bindMemory(to: Float.self)
            logits.initialize(from: floatBytes.baseAddress!, count: logitsCount)
        }
    }

    return 0
}

@_cdecl("mlx_model_eval")
public func mlx_model_eval(
    _ handle: UnsafeMutableRawPointer?,
    _ ids: UnsafePointer<Int32>?,
    _ idsLength: Int32,
    _ logits: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let handle else {
        setFFIError("handle is null")
        return -1
    }

    guard let ids else {
        setFFIError("ids is null")
        return -1
    }

    guard let logits else {
        setFFIError("logits output pointer is null")
        return -1
    }

    guard idsLength > 0 else {
        setFFIError("ids length must be greater than 0")
        return -1
    }

    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    context.invalidateGreedy()
    do {
        guard ensureCacheBatchSize(context, batchSize: 1) else {
            return -1
        }

        let idsArray = Array(UnsafeBufferPointer(start: ids, count: Int(idsLength)))
        let inputTokens = MLXArray(idsArray)
        let input = LMInput(tokens: inputTokens)
        let result = try evaluateModel(context, input: input)
        context.greedyLogits = result
        return copyLastLogits(result, expectedBatchSize: 1, to: logits)

    } catch {
        setFFIError("eval failed: \(error)")
        return -1
    }
}

@_cdecl("mlx_model_eval_batch_tokens")
public func mlx_model_eval_batch_tokens(
    _ handle: UnsafeMutableRawPointer?,
    _ ids: UnsafePointer<Int32>?,
    _ batchSize: Int32,
    _ logits: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let handle else {
        setFFIError("handle is null")
        return -1
    }

    guard let ids else {
        setFFIError("ids is null")
        return -1
    }

    guard let logits else {
        setFFIError("logits output pointer is null")
        return -1
    }

    guard batchSize > 0 else {
        setFFIError("batch size must be greater than 0")
        return -1
    }

    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    context.invalidateGreedy()
    let batchCount = Int(batchSize)
    do {
        guard ensureCacheBatchSize(context, batchSize: batchCount) else {
            return -1
        }

        let idsArray = Array(UnsafeBufferPointer(start: ids, count: batchCount))
        let inputTokens = MLXArray(idsArray).reshaped(batchCount, 1)
        let input = LMInput(tokens: inputTokens)
        let result = try evaluateModel(context, input: input)
        if batchCount == 1 { context.greedyLogits = result }
        return copyLastLogits(result, expectedBatchSize: batchCount, to: logits)
    } catch {
        setFFIError("batch eval failed: \(error)")
        return -1
    }
}

// MARK: - Cache I/O API

private func defaultSlotShapes(_ context: ModelContext) -> [[Int]] {
    [
        [1, 1, context.config.hiddenSize],
        [1, context.config.numHeads, context.config.headDim, context.config.headDim],
        [1, 1, context.config.hiddenSize],
    ]
}

private func slotShape(for shape: [Int]) -> [Int] {
    guard !shape.isEmpty else {
        return shape
    }
    return [1] + Array(shape.dropFirst())
}

private func slotShapes(for cache: KVCache, context: ModelContext) -> [[Int]] {
    let state = cache.state
    if state.isEmpty {
        return defaultSlotShapes(context)
    }
    return state.map { slotShape(for: $0.shape) }
}

private func elementCount(_ shape: [Int]) -> Int {
    shape.reduce(1, *)
}

private func cacheSlotSize(_ context: ModelContext, slot: Int) -> Int? {
    guard slot >= 0 else {
        setFFIError("slot must be non-negative")
        return nil
    }

    var totalSize = 0
    for cache in context.cache {
        for shape in slotShapes(for: cache, context: context) {
            totalSize += elementCount(shape) * 2
        }
    }
    return totalSize
}

private func installSlotArray(current: MLXArray?, slot: Int, slotArray: MLXArray) -> MLXArray {
    if let current {
        let target = resizeArrayBatch(current, batchSize: max(current.dim(0), slot + 1))
        target[slot ..< (slot + 1), 0...] = slotArray.asType(target.dtype)
        return target
    }

    var shape = slotArray.shape
    shape[0] = slot + 1
    let target = MLXArray.zeros(shape, dtype: slotArray.dtype)
    target[slot ..< (slot + 1), 0...] = slotArray
    return target
}

private func readCacheSlot(_ context: ModelContext, slot: Int, buffer: UnsafeMutableRawPointer?, bufferSize: Int32) -> Int32 {
    guard slot >= 0 else {
        setFFIError("slot must be non-negative")
        return -1
    }

    guard let buffer else {
        setFFIError("buffer is null")
        return -1
    }

    var slotArrays: [MLXArray] = []
    for cache in context.cache {
        let state = cache.state
        let arrays = state.isEmpty
            ? defaultSlotShapes(context).map { MLXArray.zeros($0, dtype: .float16) }
            : state

        for array in arrays {
            let slotArray: MLXArray
            if state.isEmpty {
                slotArray = array
            } else {
                guard slot < array.dim(0) else {
                    setFFIError("slot \(slot) out of range for cache batch size \(array.dim(0))")
                    return -1
                }
                slotArray = array[slot ..< (slot + 1), 0...]
            }

            slotArrays.append(slotArray.dtype == .float16 ? slotArray : slotArray.asType(.float16))
        }
    }

    guard slotArrays.reduce(0, { $0 + $1.nbytes }) <= Int(bufferSize) else {
        setFFIError("buffer too small for cache data")
        return -1
    }

    eval(slotArrays)
    var offset = 0
    let bufferPtr = buffer.assumingMemoryBound(to: UInt8.self)
    for array in slotArrays {
        withExtendedLifetime(array) {
            let arrayData = array.asData(access: .noCopyIfContiguous)
            let arraySize = arrayData.data.count
            _ = arrayData.data.withUnsafeBytes { bytes in
                memcpy(bufferPtr.advanced(by: offset), bytes.baseAddress!, arraySize)
            }
            offset += arraySize
        }
    }

    return Int32(offset)
}

private func writeCacheSlot(_ context: ModelContext, slot: Int, buffer: UnsafeRawPointer?, bufferSize: Int32) -> Int32 {
    guard slot >= 0 else {
        setFFIError("slot must be non-negative")
        return -1
    }

    guard let buffer else {
        setFFIError("buffer is null")
        return -1
    }

    let maybeExpectedSize = cacheSlotSize(context, slot: slot)
    guard let expectedSize = maybeExpectedSize, expectedSize == Int(bufferSize) else {
        setFFIError("buffer size mismatch: expected \(maybeExpectedSize ?? -1), got \(bufferSize)")
        return -1
    }

    var offset = 0
    let bufferPtr = buffer.assumingMemoryBound(to: UInt8.self)

    context.invalidateGreedy()
    for cacheIndex in 0..<context.cache.count {
        let state = context.cache[cacheIndex].state
        let shapes = slotShapes(for: context.cache[cacheIndex], context: context)
        let currentArrays: [MLXArray?] = state.isEmpty ? Array(repeating: nil, count: shapes.count) : state.map { Optional($0) }
        var cacheState: [MLXArray] = []

        for (arrayIndex, shape) in shapes.enumerated() {
            let expectedArraySize = elementCount(shape) * 2
            guard offset + expectedArraySize <= Int(bufferSize) else {
                setFFIError("buffer too small for cache data")
                return -1
            }

            let data = Data(bytes: bufferPtr.advanced(by: offset), count: expectedArraySize)
            let restoredSlot = MLXArray(data, shape, type: Float16.self)
            cacheState.append(installSlotArray(current: currentArrays[arrayIndex], slot: slot, slotArray: restoredSlot))
            offset += expectedArraySize
        }

        context.cache[cacheIndex].state = cacheState
    }

    guard offset == Int(bufferSize) else {
        setFFIError("buffer size mismatch: expected \(bufferSize), read \(offset)")
        return -1
    }

    return 0
}

private func zeroCacheSlot(_ context: ModelContext, slot: Int) -> Int32 {
    guard slot >= 0 else {
        setFFIError("slot must be non-negative")
        return -1
    }

    context.invalidateGreedy()
    for cacheIndex in 0..<context.cache.count {
        let state = context.cache[cacheIndex].state
        if rwkvLazyClearEnabled, context.model is Rwkv7Model, context.config.headDim == 64, state.isEmpty { continue }
        let shapes = slotShapes(for: context.cache[cacheIndex], context: context)
        let currentArrays: [MLXArray?] = state.isEmpty ? Array(repeating: nil, count: shapes.count) : state.map { Optional($0) }
        var cacheState: [MLXArray] = []

        for (arrayIndex, shape) in shapes.enumerated() {
            let dtype = currentArrays[arrayIndex]?.dtype ?? .float16
            let zeroSlot = MLXArray.zeros(shape, dtype: dtype)
            cacheState.append(installSlotArray(current: currentArrays[arrayIndex], slot: slot, slotArray: zeroSlot))
        }

        context.cache[cacheIndex].state = cacheState
    }

    return 0
}

@_cdecl("mlx_cache_get_size")
public func mlx_cache_get_size(_ handle: UnsafeMutableRawPointer?) -> Int32 {
    guard let handle else {
        setFFIError("handle is null")
        return -1
    }

    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    guard let totalSize = cacheSlotSize(context, slot: 0) else {
        return -1
    }
    return Int32(totalSize)
}

@_cdecl("mlx_cache_read")
public func mlx_cache_read(
    _ handle: UnsafeMutableRawPointer?,
    _ buffer: UnsafeMutableRawPointer?,
    _ bufferSize: Int32
) -> Int32 {
    guard let handle else {
        setFFIError("handle is null")
        return -1
    }

    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    return readCacheSlot(context, slot: 0, buffer: buffer, bufferSize: bufferSize)
}

@_cdecl("mlx_cache_write")
public func mlx_cache_write(
    _ handle: UnsafeMutableRawPointer?,
    _ buffer: UnsafeRawPointer?,
    _ bufferSize: Int32
) -> Int32 {
    guard let handle else {
        setFFIError("handle is null")
        return -1
    }

    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    return writeCacheSlot(context, slot: 0, buffer: buffer, bufferSize: bufferSize)
}

@_cdecl("mlx_cache_read_slot")
public func mlx_cache_read_slot(
    _ handle: UnsafeMutableRawPointer?,
    _ slot: Int32,
    _ buffer: UnsafeMutableRawPointer?,
    _ bufferSize: Int32
) -> Int32 {
    guard let handle else {
        setFFIError("handle is null")
        return -1
    }

    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    return readCacheSlot(context, slot: Int(slot), buffer: buffer, bufferSize: bufferSize)
}

@_cdecl("mlx_cache_write_slot")
public func mlx_cache_write_slot(
    _ handle: UnsafeMutableRawPointer?,
    _ slot: Int32,
    _ buffer: UnsafeRawPointer?,
    _ bufferSize: Int32
) -> Int32 {
    guard let handle else {
        setFFIError("handle is null")
        return -1
    }

    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    return writeCacheSlot(context, slot: Int(slot), buffer: buffer, bufferSize: bufferSize)
}

// Keep zero slots implicit until evaluation or a slot write materializes state.
@_cdecl("mlx_cache_zero_all")
public func mlx_cache_zero_all(_ handle: UnsafeMutableRawPointer?) -> Int32 {
    guard let handle else {
        setFFIError("handle is null")
        return -1
    }
    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    guard rwkvLazyClearEnabled, context.model is Rwkv7Model, context.config.headDim == 64 else { return -2 }
    context.invalidateGreedy()
    context.cache = context.model.newCache(parameters: nil)
    if rwkvLazyClearTrace { print("RWKV_LAZY_CLEAR allSlots=implicitZero") }
    return 0
}

@_cdecl("mlx_cache_zero_slot")
public func mlx_cache_zero_slot(
    _ handle: UnsafeMutableRawPointer?,
    _ slot: Int32
) -> Int32 {
    guard let handle else {
        setFFIError("handle is null")
        return -1
    }

    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    return zeroCacheSlot(context, slot: Int(slot))
}

// Experimental native diagnostic; no product caller or general sampling claim.
@_cdecl("mlx_greedy_pipeline_probe")
public func mlx_greedy_pipeline_probe(
    _ handle: UnsafeMutableRawPointer?, _ ids: UnsafePointer<Int32>?,
    _ idsLength: Int32, _ tokenCount: Int32, _ pipelined: Int32,
    _ output: UnsafeMutablePointer<Int32>?, _ logits: UnsafeMutablePointer<Float>?,
    _ timings: UnsafeMutablePointer<Double>?
) -> Int32 {
    guard let handle, let ids, let output, let logits, let timings,
        idsLength > 0, tokenCount > 0, tokenCount <= 8192,
        pipelined == 0 || pipelined == 1 else { return -1 }
    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    do {
        guard ensureCacheBatchSize(context, batchSize: 1) else { return -1 }
        let start = ProcessInfo.processInfo.systemUptime
        let input = MLXArray(Array(UnsafeBufferPointer(start: ids, count: Int(idsLength))))
        var result = try evaluateModel(context, input: LMInput(tokens: input))
        guard copyLastLogits(result, expectedBatchSize: 1, to: logits) == 0 else { return -1 }
        var token = argMax(result[0, -1, 0...]).asType(.int32)
        output[0] = token.item(Int32.self)
        let first = ProcessInfo.processInfo.systemUptime
        if tokenCount > 1 {
            for index in 1..<Int(tokenCount) {
                let previous = token
                result = try evaluateModel(context, input: LMInput(tokens: previous.reshaped(1)))
                token = argMax(result[0, -1, 0...]).asType(.int32)
                if pipelined == 1 {
                    // Submit the successor before reading the prior token on the CPU.
                    asyncEval(token)
                    if index > 1 { output[index - 1] = previous.item(Int32.self) }
                    if index == Int(tokenCount) - 1 { output[index] = token.item(Int32.self) }
                } else {
                    output[index] = token.item(Int32.self)
                }
            }
        }
        guard copyLastLogits(result, expectedBatchSize: 1, to: logits) == 0 else { return -1 }
        timings[0] = first - start
        timings[1] = ProcessInfo.processInfo.systemUptime - first
        return 0
    } catch {
        setFFIError("greedy pipeline probe failed: \(error)")
        return -1
    }
}

@_cdecl("mlx_model_eval_submit")
public func mlx_model_eval_submit(_ handle: UnsafeMutableRawPointer?, _ token: Int32) -> Int32 {
    guard let handle else { return -1 }
    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    guard !context.deferredEvalPending, token >= 0, Int(token) < context.config.vocabSize else { return -1 }
    guard context.model is Rwkv7Model,
        context.cache.allSatisfy({ !$0.state.isEmpty && $0.state.allSatisfy({ $0.dim(0) == 1 }) })
    else { return -2 }
    let previousState = context.cache.map { $0.state }
    context.invalidateGreedy()
    do {
        let result = try evaluateModel(context, input: LMInput(tokens: MLXArray([token])))
        guard result.ndim == 3, result.dim(0) == 1, result.dim(1) > 0,
            result.dim(2) == context.config.vocabSize else {
            throw NSError(domain: "MLXModelFFI", code: 6, userInfo: [NSLocalizedDescriptionKey: "invalid deferred logits shape"])
        }
        asyncEval(result)
        context.greedyLogits = result
        context.deferredEvalPending = true
        return 0
    } catch {
        for index in context.cache.indices { context.cache[index].state = previousState[index] }
        context.invalidateGreedy()
        setFFIError("deferred eval failed: \(error)")
        return -1
    }
}

@_cdecl("mlx_model_eval_finish")
public func mlx_model_eval_finish(_ handle: UnsafeMutableRawPointer?, _ logits: UnsafeMutablePointer<Float>?) -> Int32 {
    guard let handle, let logits else { return -1 }
    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    guard context.deferredEvalPending, let result = context.greedyLogits else { return -2 }
    let status = copyLastLogits(result, expectedBatchSize: 1, to: logits)
    context.deferredEvalPending = false
    return status
}

// One speculative token is retained so stop/cancel can restore the prior state.
@_cdecl("mlx_device_greedy_step")
public func mlx_device_greedy_step(
    _ handle: UnsafeMutableRawPointer?, _ outputToken: UnsafeMutablePointer<Int32>?
) -> Int32 {
    guard let handle, let outputToken else { return -1 }
    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    guard !context.deferredEvalPending, context.model is Rwkv7Model, let previousLogits = context.greedyLogits,
        previousLogits.ndim == 3, previousLogits.dim(0) == 1,
        previousLogits.dim(1) > 0, previousLogits.dim(2) == context.config.vocabSize,
        context.cache.allSatisfy({ !$0.state.isEmpty && $0.state.allSatisfy({ $0.dim(0) == 1 }) })
    else { return -2 }
    let previousState = context.cache.map { $0.state }
    let previousToken = context.greedyToken ?? argMax(previousLogits[0, -1, 0...]).asType(.int32)
    do {
        let result = try evaluateModel(context, input: LMInput(tokens: previousToken.reshaped(1)))
        let nextToken = argMax(result[0, -1, 0...]).asType(.int32)
        asyncEval(nextToken)
        outputToken.pointee = previousToken.item(Int32.self)
        context.greedyLogits = result
        context.greedyToken = nextToken
        context.greedyRollbackState = previousState
        context.greedyRollbackLogits = previousLogits
        return 0
    } catch {
        for index in context.cache.indices { context.cache[index].state = previousState[index] }
        context.invalidateGreedy()
        setFFIError("device greedy step failed: \(error)")
        return -1
    }
}

@_cdecl("mlx_device_greedy_finish")
public func mlx_device_greedy_finish(
    _ handle: UnsafeMutableRawPointer?, _ rollback: Int32,
    _ logits: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let handle, let logits, rollback == 0 || rollback == 1 else { return -1 }
    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    guard !context.deferredEvalPending else { return -2 }
    if rollback == 1 {
        guard let state = context.greedyRollbackState, let result = context.greedyRollbackLogits,
            state.count == context.cache.count else { return -1 }
        for index in context.cache.indices { context.cache[index].state = state[index] }
        context.greedyLogits = result
    }
    guard let result = context.greedyLogits else { return -2 }
    let status = copyLastLogits(result, expectedBatchSize: 1, to: logits)
    context.invalidateGreedy()
    return status
}

// The caller must verify this proposal with its unchanged CPU sampler before accepting it.
@_cdecl("mlx_device_sample_proposal")
public func mlx_device_sample_proposal(
    _ handle: UnsafeMutableRawPointer?, _ temperature: Float, _ topK: Int32,
    _ topP: Float, _ randomDraw: UInt32, _ biasIds: UnsafePointer<Int32>?,
    _ biases: UnsafePointer<Float>?, _ biasCount: Int32,
    _ outputToken: UnsafeMutablePointer<Int32>?, _ priorLogits: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let handle, let outputToken, let priorLogits, temperature.isFinite,
        topP.isFinite, topP > 0.0001, topP <= 1, topK > 1, topK <= 256,
        randomDraw >= 1, randomDraw <= 2147483646, biasCount >= 0,
        biasCount == 0 || (biasIds != nil && biases != nil) else { return -1 }
    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    guard !context.deferredEvalPending, context.model is Rwkv7Model,
        let previousLogits = context.greedyLogits,
        previousLogits.ndim == 3, previousLogits.dim(0) == 1,
        previousLogits.dim(1) > 0, previousLogits.dim(2) == context.config.vocabSize,
        Int(topK) <= context.config.vocabSize, Int(biasCount) <= context.config.vocabSize,
        context.cache.allSatisfy({ !$0.state.isEmpty && $0.state.allSatisfy({ $0.dim(0) == 1 }) })
    else { return -2 }
    var ids = [Int32](), values = [Float]()
    if biasCount > 0 {
        ids = Array(UnsafeBufferPointer(start: biasIds, count: Int(biasCount)))
        values = Array(UnsafeBufferPointer(start: biases, count: Int(biasCount)))
        guard ids.enumerated().allSatisfy({ index, id in
            id >= 0 && Int(id) < context.config.vocabSize && (index == 0 || ids[index - 1] < id)
        }), values.allSatisfy({ $0.isFinite }) else { return -1 }
    }
    let previousState = context.cache.map { $0.state }
    do {
        let verificationLogits = previousLogits[0, -1, 0...].asType(.float32)
        var adjusted = verificationLogits
        if !ids.isEmpty { adjusted = adjusted.at[MLXArray(ids)].add(MLXArray(values)) }
        let indices = argSort(-adjusted)[0..<Int(topK)]
        let selected = take(adjusted, indices, axis: 0)
        let weights = exp((selected - selected[0]) / min(5, max(0.1, temperature)))
        let cumulative = cumsum(weights / sum(weights))
        let cutoff = minimum(sum((cumulative .< topP).asType(.int32)), MLXArray(topK - 1))
        let target = cumulative[cutoff] * Float(randomDraw - 1) / Float(2147483645)
        let choice = minimum(sum((cumulative .< target).asType(.int32)), cutoff)
        let proposal = indices[choice].asType(.int32)
        asyncEval(proposal)
        let result = try evaluateModel(context, input: LMInput(tokens: proposal.reshaped(1)))
        asyncEval(result)
        // Reuse the already submitted view; a new slice would queue behind the next forward.
        let verificationData = verificationLogits.asData(access: .copy)
        verificationData.data.withUnsafeBytes { bytes in
            priorLogits.initialize(from: bytes.bindMemory(to: Float.self).baseAddress!, count: context.config.vocabSize)
        }
        outputToken.pointee = proposal.item(Int32.self)
        context.greedyLogits = result
        context.greedyToken = nil
        context.greedyRollbackState = previousState
        context.greedyRollbackLogits = previousLogits
        return 0
    } catch {
        for index in context.cache.indices { context.cache[index].state = previousState[index] }
        context.invalidateGreedy()
        setFFIError("device sample proposal failed: \(error)")
        return -1
    }
}
