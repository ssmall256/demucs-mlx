// Zero-GIL native Apple Neural Engine / Core ML dispatch for Demucs.
// Dispatches batch inference via Grand Central Dispatch with zero Python lock contention.

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include <dispatch/dispatch.h>
#include <stdlib.h>
#include <string.h>

static MLModel* s_conv_model = nil;
static NSString* s_loaded_path = nil;

int init_ane_conv(const char* model_path) {
    @autoreleasepool {
        NSString* pathStr = [NSString stringWithUTF8String:model_path];
        if (s_conv_model && [s_loaded_path isEqualToString:pathStr]) {
            return 0;
        }
        NSURL* url = [NSURL fileURLWithPath:pathStr];
        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        NSError* error = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:url configuration:config error:&error];
        if (!model) {
            NSLog(@"[demucs-ane] Error loading Core ML model: %@", error);
            return -1;
        }
        s_conv_model = model;
        s_loaded_path = pathStr;
        return 0;
    }
}

int predict_conv_batch(const float* input, __fp16* output, int total_chunks) {
    if (!s_conv_model) return -10;
    if (total_chunks <= 0) return 0;

    int pairs = (total_chunks + 1) / 2;
    __block int status = 0;

    dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    dispatch_apply(pairs, queue, ^(size_t p) {
        @autoreleasepool {
            size_t c_idx = p * 2;
            int count = (c_idx + 1 < total_chunks) ? 2 : 1;

            const float* in_ptr = input + c_idx * 2 * 343980;
            __fp16* out_ptr = output + c_idx * 48 * 85995;

            NSArray<NSNumber*>* shape_in = @[@2, @2, @343980];
            NSArray<NSNumber*>* strides_in = @[@687960, @343980, @1];
            NSError* error = nil;

            const float* actual_in = in_ptr;
            float* pad_in = NULL;
            if (count == 1) {
                pad_in = (float*)malloc(2 * 2 * 343980 * sizeof(float));
                if (!pad_in) { status = -11; return; }
                memcpy(pad_in, in_ptr, 2 * 343980 * sizeof(float));
                memcpy(pad_in + 2 * 343980, in_ptr, 2 * 343980 * sizeof(float));
                actual_in = pad_in;
            }

            MLMultiArray* in_arr = [[MLMultiArray alloc] initWithDataPointer:(void*)actual_in
                                                                       shape:shape_in
                                                                    dataType:MLMultiArrayDataTypeFloat32
                                                                     strides:strides_in
                                                                 deallocator:nil
                                                                       error:&error];
            if (!in_arr) { status = -2; if (pad_in) free(pad_in); return; }

            NSArray<NSNumber*>* shape_out = @[@2, @48, @85995];
            NSArray<NSNumber*>* strides_out = @[@4127760, @85995, @1];

            __fp16* actual_out = out_ptr;
            __fp16* pad_out = NULL;
            if (count == 1) {
                pad_out = (__fp16*)malloc(2 * 48 * 85995 * sizeof(__fp16));
                if (!pad_out) { status = -12; if (pad_in) free(pad_in); return; }
                actual_out = pad_out;
            }

            MLMultiArray* out_arr = [[MLMultiArray alloc] initWithDataPointer:(void*)actual_out
                                                                        shape:shape_out
                                                                     dataType:MLMultiArrayDataTypeFloat16
                                                                      strides:strides_out
                                                                  deallocator:nil
                                                                        error:&error];
            if (!out_arr) {
                status = -3;
                if (pad_in) free(pad_in);
                if (pad_out) free(pad_out);
                return;
            }

            NSDictionary* dict = @{@"mix": [MLFeatureValue featureValueWithMultiArray:in_arr]};
            MLDictionaryFeatureProvider* feat = [[MLDictionaryFeatureProvider alloc] initWithDictionary:dict error:&error];
            if (!feat) {
                status = -4;
                if (pad_in) free(pad_in);
                if (pad_out) free(pad_out);
                return;
            }

            MLPredictionOptions* options = [[MLPredictionOptions alloc] init];
            [options setOutputBackings:@{@"y0": out_arr}];

            id<MLFeatureProvider> res = [s_conv_model predictionFromFeatures:feat options:options error:&error];
            if (!res) {
                NSLog(@"[demucs-ane] Prediction failed: %@", error);
                status = -5;
            } else if (count == 1) {
                memcpy(out_ptr, pad_out, 48 * 85995 * sizeof(__fp16));
            }

            if (pad_in) free(pad_in);
            if (pad_out) free(pad_out);
        }
    });

    return status;
}
