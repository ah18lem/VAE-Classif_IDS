#include <Arduino.h>
#include "model_data.h"
#include "esp32_test_data.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define INPUT_DIM 18
#define NUM_CLASSES 5
#define OPS_PER_INFERENCE 1276

constexpr int ARENA_SIZE = 200 * 1024;
alignas(16) uint8_t arena[ARENA_SIZE];

const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interp = nullptr;
TfLiteTensor *input = nullptr, *output = nullptr;
tflite::MicroMutableOpResolver<5> resolver;

bool done = false;

float dequant(TfLiteTensor* t, int i) {
  if (t->type == kTfLiteFloat32) return t->data.f[i];
  if (t->type == kTfLiteInt8) return (t->data.int8[i] - t->params.zero_point) * t->params.scale;
  if (t->type == kTfLiteUInt8) return (t->data.uint8[i] - t->params.zero_point) * t->params.scale;
  return 0.0f;
}

int quant(float x, TfLiteTensor* t, int min_v, int max_v) {
  int q = round(x / t->params.scale + t->params.zero_point);
  return constrain(q, min_v, max_v);
}

bool fill_input(const float x[INPUT_DIM]) {
  if (!input) return false;

  for (int i = 0; i < INPUT_DIM; i++) {
    if (input->type == kTfLiteFloat32) input->data.f[i] = x[i];
    else if (input->type == kTfLiteInt8) input->data.int8[i] = (int8_t)quant(x[i], input, -128, 127);
    else if (input->type == kTfLiteUInt8) input->data.uint8[i] = (uint8_t)quant(x[i], input, 0, 255);
    else return false;
  }

  return true;
}

int predict(const float x[INPUT_DIM], float& conf, uint32_t& us) {
  if (!fill_input(x)) return -1;

  uint32_t t0 = micros();
  if (interp->Invoke() != kTfLiteOk) return -1;
  us = micros() - t0;

  int pred = 0;
  conf = dequant(output, 0);

  for (int i = 1; i < NUM_CLASSES; i++) {
    float s = dequant(output, i);
    if (s > conf) {
      conf = s;
      pred = i;
    }
  }

  return pred;
}

void setup_tflite() {
  tfl_model = tflite::GetModel(g_model);

  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Erreur version TFLite");
    while (true) delay(1000);
  }

  // Ops modèle
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();

  static tflite::MicroInterpreter static_interp(tfl_model, resolver, arena, ARENA_SIZE);
  interp = &static_interp;

  if (interp->AllocateTensors() != kTfLiteOk) {
    Serial.println("Erreur AllocateTensors");
    while (true) delay(1000);
  }

  input = interp->input(0);
  output = interp->output(0);

  Serial.print("model_size_bytes,"); Serial.println(g_model_len);
  Serial.print("input_type,"); Serial.println(input->type);
  Serial.print("output_type,"); Serial.println(output->type);
}

void print_csv(String key, float value, int digits = 6) {
  Serial.print(key);
  Serial.print(",");
  Serial.println(value, digits);
}

void run_test() {
  int cm[NUM_CLASSES][NUM_CLASSES] = {};
  int total[NUM_CLASSES] = {};
  int correct[NUM_CLASSES] = {};
  float conf_sum[NUM_CLASSES] = {};

  uint64_t total_us = 0;

  Serial.println("ESP32_PER_SAMPLE_BEGIN");
  Serial.println("row_index,true_class_id,true_label,predicted_class_id,predicted_label,confidence,inference_time_us,inference_time_ms,correct");

  for (int i = 0; i < TEST_NUM_SAMPLES; i++) {
    int y = TEST_Y[i];
    float conf = 0.0f;
    uint32_t us = 0;
    int pred = predict(TEST_X[i], conf, us);

    if (pred < 0 || pred >= NUM_CLASSES) {
      pred = 0;
      conf = 0.0f;
    }

    bool ok = pred == y;
    cm[y][pred]++;
    total[y]++;
    correct[y] += ok;
    conf_sum[y] += conf;
    total_us += us;

    Serial.print(i); Serial.print(",");
    Serial.print(y); Serial.print(",");
    Serial.print(TEST_CLASS_NAMES[y]); Serial.print(",");
    Serial.print(pred); Serial.print(",");
    Serial.print(TEST_CLASS_NAMES[pred]); Serial.print(",");
    Serial.print(conf, 6); Serial.print(",");
    Serial.print(us); Serial.print(",");
    Serial.print(us / 1000.0f, 6); Serial.print(",");
    Serial.println(ok ? 1 : 0);
  }

  Serial.println("ESP32_PER_SAMPLE_END");

  float macro_p = 0, macro_r = 0, macro_f1 = 0;
  int total_ok = 0;

  for (int c = 0; c < NUM_CLASSES; c++) {
    int tp = cm[c][c], fp = 0, fn = 0;

    for (int k = 0; k < NUM_CLASSES; k++) {
      if (k != c) {
        fp += cm[k][c];
        fn += cm[c][k];
      }
    }

    float p = (tp + fp) ? (float)tp / (tp + fp) : 0.0f;
    float r = (tp + fn) ? (float)tp / (tp + fn) : 0.0f;
    float f1 = (p + r) ? 2 * p * r / (p + r) : 0.0f;

    macro_p += p;
    macro_r += r;
    macro_f1 += f1;
    total_ok += tp;
  }

  macro_p /= NUM_CLASSES;
  macro_r /= NUM_CLASSES;
  macro_f1 /= NUM_CLASSES;

  float avg_ms = ((float)total_us / TEST_NUM_SAMPLES) / 1000.0f;
  float samples_s = 1000000.0f * TEST_NUM_SAMPLES / total_us;
  float ops_s = samples_s * OPS_PER_INFERENCE;
  float acc = (float)total_ok / TEST_NUM_SAMPLES;

  Serial.println("Resultats");
  print_csv("num_samples", TEST_NUM_SAMPLES, 0);
  print_csv("model_size_bytes", g_model_len, 0);
  print_csv("avg_inference_time_ms", avg_ms);
  print_csv("ops_per_inference", OPS_PER_INFERENCE, 0);
  print_csv("ops_per_second", ops_s, 2);
  print_csv("samples_per_second", samples_s, 4);
  print_csv("accuracy", acc);
  print_csv("precision_macro", macro_p);
  print_csv("recall_macro", macro_r);
  print_csv("macro_f1", macro_f1);
  Serial.println("ESP32_SUMMARY_END");

  Serial.println("ESP32_RELIABILITY_BEGIN");
  Serial.println("class_id,class,reliability_score,correct,total,mean_confidence");

  for (int c = 0; c < NUM_CLASSES; c++) {
    float rel = total[c] ? (float)correct[c] / total[c] : 0.0f;
    float mc = total[c] ? conf_sum[c] / total[c] : 0.0f;

    Serial.print(c); Serial.print(",");
    Serial.print(TEST_CLASS_NAMES[c]); Serial.print(",");
    Serial.print(rel, 6); Serial.print(",");
    Serial.print(correct[c]); Serial.print(",");
    Serial.print(total[c]); Serial.print(",");
    Serial.println(mc, 6);
  }

  Serial.println("ESP32_RELIABILITY_END");
}

void setup() {
  Serial.begin(115200);
  delay(3000);

  Serial.println("ESP32 IDS");
  setup_tflite();

  Serial.print("samples,");
  Serial.println(TEST_NUM_SAMPLES);
}

void loop() {
  if (!done) {
    done = true;
    run_test();
    Serial.println("Test terminé");
  }

  delay(1000);
}
