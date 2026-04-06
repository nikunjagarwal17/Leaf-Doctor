# Plant Disease Detection Model - Benchmark Report

**Generated:** 2026-04-06 13:12:13  
**Model:** plant_disease_model_v2.pt  
**Dataset:** Plant_leave_diseases_dataset_with_augmentation  

## Summary

| Metric | Value |
|--------|-------|
| Accuracy | 0.9527 |
| Macro F1 | 0.9411 |
| Weighted F1 | 0.9526 |
| Classes | 39 |
| Test Samples | 9,223 |
| Inference Latency | 2.27ms (±0.77ms) |

## Aggregate Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 0.9527 |
| Macro Precision | 0.9438 |
| Macro Recall | 0.9430 |
| Macro F1 | 0.9411 |
| Weighted F1 | 0.9526 |

## Per-Class Performance (All 39 Classes)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------| 
| Apple___Apple_scab | 0.9045 | 0.9530 | 0.9281 | 149 |
| Apple___Black_rot | 0.9928 | 0.8671 | 0.9257 | 158 |
| Apple___Cedar_apple_rust | 0.9675 | 0.9933 | 0.9803 | 150 |
| Apple___healthy | 0.9610 | 0.9447 | 0.9528 | 235 |
| Background_without_leaves | 0.9899 | 0.9657 | 0.9777 | 204 |
| Blueberry___healthy | 0.9911 | 0.9610 | 0.9758 | 231 |
| Cherry___Powdery_mildew | 0.9931 | 0.9728 | 0.9828 | 147 |
| Cherry___healthy | 0.8129 | 0.9789 | 0.8882 | 142 |
| Corn___Cercospora_leaf_spot Gray_leaf_spot | 0.9239 | 0.6296 | 0.7489 | 135 |
| Corn___Common_rust | 1.0000 | 0.9904 | 0.9952 | 208 |
| Corn___Northern_Leaf_Blight | 0.7268 | 0.9527 | 0.8246 | 148 |
| Corn___healthy | 0.9649 | 1.0000 | 0.9821 | 165 |
| Grape___Black_rot | 0.9200 | 0.9641 | 0.9415 | 167 |
| Grape___Esca_(Black_Measles) | 0.9945 | 0.9239 | 0.9579 | 197 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 0.9878 | 0.9939 | 0.9908 | 163 |
| Grape___healthy | 0.9742 | 0.9618 | 0.9679 | 157 |
| Orange___Haunglongbing_(Citrus_greening) | 0.9920 | 1.0000 | 0.9960 | 863 |
| Peach___Bacterial_spot | 0.9822 | 0.9457 | 0.9636 | 350 |
| Peach___healthy | 0.9357 | 0.9938 | 0.9639 | 161 |
| Pepper,_bell___Bacterial_spot | 0.9294 | 0.9753 | 0.9518 | 162 |
| Pepper,_bell___healthy | 0.9785 | 0.9620 | 0.9702 | 237 |
| Potato___Early_blight | 0.9790 | 0.9589 | 0.9689 | 146 |
| Potato___Late_blight | 0.8288 | 0.9453 | 0.8832 | 128 |
| Potato___healthy | 0.8418 | 0.9868 | 0.9085 | 151 |
| Raspberry___healthy | 1.0000 | 0.9037 | 0.9494 | 135 |
| Soybean___healthy | 0.9798 | 0.9604 | 0.9700 | 757 |
| Squash___Powdery_mildew | 0.9926 | 0.9889 | 0.9907 | 270 |
| Strawberry___Leaf_scorch | 0.9662 | 0.9167 | 0.9408 | 156 |
| Strawberry___healthy | 0.9667 | 0.9539 | 0.9603 | 152 |
| Tomato___Bacterial_spot | 0.9342 | 0.9739 | 0.9536 | 306 |
| Tomato___Early_blight | 0.8780 | 0.7347 | 0.8000 | 147 |
| Tomato___Late_blight | 0.9451 | 0.8142 | 0.8748 | 296 |
| Tomato___Leaf_Mold | 0.9632 | 0.9097 | 0.9357 | 144 |
| Tomato___Septoria_leaf_spot | 0.8412 | 0.9614 | 0.8973 | 259 |
| Tomato___Spider_mites Two-spotted_spider_mite | 0.9395 | 0.9549 | 0.9472 | 244 |
| Tomato___Target_Spot | 0.8930 | 0.9320 | 0.9121 | 206 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 0.9874 | 0.9836 | 0.9855 | 795 |
| Tomato___Tomato_mosaic_virus | 0.9542 | 0.9865 | 0.9701 | 148 |
| Tomato___healthy | 0.9960 | 0.9803 | 0.9881 | 254 |


## Top 10 Best Classes (F1)

1. **Orange___Haunglongbing_(Citrus_greening)**: F1=0.9960
2. **Corn___Common_rust**: F1=0.9952
3. **Grape___Leaf_blight_(Isariopsis_Leaf_Spot)**: F1=0.9908
4. **Squash___Powdery_mildew**: F1=0.9907
5. **Tomato___healthy**: F1=0.9881
6. **Tomato___Tomato_Yellow_Leaf_Curl_Virus**: F1=0.9855
7. **Cherry___Powdery_mildew**: F1=0.9828
8. **Corn___healthy**: F1=0.9821
9. **Apple___Cedar_apple_rust**: F1=0.9803
10. **Background_without_leaves**: F1=0.9777


## Top 10 Worst Classes (F1)

1. **Corn___Cercospora_leaf_spot Gray_leaf_spot**: F1=0.7489
2. **Tomato___Early_blight**: F1=0.8000
3. **Corn___Northern_Leaf_Blight**: F1=0.8246
4. **Tomato___Late_blight**: F1=0.8748
5. **Potato___Late_blight**: F1=0.8832
6. **Cherry___healthy**: F1=0.8882
7. **Tomato___Septoria_leaf_spot**: F1=0.8973
8. **Potato___healthy**: F1=0.9085
9. **Tomato___Target_Spot**: F1=0.9121
10. **Apple___Black_rot**: F1=0.9257


## Latency Benchmark (ms)

| Metric | Value |
|--------|-------|
| Mean | 2.2707 |
| Std | 0.7720 |
| P50 | 2.7255 |
| P95 | 2.8852 |
| P99 | 2.9127 |

## Config

- Epochs: 30
- Batch Size: 64
- Learning Rate: 0.001
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Seed: 42

## Files Generated

- benchmark_report.json
- benchmark_report_confusion_matrix.csv
- benchmark_report_per_class.csv
- benchmark_report.png
- training_history.png
- BENCHMARK.md (this file)
