# YOLOv12s를 이용한 음식 재료 탐지 프로젝트

이 프로젝트는 YOLOv12s 모델을 사용하여 다양한 음식 재료를 탐지하는 딥러닝 프로젝트입니다. Roboflow의 'FOOD INGREDIENTS detection' 데이터셋으로 모델을 학습시켰습니다.


## 📚 데이터셋 (Dataset)

이 프로젝트는 [Roboflow Universe의 'FOOD INGREDIENTS detection' 데이터셋](https://universe.roboflow.com/food-detection-d7q6o/food-ingredients-detection-6ce7j/dataset/1)을 사용했습니다.

**총 120개의 클래스**를 포함하고 있으며, 주요 클래스는 다음과 같습니다:
`Akabare Khursani`, `Apple`, `Artichoke`, `Ash Gourd -Kubhindo-`, `Asparagus -Kurilo-`, `Avocado`, `Bacon`, `Bamboo Shoots -Tama-`, `Banana`, `Beans`, `Beaten Rice -Chiura-`, `Beef`, `Beetroot`, `Bethu ko Saag`, `Bitter Gourd`, `Black Lentils`, `Black beans`, `Bottle Gourd -Lauka-`, `Bread`, `Brinjal`, `Broad Beans -Bakullo-`, `Broccoli`, `Buff Meat`, `Butter`, `Cabbage`, `Capsicum`, `Carrot`, `Cassava -Ghar Tarul-`, `Cauliflower`, `Chayote-iskus-`, `Cheese`, `Chicken`, `Chicken Gizzards`, `Chickpeas`, `Chili Pepper -Khursani-`, `Chili Powder`, `Chowmein Noodles`, `Cinnamon`, `Coriander -Dhaniya-`, `Corn`, `Cornflakec`, `Crab Meat`, `Cucumber`, `Egg`, `Farsi ko Munta`, `Fiddlehead Ferns -Niguro-`, `Fish`, `Garden Peas`, `Garden cress-Chamsur ko saag-`, `Garlic`, `Ginger`, `Green Brinjal`, `Green Lentils`, `Green Mint -Pudina-`, `Green Peas`, `Green Soyabean -Hariyo Bhatmas-`, `Gundruk`, `Ham`, `Ice`, `Jack Fruit`, `Ketchup`, `Lapsi -Nepali Hog Plum-`, `Lemon -Nimbu-`, `Lime -Kagati-`, `Long Beans -Bodi-`, `Masyaura`, `Milk`, `Minced Meat`, `Moringa Leaves -Sajyun ko Munta-`, `Mushroom`, `Mutton`, `Nutrela -Soya Chunks-`, `Okra -Bhindi-`, `Olive Oil`, `Onion`, `Onion Leaves`, `Orange`, `Palak -Indian Spinach-`, `Palungo -Nepali Spinach-`, `Paneer`, `Papaya`, `Pea`, `Pear`, `Pointed Gourd -Chuche Karela-`, `Pork`, `Potato`, `Pumpkin -Farsi-`, `Radish`, `Rahar ko Daal`, `Rayo ko Saag`, `Red Beans`, `Red Lentils`, `Rice -Chamal-`, `Sajjyun -Moringa Drumsticks-`, `Salt`, `Sausage`, `Snake Gourd -Chichindo-`, `Soy Sauce`, `Soyabean -Bhatmas-`, `Sponge Gourd -Ghiraula-`, `Stinging Nettle -Sisnu-`, `Strawberry`, `Sugar`, `Sweet Potato -Suthuni-`, `Taro Leaves -Karkalo-`, `Taro Root-Pidalu-`, `Thukpa Noodles`, `Tofu`, `Tomato`, `Tori ko Saag`, `Tree Tomato -Rukh Tamatar-`, `Turnip`, `Wallnut`, `Water Melon`, `Wheat`, `Yellow Lentils`, `kimchi`, `mayonnaise`, `noodle`, `seaweed`

## ⚙️ 설치 및 실행 방법 (Installation & Usage)

### 1. 필요 라이브러리 설치

```bash
pip install ultralytics supervision roboflow
```

### 2. 모델 학습 (Training)

`train_yolo12_object_detection_on.ipynb` 노트북 파일의 코드를 참고하여 학습을 진행할 수 있습니다. 핵심 학습 명령어는 다음과 같습니다.

```bash
yolo task=detect mode=train model=yolo12s.pt data='raw_Data.FOOD_INGREDIENTS detection.v1i.yolov8/data.yaml' epochs=400 imgsz=640 batch=40

```
## 📊 모델 학습 결과 (Training Results)

학습은 400 epoch 동안 진행되었으며, 결과는 다음과 같습니다.

| ![Results](train/train/results.png) | ![Confusion Matrix](train/train/confusion_matrix.png) |
| :---------------------------------: | :----------------------------------------------------: |
|        **Training Results**         |                **Confusion Matrix**                  |

| ![학습 결과 데이터 보기 (results.csv)](https://drive.google.com/drive/folders/1s0oqx0szBrfxxXTw9QbN92m79h9vjQsw) |


### 3. 예측 (Inference)

학습된 모델(`best.pt`)을 사용하여 새로운 이미지에 대한 예측을 수행할 수 있습니다.

```bash
yolo task=detect mode=predict model='train/train/weights/best.pt' source='predict_images/'
```
* `source` 인자에는 예측하고 싶은 이미지나 동영상이 있는 폴더 경로를 지정합니다.

### 🚀 데모 (Prediction Examples)

아래는 학습된 모델로 이미지를 예측한 결과입니다.

| ![예측 결과 1](predict_result/스크린샷%202025-08-08%20104133.png) | ![예측 결과 2](predict_result/스크린샷%202025-08-08%20104143.png) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![예측 결과 3](predict_result/스크린샷%202025-08-08%20104157.png) | ![예측 결과 4](predict_result/2-3.png) |

---

### 데모 분석 (Prediction Examples analysis)

1. 과적합?
| ![mAP50-95 Curve](train/train/results.csv_mAP50-95.jpg) |

2. 데이터셋 train과 valid data의 유사성?
| ![Garlic Train/Valid Image](train/train/garlic_train_valid_image.jpg) |

3. 데이터셋 train data의 bounding box 분산?
| ![Train Data Variance](train/train/train_data_variance.png) |