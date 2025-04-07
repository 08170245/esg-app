from flask import Flask, request, jsonify,render_template
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import os
import csv
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
import exifread
from pillow_heif import register_heif_opener
import datetime

app = Flask(__name__)
@app.route('/')

def index():
  return render_template('app_test.html')

# Load the trained model
model_path = './vgg_vegetable_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.vgg19()  # If you saved only the state_dict

# model = torch.load('complete_vgg_vegetable_model_model.pth') # If you saved the complete model
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# Image transformations
transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 选择模型中要获取特征的层
layer = model.classifier[5]  # 例如，选择分类器的第一个层

# 新增的檢測文件類型並適當處理的函數
def process_image(image_path):
    file_extension = os.path.splitext(image_path)[-1].lower()

    if file_extension == '.heic':
        # 處理HEIC檔案
        register_heif_opener()
        image = Image.open(image_path)
        exif_data = image.info.get('exif', b'')
        converted_path = image_path.replace('.HEIC', '.jpg')
        image.save(converted_path, "JPEG", exif=exif_data)
        return converted_path
    else:
        # 對於JPG和PNG，直接返回原路徑
        return image_path

def exif_information(image_path):
    with open(image_path, 'rb') as image_file:
        exif_data = exifread.process_file(image_file)

    pictureTime = exif_data.get('Image DateTime')
    pictureCamera = exif_data.get('Image Model')
    pictureIosTime = exif_data.get('EXIF DateTimeOriginal')
    pictureIosCamera = exif_data.get('EXIF LensModel')
    currentTime = datetime.datetime.now()
    #print('exif_data: ', exif_data)
    print('pictureCamera: ', pictureCamera)
    print('pictureIosCamera: ', pictureIosCamera)
    print('pictureIosTime: ', pictureIosTime)

    # 檢查相機資訊
    if not pictureCamera and not pictureIosCamera:
        no_camera = 'no camera'
        return  no_camera# 提前退出函数

    # 檢查照片時間
    if not pictureTime and not pictureIosTime:
        no_time = 'no time'
        return  no_time# 提前退出函数

    if pictureTime :    # 判斷 Android 手機
        pictureTime = datetime.datetime.strptime(str(pictureTime), '%Y:%m:%d %H:%M:%S')
    elif pictureIosTime :    # 判斷 IOS 手機
        pictureTime = datetime.datetime.strptime(str(pictureIosTime), '%Y:%m:%d %H:%M:%S')
    else:
        too_much_time = 'too much time'
        return too_much_time

    time_difference = currentTime - pictureTime
    if time_difference.total_seconds() > 14400:
        too_much_time = 'too much time'
        return too_much_time 
    else:
        return True


# 定義提取特徵向量的函数
def get_feature_vector(image_path, layer):
    # 定義一個鉤子函数
    def hook(model, input, output):
        global feature_vector
        feature_vector = output.detach()

    handle = layer.register_forward_hook(hook)

    image = Image.open(image_path).convert('RGB')
    image = transformer(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        _ = model(image)

    # 移除钩子
    handle.remove()
    return feature_vector

# Load feature vectors from a CSV file
def load_feature_vectors(csv_file_path):
    if os.path.isfile(csv_file_path):
        with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader, None)  # skip the header
            return np.array(list(reader)).astype(float)
    else:
        # Return an empty array if the file does not exist
        return np.empty((0, 4096))  # 4096 是 VGG16 最后一层的特征数量，请根据您的模型进行调整

# Check for similarity using KNN
def check_similarity(new_feature_vector, all_feature_vectors, threshold=0.8):
    k = min(5, len(all_feature_vectors))
    if k == 0:
        return False

    neighbors = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(all_feature_vectors)
    distances, indices = neighbors.kneighbors([new_feature_vector])
    return np.any(distances < threshold)

# Save feature vector to a CSV file
def save_feature_vector_to_csv(feature_vector, file_path):
    feature_vector_list = feature_vector.cpu().numpy().flatten().tolist()
    file_exists = os.path.isfile(file_path)
    mode = 'a' if file_exists else 'w'

    with open(file_path, mode, newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['feature' + str(i) for i in range(len(feature_vector_list))])
        writer.writerow(feature_vector_list)

@app.route('/predict',methods=[ "GET",'POST'])
def predict():
    data = request.get_json()
    # 收到的json沒問題（內容有id,user_id,path ）
    if data and all(k in data for k in ["id", "user_id", "path"]):
        base_nas_path = "/home/user/vegetarians/"  # 假設的NAS路徑

        image_path = os.path.join(base_nas_path, data["path"]) #完整圖片路徑
        print(image_path)

        # 確認路徑底下有沒有向量csv
        image_directory = os.path.dirname(image_path) # 圖片所在目錄
        print(image_directory)

        csv_file_path = os.path.join(image_directory, "feature.csv") #csv完整路徑
        print(csv_file_path)
        
        # 檢查是否存在feature.csv文件，如果不存在就建立一个空的csv文件
        if not os.path.exists(csv_file_path):
            with open(csv_file_path, 'w', newline='') as file:
                writer = csv.writer(file)

        #path路徑中確定是圖片
        if os.path.exists(image_path):
            try:
                # 嘗試載入圖片
                image = Image.open(image_path).convert('RGB')
            except UnidentifiedImageError:
                # 如果文件不是圖片格式
                return jsonify({'error': 'File is not an image'}), 400
            except Exception as e:
                # 處理其他錯誤
                return jsonify({'error': 'Error loading the image', 'details': str(e)}), 500
            
            image_path = process_image(image_path)
            exif_info = exif_information(image_path)
            message_prefix = "發生錯誤:"

            if exif_info in ['no camera', 'no time', 'too much time']:
            # if exif_info in ['no time', 'too much time']:
                error_messages = {
                    'no camera': "無法獲取相機資訊。",
                    'no time': "無法獲取照片拍攝時間。",
                    'too much time': "照片拍攝時間和當前時間相差超過4小時。"
                }
                return jsonify({
                    "result": False,
                    "message": f"{message_prefix}{error_messages[exif_info]}",
                    "data": {
                    "id": data["id"],
                    "path": data["path"],
                    "user_id": data["user_id"],
                        "possibility": None,
                        "is_vegetarian": None
                    }
                })

            image = Image.open(image_path).convert('RGB')
            image = transformer(image).unsqueeze(0)  # Add batch dimension
            image = image.to(device)

            with torch.no_grad():
                logits = model(image)
                probabilities = F.softmax(logits, dim=1)  # Apply softmax to get probabilities
                _, predicted = torch.max(probabilities, 1)

            predicted_idx = int(predicted.item())
            predicted_prob = probabilities[0][predicted_idx].item()

            if predicted_idx == 0:
                # catch feature vectors
                feature_vec = get_feature_vector(image_path, layer)
                # print('特征向量:',feature_vec)

                # Load all feature vectors
                all_feature_vectors = load_feature_vectors(csv_file_path)

                # Check the similarity of the new image
                new_feature_vector = feature_vec.cpu().numpy().flatten()
                is_similar = check_similarity(new_feature_vector, all_feature_vectors)
                if not is_similar:
                    result_data = {
                        "id": data["id"],
                        "path": data["path"],
                        "user_id": data["user_id"],
                        "possibility": predicted_prob,
                        "is_vegetarian": 'true' if predicted_prob >= 0.7 else 'false'
                    }
                    # 由于图片是独特的，保存其特征向量
                    save_feature_vector_to_csv(feature_vec, csv_file_path)
                else:
                    return jsonify({
                        "result": False,
                        "message": "圖片以上傳過，若有疑問請洽客服。",
                        "data": {
                            "id": data["id"],
                            "path": data["path"],
                            "user_id": data["user_id"], 
                            "possibility": None,
                            "is_vegetarian": None
                        }
                    })

            else:
                result_data = {
                    "id": data["id"],
                    "path": data["path"],
                    "user_id": data["user_id"],
                    "possibility": predicted_prob,
                    "is_vegetarian": 'false'
                }

            return jsonify({
                "result": True,
                "message": "成功",
                "data": result_data
            })
        
        else:
            return jsonify({'error': 'No image provided'})
    else:
        return jsonify({'error': 'Request JSON must contain id, user_id, and path'})
    
if __name__ == '__main__':
    app.run(host="0.0.0.0" ,port=5346)
