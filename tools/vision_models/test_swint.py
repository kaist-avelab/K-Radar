import torch
from torchvision.models import swin_t, swin_b
from torchvision import transforms
from PIL import Image
from torchvision.models import Swin_T_Weights, Swin_B_Weights

def load_and_preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return transform(image).unsqueeze(0)

def get_prediction(model, image_tensor, categories):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    return [(categories[top5_catid[i]], float(top5_prob[i])) for i in range(5)]

def main():
    # 모델과 클래스 레이블 로드
    weights = Swin_B_Weights.DEFAULT
    model = swin_b(weights=weights)
    
    categories = weights.meta["categories"]
    
    # 이미지 경로
    image_path = './tools/vision_models/car.jpg'
    
    # 이미지 전처리
    image_tensor = load_and_preprocess_image(image_path)

    # Case 2 (K-Radar)
    image_path = './tools/vision_models/kradar_seq1.png'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # 예측
    # predictions = get_prediction(model, image_tensor, categories)
    
    # # 결과 출력
    # print("\nTop 5 predictions:")
    # for class_name, prob in predictions:
    #     print(f"{class_name}: {prob:.3f}")

    import torch.nn as nn
    feature_model = nn.Sequential(
        model.features[:4]
    )
    
    print(len(model.features))

    out_feature = feature_model(image_tensor)

    print(image_tensor.shape)
    print(out_feature.shape)
    
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    img = torch.squeeze(image_tensor.detach(), 0).permute(1, 2, 0).numpy()
    ax1.imshow(img)
    ax1.set_title('input')
    
    # Plot feature map
    feature = torch.mean(torch.squeeze(out_feature.detach(), 0), dim=2).numpy()
    im = ax2.imshow(feature, cmap='viridis')
    ax2.set_title('feature')
    
    # Add colorbar to feature map
    plt.colorbar(im, ax=ax2)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

    # print(model)

if __name__ == "__main__":
    main()
