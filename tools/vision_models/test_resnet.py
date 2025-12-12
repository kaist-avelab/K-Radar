import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import (
    # ResNet variants
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    # ResNeXt variants
    resnext50_32x4d, ResNeXt50_32X4D_Weights,
    resnext101_32x8d, ResNeXt101_32X8D_Weights,
    resnext101_64x4d, ResNeXt101_64X4D_Weights,
    # Swin variants
    swin_t, Swin_T_Weights,
    swin_s, Swin_S_Weights,
    swin_b, Swin_B_Weights,
    # MobileNet variants
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
)

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
    weights = ResNet101_Weights.DEFAULT
    model = resnet101(weights=weights)
    
    categories = weights.meta["categories"]
    
    # Case 1 (image)
    image_path = './tools/vision_models/car.jpg'
    image_tensor = load_and_preprocess_image(image_path)

    # Case 2 (K-Radar)
    image_path = './tools/vision_models/kradar_seq1.png'
    image_tensor = load_and_preprocess_image(image_path)
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    #                        std=[0.229, 0.224, 0.225])
    # ])
    # image = Image.open(image_path)
    # if image.mode != 'RGB':
    #     image = image.convert('RGB')
    
    predictions = get_prediction(model, image_tensor, categories)
    
    # 결과 출력
    print("\nTop 5 predictions:")
    for class_name, prob in predictions:
        print(f"{class_name}: {prob:.3f}")

    import torch.nn as nn
    feature_model = nn.Sequential(
        model.conv1,      # /2
        model.bn1,
        model.relu,
        model.maxpool,    # /2
        model.layer1,     
        model.layer2,     # /2
        # model.layer3,     # /2
    )

    out_feature = feature_model(image_tensor)

    print(image_tensor.shape)
    print(out_feature.shape)
    
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    img = torch.squeeze(image_tensor.detach(), 0).permute(1, 2, 0).numpy()
    ax1.imshow(img)
    ax1.set_title('input')
    
    # Plot feature map
    feature = torch.mean(torch.squeeze(out_feature.detach(), 0), dim=0).numpy()
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
