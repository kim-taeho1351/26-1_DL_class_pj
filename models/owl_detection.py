import os
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

def main():
    # 이미지 경로 설정 (상대 경로 적용)
    # 이 파일(owl_detection.py)의 상위 폴더(dl_project)를 기준으로 data/test_img.jpg 찾기
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(base_dir, "data", "test_img.jpg")

    print(f"이미지 경로: {image_path}")

    if os.path.exists(image_path):
        # 이미지 파일 열고 RGB로 변환 (오류 방지)
        sample_image = Image.open(image_path).convert("RGB")
        # 메모리 및 속도 최적화를 위해 이미지 크기 제한 (OWL-v2는 내부적으로 크기를 자동 조절)
        sample_image.thumbnail((800, 800)) 
        print("이미지를 성공적으로 불러왔습니다!")
    else:
        print(f"오류: 이미지를 찾을 수 없습니다.")
        print("프로젝트 최상위의 'data' 폴더 안에 'test_img.jpg' 파일이 있는지 확인해 주세요.")
        sys.exit(1)

    # 모델과 전처리기 로드
    model_id = "google/owlv2-base-patch16-ensemble"

    processor = Owlv2Processor.from_pretrained(model_id)
    
    # 장치 할당 (Device Agnostic)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device)
    
    # 모델 상태를 '평가 모드'로 고정
    model.eval()

    print(f"모델 로드 완료 (장치: {device})")
    print(f"파라미터 수: {model.num_parameters():,}\n")

    # Open-Vocabulary 탐지 실행
    # 찾고 싶은 객체 입력
    # 텍스트는 이중 리스트 형태로 넣어야 함 (배치 처리 지원 때문)
    texts = [["a student", "a laptop", "a book", "a pen"]]
    print(f"검색 대상: {texts[0]}")

    # 전처리 (텍스트와 이미지를 텐서로 변환)
    inputs = processor(text=texts, images=sample_image, return_tensors="pt").to(device)

    # 모델 추론
    with torch.no_grad():
        outputs = model(**inputs)

    # 후처리
    # 임계값 설정 및 원본 이미지 크기에 맞게 박스 좌표 복원
    # threshold(임계값): 모델이 해당 객체일 확률이 몇 % 이상일 때만 화면에 그릴지 결정 (보통 0.1~0.2)
    target_sizes = torch.Tensor([sample_image.size[::-1]]) # (Height, Width) 형태로 변환
    results = processor.image_processor.post_process_object_detection(
        outputs=outputs, 
        target_sizes=target_sizes, 
        threshold=0.15
    )

    # 첫 번째 이미지의 결과만 추출
    res = results[0]

    # 결과 시각화 (Matplotlib 활용)
    print("처리가 완료되었습니다. 화면에 바운딩 박스 결과 창이 나타납니다.")
    print("-" * 50)
    
    fig, ax = plt.subplots(1, figsize=(12, 8)) # fig: 결과 이미지 저장 시 사용 가능
    ax.imshow(sample_image)

    # 프롬프트별로 박스 색상을 다르게 지정하기 위한 컬러맵
    colors = ['#FF3B30', '#007AFF', '#34C759', '#FF9500', '#AF52DE']

    # 모델이 찾아낸 박스, 점수, 라벨을 순회하며 그림
    for score, label_idx, box in zip(res["scores"], res["labels"], res["boxes"]):
        box = box.cpu().tolist()
        xmin, ymin, xmax, ymax = box
        label_text = texts[0][label_idx.item()]
        confidence = score.item()
        
        # 라벨에 맞는 색상 선택
        color = colors[label_idx.item() % len(colors)]

        # 사각형 테두리 그리기
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, 
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # 텍스트 라벨 달기 (라벨, 확률)
        ax.text(
            xmin, ymin - 5, 
            f"{label_text}: {confidence:.2f}", 
            color='white', fontsize=10, weight='bold',
            bbox=dict(facecolor=color, edgecolor='none', pad=2)
        )

    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 이 스크립트가 직접 실행될 때만 main() 함수 호출
if __name__ == "__main__":
    main()