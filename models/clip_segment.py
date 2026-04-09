import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

def plot_results(image, prompts, masks):
    # 모델이 예측한 마스크(히트맵)를 원본 이미지 위에 겹쳐서 시각화하는 함수
    n_prompts = len(prompts)
    plt.figure(figsize=(n_prompts * 5, 5))

    for i in range(n_prompts):
        plt.subplot(1, n_prompts, i + 1)

        # 원본 이미지 배경으로 출력
        plt.imshow(image)

        # 모델이 생성한 마스크(히트맵)를 반투명하게 덮어쓰기
        # 마스크 값은 확률(Logits)이므로 sigmoid를 씌워 0~1 사이로 만들어 시각화
        mask_visual = torch.sigmoid(masks[i]).numpy()
        plt.imshow(mask_visual, cmap='jet', alpha=0.6)

        plt.title(f"Prompt: '{prompts[i]}'")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    # 이미지 경로
    # # 이 파일(clip_segment.py)의 상위 폴더(dl_project)를 기준으로 data/test_img.jpg 찾기
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(base_dir, "data", "test_img.jpg")

    print(f"이미지 경로: {image_path}")

    if os.path.exists(image_path):
        # 이미지 파일 열고 RGB로 변환 (오류 방지)
        sample_image = Image.open(image_path).convert("RGB")
        # 이미지 최대 해상도 제한 (비율 유지하면서 가로, 세로 최대 512 픽셀이 되도록 줄임)
        sample_image.thumbnail((512, 512))
        print("이미지를 성공적으로 불러왔습니다!")
    else:
        print(f"오류: 이미지를 찾을 수 없습니다.")
        print("프로젝트 최상위의 'data' 폴더 안에 'test_img.jpg' 파일이 있는지 확인해 주세요.")
        sys.exit(1)

    # 모델과 프로세서 로드
    model_id = "CIDAS/clipseg-rd64-refined"
    print(f"\n[{model_id}] 모델 로딩")

    processor = CLIPSegProcessor.from_pretrained(model_id)
    
    # 'torch_dtype'을 지정하여 메모리 사용량 절약
    model = CLIPSegForImageSegmentation.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )

    # 장치 할당 (Device Agnostic)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # 추론만 할 때는 드롭아웃 등을 꺼서 결과를 일정하게 유지
    model.eval()

    print(f"모델 로드 완료 (장치: {device})")
    print(f"파라미터 수: {model.num_parameters():,}\n")

    # Open-Vocabulary 분할 실행 (추론 준비)
    # 분할해 보고 싶은 텍스트 프롬프트 리스트 (영어 권장)
    prompts = ["a student", "a book", "a window"]
    print(f"분할 대상: {prompts}")

    # 통합 전처리
    inputs = processor(
        text=prompts,
        images=[sample_image] * len(prompts), # 텍스트 개수만큼 이미지를 복사해서 넣음
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # 모델 추론
    with torch.no_grad():
        outputs = model(**inputs)

    # 후처리 (결과 시각화)
    logits = outputs.logits
    if len(prompts) == 1:
        logits = logits.unsqueeze(0) # 프롬프트가 1개일 때 차원 맞춰주기

    # 원본 이미지 크기에 맞게 보간법(Interpolation)으로 마스크 크기 조정
    pred_masks = F.interpolate(
        logits.unsqueeze(1),
        size=(sample_image.size[1], sample_image.size[0]), # (Height, Width) 순서
        mode="bilinear", 
        align_corners=False
    ).squeeze(1).cpu()

    # 결과 화면 출력 (Matplotlib)
    print("처리가 완료되었습니다. 화면에 히트맵 창이 나타납니다.")
    plot_results(sample_image, prompts, pred_masks)

# 이 스크립트가 직접 실행될 때만 main() 함수 호출
if __name__ == "__main__":
    main()