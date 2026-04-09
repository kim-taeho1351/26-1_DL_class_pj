import os
import sys
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def main():
    # 이미지 경로 설정
    # 이 파일(qwen_vlm.py)의 상위 폴더(dl_project)를 기준으로 data/test_img.jpg 찾기
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(base_dir, "data", "test_img.jpg")

    print(f"이미지 경로: {image_path}")

    if os.path.exists(image_path):
        sample_image = Image.open(image_path).convert("RGB")
        sample_image.thumbnail((512, 512))
        print("이미지를 성공적으로 불러왔습니다!")
        
        # 이미지 창 띄워서 확인
        sample_image.show() 
    else:
        print(f"오류: 이미지를 찾을 수 없습니다.")
        print("프로젝트 최상위의 'data' 폴더 안에 'test_img.jpg' 파일이 있는지 확인해 주세요.")
        sys.exit(1)

    # 모델과 프로세서 로드
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    print(f"\n[{model_id}] 모델 로딩")

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )

    # 장치 할당 (Device Agnostic)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    print(f"모델 로드 완료 (장치: {device})")
    print(f"파라미터 수: {model.num_parameters():,}\n")

    # 모델 프롬프트 입력 받기
    print("====================================")
    print(" 프롬프트를 입력해 주세요.")
    print("====================================\n")

    user_prompt = input("프롬프트: ")

    # 빈 입력 방지 (프로그램 안전 종료)
    if not user_prompt.strip():
        print("프롬프트가 입력되지 않아 프로그램을 종료합니다.")
        sys.exit(0)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample_image}, 
                {"type": "text", "text": user_prompt}, 
            ],
        }
    ]

    # 입력 데이터 전처리
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,    
        videos=video_inputs,    
        padding=True,
        return_tensors="pt"
    ).to(model.device) 

    # 추론(생성) 및 결과 출력
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            repetition_penalty=1.2,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    # 생성된 답변에서 질문 내용 잘라내기
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # 생성 토큰 디코딩
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0] 

    print("Qwen의 답변:")
    print("-" * 50)
    print(output_text)

# 이 스크립트가 직접 실행될 때만 main() 함수 호출
if __name__ == "__main__":
    main()