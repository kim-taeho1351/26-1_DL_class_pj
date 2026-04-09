import os
import sys
import subprocess

def clear_screen():
    # 운영체제에 맞게 터미널 화면을 지워주는 함수 (깔끔한 UI용)
    os.system('cls' if os.name == 'nt' else 'clear')

def run_script(script_name):
    # 지정된 파이썬 스크립트를 독립된 프로세스로 실행
    # 현재 main.py가 있는 폴더를 기준으로 경로 생성
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, "models", script_name)
    
    if not os.path.exists(script_path):
        print(f"\n오류: {script_path} 파일을 찾을 수 없습니다.")
        return

    print(f"\n[{script_name}] 스크립트를 실행합니다.\n")
    print("-" * 50)
    
    # 스크립트 실행 (실행이 끝나면 VRAM이 자동으로 반환)
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError:
        print(f"\n스크립트 실행 중 오류가 발생하여 강제 종료되었습니다.")
    
    print("-" * 50)
    input("\n메인 메뉴로 돌아가려면 Enter를 누르세요.")

def main():
    while True:
        clear_screen()
        print("==================================================")
        print(" 🎓 DeepLearning Class Foundation Models Demo ")
        print("==================================================")
        print(" 1. Qwen2.5-VL")
        print(" 2. CLIPSeg")
        print(" 3. OWL-v2")
        # DINOv2 코드를 추가하셨다면 아래 주석을 해제하세요.
        # print(" 4. DINOv2 (시각적 특징 추출 및 대응점 매칭)")
        print(" 0. 프로그램 종료")
        print("==================================================")
        
        choice = input("실행할 모델의 번호를 입력하세요: ")
        
        if choice == '1':
            run_script("qwen_vlm.py")
        elif choice == '2':
            run_script("clip_segment.py")
        elif choice == '3':
            run_script("owl_detection.py")
        elif choice == '0':
            print("\n프로그램을 종료합니다. 감사합니다!\n")
            sys.exit(0)
        else:
            input("\n잘못된 입력입니다. Enter를 눌러 다시 시도하세요.")

if __name__ == "__main__":
    main()