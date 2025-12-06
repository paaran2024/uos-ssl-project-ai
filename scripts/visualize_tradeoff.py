import os
import re
import subprocess
import csv
import matplotlib.pyplot as plt
import numpy as np

"""
visualize_tradeoff.py: mac_constraint 값에 따른 모델 성능(PSNR)과
                       연산량(FLOPs)의 관계를 시각화하는 스크립트입니다.

사용법:
    ai/ 디렉토리에서 아래 명령어를 실행하세요.
    python scripts/visualize_tradeoff.py

동작 과정:
    1. mac_constraint_values 리스트에 정의된 값들을 순회합니다.
    2. 각 값에 대해 config_catanet.yml 파일의 mac_constraint를 임시로 수정합니다.
    3. run_catanet_pruning.py 스크립트를 실행하고, 그 결과 출력을 캡처합니다.
    4. 출력에서 'FLOPs 비율'과 'Pruned Performance: PSNR' 값을 정규식을 사용해 추출합니다.
    5. 모든 실행이 끝나면, 수집된 결과를 tradeoff_results.csv 파일로 저장합니다.
    6. FLOPs 비율(x축)과 PSNR(y축)의 관계를 나타내는 그래프를 생성하여
       tradeoff_visualization.png 파일로 저장합니다.
    7. 스크립트 종료 후 config_catanet.yml 파일은 원래 상태로 복원됩니다.
"""

def read_config(config_path):
    """설정 파일을 읽어 그 내용을 반환합니다."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return f.read()

def write_config(config_path, content):
    """설정 파일에 내용을 씁니다."""
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)

def parse_output(output_text):
    """스크립트 출력에서 PSNR과 FLOPs 비율을 추출합니다."""
    try:
        psnr_match = re.search(r"Pruned Performance: PSNR=([\d.]+)", output_text)
        # 인코딩 문제로 '비율' 한글이 깨질 수 있으므로, 영어와 특수문자를 기준으로 검색합니다.
        flops_match = re.search(r"FLOPs \S+: ([\d.]+)%", output_text)
        
        psnr = float(psnr_match.group(1)) if psnr_match else None
        flops_ratio = float(flops_match.group(1)) if flops_match else None
        
        return psnr, flops_ratio
    except (AttributeError, ValueError) as e:
        print(f"출력 파싱 중 오류 발생: {e}")
        return None, None

def main():
    # --- 설정 ---
    # 이 스크립트는 'ai' 디렉토리에서 실행되어야 합니다.
    ai_dir = os.path.dirname(os.path.abspath(__file__))
    if not ai_dir.endswith('scripts'):
       ai_dir = '.' # ai 폴더에서 직접 실행 시
    else:
       ai_dir = os.path.dirname(ai_dir) # scripts 폴더에서 실행 시

    os.chdir(ai_dir)
    print(f"Working directory set to: {os.getcwd()}")


    config_path = 'config_catanet.yml'
    pruning_script = 'run_catanet_pruning.py'
    results_csv = 'tradeoff_results.csv'
    visualization_png = 'tradeoff_visualization.png'
    
    # 테스트할 mac_constraint 값들
    mac_constraint_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []

    print("--- 시작: 가지치기 성능 트레이드오프 분석 ---")

    # 원본 설정 파일 내용 저장
    original_config_content = read_config(config_path)

    try:
        for constraint in mac_constraint_values:
            print(f"\n--- mac_constraint = {constraint} 으로 테스트 시작 ---")
            
            # 1. 설정 파일 수정
            current_config = re.sub(
                r"mac_constraint:\s*[\d.]+",
                f"mac_constraint: {constraint}",
                original_config_content
            )
            write_config(config_path, current_config)
            print(f"'{config_path}'의 'mac_constraint'를 {constraint}(으)로 업데이트했습니다.")

            # 2. 프루닝 스크립트 실행
            print(f"'{pruning_script}' 실행 중...")
            process = subprocess.run(
                ['python', pruning_script, '--config', config_path],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            if process.returncode != 0:
                print(f"오류: mac_constraint = {constraint} 에서 스크립트 실행 실패.")
                print(process.stderr)
                continue

            # 3. 결과 파싱
            output = process.stdout
            psnr, flops_ratio = parse_output(output)
            
            if psnr is not None and flops_ratio is not None:
                print(f"성공: FLOPs 비율 = {flops_ratio}%, PSNR = {psnr:.4f}")
                results.append([constraint, flops_ratio, psnr])
            else:
                print("오류: 출력에서 결과(PSNR, FLOPs)를 찾지 못했습니다.")
                # 디버깅을 위해 전체 출력 표시
                print("--- 전체 출력 ---")
                print(output)
                print("-----------------")


    finally:
        # 4. 설정 파일 원상 복구
        print("\n--- 분석 완료. 원본 설정 파일로 복원합니다. ---")
        write_config(config_path, original_config_content)

    if not results:
        print("분석 결과가 없습니다. 스크립트를 종료합니다.")
        return

    # 5. CSV 파일로 결과 저장
    with open(results_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['mac_constraint', 'flops_ratio_percent', 'psnr_db'])
        # Write results sorted by mac_constraint for clarity in the CSV
        sorted_results = sorted(results, key=lambda x: x[0])
        writer.writerows(sorted_results)
    print(f"결과를 '{results_csv}' 파일에 저장했습니다.")

    # 6. 그래프 생성 및 저장
    results.sort(key=lambda x: x[0]) # mac_constraint 값으로 정렬
    constraints = [r[0] for r in results]
    flops_ratios = [r[1] for r in results]
    psnrs = [r[2] for r in results]

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # 왼쪽 Y축: PSNR
    color = 'tab:blue'
    ax1.set_xlabel('mac_constraint (유지할 연산량 비율)')
    ax1.set_ylabel('PSNR (dB)', color=color)
    ax1.plot(constraints, psnrs, 'o-', color=color, label='PSNR')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--')

    # 오른쪽 Y축: FLOPs 비율
    ax2 = ax1.twinx()  # ax1과 x축을 공유하는 두 번째 y축 생성
    color = 'tab:red'
    ax2.set_ylabel('FLOPs 비율 (%)', color=color)
    ax2.plot(constraints, flops_ratios, 's--', color=color, label='FLOPs Ratio')
    ax2.tick_params(axis='y', labelcolor=color)

    # 그래프 제목 및 범례
    plt.title('mac_constraint에 따른 PSNR 및 FLOPs 비율 변화')
    fig.tight_layout()  # 레이아웃 조정
    # 범례를 하나로 합치기
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right')
    
    plt.savefig(visualization_png)
    print(f"시각화 그래프를 '{visualization_png}' 파일에 저장했습니다.")

if __name__ == '__main__':
    main()
