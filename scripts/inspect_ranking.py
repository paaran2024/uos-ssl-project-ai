import pickle
import numpy as np
import argparse
import os

"""
inspect_ranking.py: 가지치기(pruning) 중요도 점수 랭킹 파일을 검사하는 스크립트입니다.

사용법:
    이 스크립트는 `prune/head_pruning.py` 또는 `prune/vision_pruning.py`에서 생성된
    `.pkl` 랭킹 파일의 내용을 로드하여, 각 헤드 또는 뉴런의 중요도 점수를 확인하는 데 사용됩니다.
    점수가 0이 아닌 유의미한 값으로 계산되었는지 검증할 수 있습니다.

실행 방법 (ai/ 디렉토리에서):
    # 헤드 랭킹 파일 검사 (예시)
    python scripts/inspect_ranking.py --file ./storage/vision/DIV2K/CATANet-L/head_ranking_body.pkl

    # 뉴런 랭킹 파일 검사 (예시)
    python scripts/inspect_ranking.py --file ./storage/vision/DIV2K/CATANet-L/neuron_ranking_body.pkl

출력 내용:
    - 로드된 랭킹 파일의 경로
    - 총 랭킹 항목 수
    - 가장 덜 중요한 상위 5개 항목의 점수 및 상세 정보
    - 가장 중요한 상위 5개 항목의 점수 및 상세 정보

예상되는 정상적인 출력:
    스코어(Score) 값이 0이 아닌 음수 값으로 다양하게 나타나야 합니다.
    (스코어는 손실(loss)에 -1을 곱한 값이므로 음수가 됩니다. 절대값이 클수록 더 중요합니다.)
"""

def inspect_ranking_file(file_path):
    print(f"--- Inspecting Ranking File ---")
    print(f"Loading: {file_path}")

    try:
        with open(file_path, 'rb') as f:
            ranking = pickle.load(f)

        if not ranking:
            print("Error: Ranking file is empty.")
            return

        print(f"\nTotal items ranked: {len(ranking)}")

        # The list is typically sorted from least important (lowest score) to most important (highest score)
        # However, our current search algorithm expects MOST important first, so it reverses it.
        # Here we'll show original sorting from the file (least important first).

        print("\n--- LEAST Important (First 5) ---")
        for i in range(min(5, len(ranking))):
            score, *rest = ranking[i]
            print(f"Rank {i+1}: Score={score:.8f}, Details={rest}")

        print("\n--- MOST Important (Last 5) ---")
        for i in range(max(0, len(ranking) - 5), len(ranking)):
            score, *rest = ranking[i]
            print(f"Rank {i+1}: Score={score:.8f}, Details={rest}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please ensure the ranking has been generated.")
    except Exception as e:
        print(f"An error occurred: {e}")

    print("\n--- Inspection Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a pruning ranking .pkl file.")
    parser.add_argument("--file", required=True, help="Path to the ranking .pkl file (e.g., ./storage/vision/DIV2K/CATANet-L/head_ranking_body.pkl)")
    args = parser.parse_args()

    # Ensure the path is relative to the current working directory if not absolute
    full_path = os.path.join(os.getcwd(), args.file)
    if not os.path.exists(full_path):
        # Try relative to the script's location if the first try fails
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, args.file)

    inspect_ranking_file(full_path)
