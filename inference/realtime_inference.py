"""Real-time sign language recognition from webcam."""

from capture import main as capture_main


def main():
    print("Run live prediction pipeline (extend capture.py with model inference).")
    capture_main()


if __name__ == "__main__":
    main()
