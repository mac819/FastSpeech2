from pathlib import Path


PKG_PATH = Path(__file__).parent.parent.parent
DATA = PKG_PATH / 'data'

if __name__=="__main__":
    print(PKG_PATH)
    # Logic to read LJSPeech metadata and copy the audio files and
    # Text inside a filder with same file namea and wav and txt extension