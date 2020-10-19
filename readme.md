sudo apt install build-essential
sudo apt-get install python3-venv
sudo apt install python3-opencv

python3 -m venv venv
. venv/bin/activate

python3 -m pip install wheel
python3 -m pip install numpy pandas keras sklearn tensorflow imutils vidgear
python3 -m pip install --upgrade pip
python3 -m pip install opencv-python




sudo apt install ffmpeg
ffplay /dev/video0


python3 webstreaming.py --ip 0.0.0.0 --port 8000

python3 test2.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel