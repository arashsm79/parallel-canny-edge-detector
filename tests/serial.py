import subprocess

commands = [
    './build/src/Main ./input/640x360.jpg ./output/640x360.jpg 50 250',
    './build/src/Main ./input/1280x720.jpg ./output/1280x720.jpg 30 90',
    './build/src/Main ./input/1920x1080.jpg ./output/1920x1080.jpg 150 200',
    './build/src/Main ./input/2560x1440.jpg ./output/2560x1440.jpg 100 180',
    './build/src/Main ./input/3840x2160.jpg ./output/3840x2160.jpg 10 30',
    './build/src/Main ./input/7680x4320.jpg ./output/7680x4320.jpg 60 150',
    './build/src/Main ./input/15360x8640.jpg ./output/15360x8640.jpg 0 20',
]

IT = 10

for command in commands:
    sum = 0.0
    for i in range(IT):
        res = subprocess.run(command, capture_output=True, shell=True, text=True)
        sum = sum + float(res.stdout)
    print(sum / IT)
