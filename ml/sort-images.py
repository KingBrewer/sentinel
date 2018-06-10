import lightnet
import os

model = lightnet.load('yolo')

start_path = '/home/user/cctv-raw'
for path,dirs,files in os.walk(start_path):
    for filename in files:
        print(filename)
        filepath = os.path.join(path,filename)
        image = lightnet.Image.from_bytes(open(filepath, 'rb').read())
        boxes = model(image, thresh=0.5)
        for box in boxes:
            id, cls, prob, coords = box
            if cls == 'person':
                print("person: - {}".format(prob))
                os.rename(filepath, '/home/user/train/person/{}'.format(filename))
                break
        try:
            os.rename(filepath, '/home/user/train/not-person/{}'.format(filename))
            print("not person")
        except:
            pass
