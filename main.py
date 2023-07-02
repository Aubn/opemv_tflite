import sensor, image, time, os, tf, uos, gc
from pyb import UART
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((240, 240))
sensor.skip_frames(time=2000)
net = "trained.tflite"
labels = [1,2,5,6,7,8]
xred_threshold = (51, 84, -31, -3, -26, -2)
clock = time.clock()
uart = UART(3, 115200)
uart.init(115200, bits=8, parity=None, stop=1)
while(True):
    clock.tick()
    img = sensor.snapshot()
    img = img.negate()
    #img = img.binary([xred_threshold], invert=False, zero=True)
    img = img.negate()
    for obj in tf.classify(net, img,min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
        img.draw_rectangle(obj.rect())
        out = obj.output()
        max_idx = out.index(max(out))
        if max(out)>0.85:
            ans_0 = labels[max_idx]
            data = bytearray([ans_0])
            uart.write(data)
        else:
            ans_0 = 0
    print(ans_0)
    print(clock.fps(), "fps")
